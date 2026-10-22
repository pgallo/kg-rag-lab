"""Answer generation behind a small ``LLM`` interface.

``StubLLM`` is deterministic and template based. It reads *only* the retrieved facts: it
links the question, selects a schema path (see ``analysis``) and walks that path inside the
retrieved triples. The ontology (class hierarchy) is treated as background knowledge, as a
reader would know that a carcinoma is a solid tumour; every instance fact it uses must come
from the context. If the chain is broken in the context, the answer is incomplete, which is
exactly what the ablation is meant to expose.

The hosted adapter is optional: it is only constructed when ``LLM_API_URL``, ``LLM_API_KEY``
and ``LLM_MODEL`` are all set. It uses only the standard library. Tests never need it.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from rdflib import URIRef

from kgrag.analysis import analyse
from kgrag.kg import KnowledgeGraph, Triple
from kgrag.linking import EntityLinker
from kgrag.retrieval.base import RetrievalResult
from kgrag.retrieval.graph import expand_seeds, follow_schema_path

NO_ANSWER = "No answer found in the retrieved context."


@dataclass
class Answer:
    text: str
    items: list[str]
    provenance: list[Triple] = field(default_factory=list)
    model: str = ""


class LLM(Protocol):
    name: str

    def answer(self, question: str, context: RetrievalResult) -> Answer: ...


def format_context(context: RetrievalResult) -> str:
    return "\n".join(f"[{i}] {f.text}" for i, f in enumerate(context.facts, start=1))


class StubLLM:
    name = "stub"

    def __init__(self, kg: KnowledgeGraph, linker: EntityLinker | None = None) -> None:
        self.kg = kg
        self.linker = linker or EntityLinker(kg)

    def answer(self, question: str, context: RetrievalResult) -> Answer:
        allowed = set(context.triples)
        links = self.linker.link(question)
        if not links:
            # Nothing linked against the whole graph: retry against entities in the context only.
            nodes = {n for s, _, o in allowed for n in (s, o)}
            links = self.linker.link_restricted(question, nodes)
        analysis = analyse(self.kg, self.linker, question, links)
        answers: set[URIRef] = set()
        used: list[Triple] = []
        for seed, path in analysis.paths:
            starts, _ = expand_seeds(self.kg, [seed])
            ends, triples = follow_schema_path(self.kg, starts, path, allowed)
            answers |= {e for e in ends if self.kg.type_matches(e, analysis.answer_type)}
            used += triples
        if not analysis.paths and analysis.answer_type is not None:
            # No schema path: fall back to typed neighbours of the seeds inside the context.
            seeds = set(analysis.seeds)
            for s, p, o in context.triples:
                for a, b in ((s, o), (o, s)):
                    if a in seeds and self.kg.type_matches(b, analysis.answer_type):
                        answers.add(b)
                        used.append((s, p, o))
        items = sorted(self.kg.label(a) for a in answers)
        if not items:
            return Answer(NO_ANSWER, [], [], self.name)
        path = f" (via {analysis.best_path})" if analysis.best_path else ""
        text = f"{', '.join(items)}.{path} Supported by {len(set(used))} retrieved fact(s)."
        return Answer(text, items, sorted(set(used)), self.name)


ANSWER_PROMPT = """You answer questions using only the numbered facts below.
Return JSON: {{"answers": [<entity names exactly as written in the facts>], "cited": [<fact numbers>]}}.
If the facts do not contain the answer, return {{"answers": [], "cited": []}}.

Facts:
{facts}

Question: {question}
JSON:"""


class HostedLLM:  # pragma: no cover - requires network and a configured endpoint
    """Reader backed by a hosted model behind a JSON chat-completions style endpoint.

    Configured only through ``LLM_API_URL``, ``LLM_API_KEY`` and ``LLM_MODEL``. The HTTP call
    uses the standard library, so no client package is required.
    """

    name = "hosted"

    def __init__(self, model: str | None = None, max_tokens: int = 512, timeout: float = 60.0) -> None:
        config = hosted_config()
        if config is None:
            raise RuntimeError(HOSTED_MISSING)
        self.url, self.api_key, default_model = config
        self.model = model or default_model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return extract_chat_text(data)

    def answer(self, question: str, context: RetrievalResult) -> Answer:
        raw = self.complete(ANSWER_PROMPT.format(facts=format_context(context), question=question))
        data = parse_json_object(raw)
        items = [str(a) for a in data.get("answers", [])]
        cited = [int(i) for i in data.get("cited", []) if str(i).isdigit()]
        prov = [context.facts[i - 1].triple for i in cited if 0 < i <= len(context.facts)]
        text = ", ".join(items) + "." if items else NO_ANSWER
        return Answer(text, items, prov, self.model)


def extract_chat_text(data: dict) -> str:
    """Text of the first choice in a chat-completions style response, or "" if absent."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def parse_json_object(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


HOSTED_ENV = ("LLM_API_URL", "LLM_API_KEY", "LLM_MODEL")
HOSTED_MISSING = "hosted LLM requested but LLM_API_URL, LLM_API_KEY and LLM_MODEL must all be set"


def hosted_config() -> tuple[str, str, str] | None:
    """(url, key, model) from the environment, or None if any of them is missing."""
    url, key, model = (os.environ.get(name, "").strip() for name in HOSTED_ENV)
    if not (url and key and model):
        return None
    return url, key, model


def hosted_available() -> bool:
    return hosted_config() is not None


def make_llm(kind: str, kg: KnowledgeGraph, linker: EntityLinker | None = None) -> LLM:
    """``kind``: "stub", "hosted", or "auto" (hosted if configured, else stub)."""
    if kind == "hosted" or (kind == "auto" and hosted_available()):
        if not hosted_available():
            raise RuntimeError(HOSTED_MISSING)
        return HostedLLM()
    return StubLLM(kg, linker)
