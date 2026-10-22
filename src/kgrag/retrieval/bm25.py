"""Okapi BM25 over the abstract corpus (small own implementation, no extra dependency)."""

from __future__ import annotations

import math
import re
from collections import Counter

from kgrag.kg import KnowledgeGraph
from kgrag.retrieval.base import Provenance, RetrievalResult, passages_to_facts

_TOKEN = re.compile(r"[a-z0-9]+(?:-[0-9]+)?")
STOPWORDS = frozenset(
    "a an and are as at be by did do does for from has have in is it of on or that the to was were "
    "what which with who whom".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = sum(self.len) / max(1, len(docs))
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        # BM25+ style floor keeps idf positive for very common terms.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.len, strict=True):
            s = 0.0
            for q in query:
                f = tf.get(q)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    s += self.idf[q] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out


class BM25Retriever:
    name = "bm25"

    def __init__(self, kg: KnowledgeGraph, provenance: Provenance | None = None) -> None:
        self.kg = kg
        self.prov = provenance or Provenance(kg)
        self.passages = kg.passages
        self.index = BM25Index([tokenize(p.text) for p in self.passages])

    def rank(self, question: str) -> list[tuple]:
        scores = self.index.scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(self.passages[i].iri, scores[i]) for i in order if scores[i] > 0]

    def retrieve(self, question: str, k: int = 20) -> RetrievalResult:
        return passages_to_facts(self.prov, self.rank(question), k, self.name)
