"""Evaluation metrics.

* evidence recall@k: fraction of the gold evidence chain (all instance triples on a gold
  schema path from the question entity to a gold answer) present in the top-k facts;
* answer-entity recall@k: fraction of gold answers that appear as a node in the top-k facts;
* exact match: predicted answer set equals the gold set (case and whitespace normalised);
* fuzzy F1: set F1 where items match if their token-sort ratio is >= 90 (absorbs small
  surface differences such as "Haematological malignancy" vs "hematological malignancy");
* faithfulness: fraction of the reader's provenance triples present in the retrieved context,
  and whether each answer is supported by at least one such triple.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rapidfuzz import fuzz
from rdflib import URIRef

from kgrag.analysis import SchemaPath, Step
from kgrag.kg import KnowledgeGraph, Triple


def evidence_chain(kg: KnowledgeGraph, start: URIRef, steps: tuple[Step, ...]) -> set[Triple]:
    from kgrag.retrieval.graph import expand_seeds, follow_schema_path

    starts, _ = expand_seeds(kg, [start])
    path = SchemaPath(start, start, steps, 0.0)
    _, triples = follow_schema_path(kg, starts, path)
    return set(triples)


def recall_at_k(gold: Iterable, retrieved: list, k: int) -> float:
    gold = set(gold)
    if not gold:
        return 1.0
    return len(gold & set(retrieved[:k])) / len(gold)


def answer_entity_recall(gold_iris: Iterable[URIRef], triples: list[Triple], k: int) -> float:
    nodes = {n for s, _, o in triples[:k] for n in (s, o)}
    gold = set(gold_iris)
    return len(gold & nodes) / len(gold) if gold else 1.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip("."))


def exact_match(pred: Iterable[str], gold: Iterable[str]) -> bool:
    return {_norm(p) for p in pred} == {_norm(g) for g in gold}


def fuzzy_f1(pred: list[str], gold: list[str], threshold: float = 90.0) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    unmatched = list(gold)
    tp = 0
    for p in pred:
        best = max(unmatched, key=lambda g: fuzz.token_sort_ratio(_norm(p), _norm(g)), default=None)
        if best is not None and fuzz.token_sort_ratio(_norm(p), _norm(best)) >= threshold:
            tp += 1
            unmatched.remove(best)
    precision, recall = tp / len(pred), tp / len(gold)
    return 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)


def faithfulness(provenance: list[Triple], context: list[Triple]) -> float:
    """Share of cited facts present in the retrieved context (1.0 when nothing is cited)."""
    if not provenance:
        return 1.0
    ctx = set(context)
    return sum(1 for t in provenance if t in ctx) / len(provenance)


def supported_answers(
    kg: KnowledgeGraph, items: list[str], provenance: list[Triple], context: list[Triple]
) -> float:
    """Share of answer items that occur in at least one cited triple that is in the context."""
    if not items:
        return 1.0
    ctx = set(context)
    labels = {kg.label(n).lower() for t in provenance if t in ctx for n in (t[0], t[2])}
    return sum(1 for i in items if i.lower() in labels) / len(items)
