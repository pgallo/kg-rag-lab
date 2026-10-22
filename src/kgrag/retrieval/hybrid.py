"""Hybrid retrieval with reciprocal rank fusion (RRF).

RRF combines rankings, not scores: score(d) = sum_i w_i / (k + rank_i(d)). That avoids
calibrating BM25 scores, cosine similarities and PageRank mass against each other, and it is
robust when one retriever returns nothing (it simply contributes no votes).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence

from kgrag.kg import Triple
from kgrag.retrieval.base import Fact, RetrievalResult, Retriever


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[Hashable, float]]:
    weights = weights or [1.0] * len(rankings)
    scores: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, tuple[int, int]] = {}
    for r, (ranking, w) in enumerate(zip(rankings, weights, strict=True)):
        for rank, item in enumerate(ranking, start=1):
            scores[item] += w / (k + rank)
            first_seen.setdefault(item, (rank, r))
    # Ties broken by best individual rank, then by retriever order: fully deterministic.
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


WeightFn = Callable[[list[RetrievalResult]], list[float]]


def adaptive_weights(results: list[RetrievalResult]) -> list[float]:
    """Query-adaptive fusion weights, read from the graph retriever's own diagnostics.

    * no entity linked: the graph has nothing to say, rely on the text retrievers;
    * the question is compositional (three or more reasoning steps, counting a subclass
      expansion of a disease group as one step): text rankings mostly add one-hop
      distractors, so they get a small weight;
    * otherwise (entity linked, one relation): equal votes.
    The graph retriever is expected first in the list.
    """
    graph = results[0]
    n = len(results) - 1
    if not graph.seeds:
        return [0.0] + [1.0] * n
    path = graph.notes.get("schema_path", "")
    hops = (path.count("/") + 1 if path else 1) + (1 if graph.schema_facts else 0)
    if hops >= 3:
        return [1.0] + [0.2] * n
    return [1.0] + [0.5] * n


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        weights: Sequence[float] | WeightFn | None = None,
        rrf_k: int = 60,
        depth: int | None = None,
        name: str = "hybrid",
    ) -> None:
        self.name = name
        self.retrievers = list(retrievers)
        self.weights = weights if weights is not None else [1.0] * len(self.retrievers)
        self.rrf_k, self.depth = rrf_k, depth

    def retrieve(self, question: str, k: int = 20) -> RetrievalResult:
        # Fusing lists of length k (by default) keeps a long low-ranked tail from accumulating votes.
        results = [r.retrieve(question, k=self.depth or k) for r in self.retrievers]
        by_triple: dict[Triple, Fact] = {}
        for res in results:
            for f in res.facts:
                by_triple.setdefault(f.triple, f)
        weights = self.weights(results) if callable(self.weights) else list(self.weights)
        fused = reciprocal_rank_fusion([res.triples for res in results], k=self.rrf_k, weights=weights)
        facts = [Fact(t, score, by_triple[t].text, by_triple[t].sources) for t, score in fused[:k]]
        seeds = [s for res in results for s in res.seeds]
        notes = {f"{res.retriever}.{key}": v for res in results for key, v in res.notes.items()}
        passages = [p for res in results for p in res.passages]
        schema = [f for res in results for f in res.schema_facts]
        notes["fusion_weights"] = ", ".join(
            f"{r.name}={w:g}" for r, w in zip(self.retrievers, weights, strict=True)
        )
        return RetrievalResult(
            self.name, facts, passages=passages, seeds=seeds, notes=notes, schema_facts=schema
        )
