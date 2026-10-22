"""Wiring: build the graph, the linker, every retriever and the reader once, and share them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kgrag.kg import KnowledgeGraph
from kgrag.linking import EntityLinker
from kgrag.llm import LLM, Answer, make_llm
from kgrag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    GraphRetriever,
    HybridRetriever,
    Provenance,
    RetrievalResult,
    Retriever,
)
from kgrag.retrieval.dense import make_embedder
from kgrag.retrieval.hybrid import adaptive_weights

RETRIEVERS = ("bm25", "dense", "graph", "hybrid-uniform", "hybrid")


@dataclass
class Pipeline:
    kg: KnowledgeGraph
    linker: EntityLinker
    retrievers: dict[str, Retriever]
    llm: LLM

    @classmethod
    def build(
        cls, kg: KnowledgeGraph | None = None, llm: str = "stub", embedder: str = "hashing"
    ) -> Pipeline:
        kg = kg or KnowledgeGraph.synthetic()
        linker = EntityLinker(kg)
        prov = Provenance(kg)
        bm25 = BM25Retriever(kg, prov)
        dense = DenseRetriever(kg, make_embedder(embedder), prov)
        graph = GraphRetriever(kg, linker, prov)
        # Graph first: adaptive_weights reads its diagnostics (linked seeds, schema path length).
        hybrid = HybridRetriever([graph, bm25, dense], weights=adaptive_weights)
        uniform = HybridRetriever([graph, bm25, dense], name="hybrid-uniform")  # plain RRF baseline
        retrievers: dict[str, Retriever] = {
            "bm25": bm25,
            "dense": dense,
            "graph": graph,
            "hybrid-uniform": uniform,
            "hybrid": hybrid,
        }
        return cls(kg, linker, retrievers, make_llm(llm, kg, linker))

    @classmethod
    def from_path(cls, path: str | Path | None, **kwargs) -> Pipeline:
        kg = KnowledgeGraph.load(path) if path and Path(path).exists() else KnowledgeGraph.synthetic()
        return cls.build(kg, **kwargs)

    def ask(self, question: str, retriever: str = "hybrid", k: int = 20) -> tuple[RetrievalResult, Answer]:
        context = self.retrievers[retriever].retrieve(question, k=k)
        return context, self.llm.answer(question, context)
