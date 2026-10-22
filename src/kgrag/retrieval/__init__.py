"""Retrievers. All return ranked facts so they can be compared on the same budget."""

from kgrag.retrieval.base import Fact, Provenance, RetrievalResult, Retriever
from kgrag.retrieval.bm25 import BM25Retriever
from kgrag.retrieval.dense import DenseRetriever, HashingSVDEmbedder
from kgrag.retrieval.graph import GraphRetriever, GraphRetrieverConfig
from kgrag.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "Fact",
    "GraphRetriever",
    "GraphRetrieverConfig",
    "HashingSVDEmbedder",
    "HybridRetriever",
    "Provenance",
    "RetrievalResult",
    "Retriever",
    "reciprocal_rank_fusion",
]
