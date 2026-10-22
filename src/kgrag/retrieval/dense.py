"""Dense retrieval with a pluggable embedder.

The default embedder is fully offline and deterministic: hashed word and character n-gram
TF-IDF features reduced with truncated SVD (latent semantic analysis). It needs no model
download, so tests and CI are reproducible. If the optional ``embeddings`` extra is installed,
a neural sentence embedder can be selected through ``make_embedder``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from kgrag.kg import KnowledgeGraph
from kgrag.retrieval.base import Provenance, RetrievalResult, passages_to_facts


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> Embedder: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingSVDEmbedder:
    def __init__(self, dim: int = 96, n_features: int = 2**16, seed: int = 0) -> None:
        self.words = HashingVectorizer(
            n_features=n_features, ngram_range=(1, 2), alternate_sign=False, norm=None, stop_words="english"
        )
        self.chars = HashingVectorizer(
            n_features=n_features, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm=None
        )
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.dim, self.seed = dim, seed
        self.svd: TruncatedSVD | None = None

    def _features(self, texts: list[str]):
        return hstack([self.words.transform(texts), self.chars.transform(texts)]).tocsr()

    def fit(self, texts: list[str]) -> HashingSVDEmbedder:
        x = self.tfidf.fit_transform(self._features(texts))
        dim = min(self.dim, x.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=7, random_state=self.seed)
        self.svd.fit(x)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("call fit() first")
        x = self.tfidf.transform(self._features(texts))
        return normalize(self.svd.transform(x))


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


def make_embedder(kind: str = "hashing") -> Embedder:
    if kind == "hashing":
        return HashingSVDEmbedder()
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")


class DenseRetriever:
    name = "dense"

    def __init__(
        self, kg: KnowledgeGraph, embedder: Embedder | None = None, provenance: Provenance | None = None
    ) -> None:
        self.kg = kg
        self.prov = provenance or Provenance(kg)
        self.passages = kg.passages
        texts = [p.text for p in self.passages]
        self.embedder = (embedder or HashingSVDEmbedder()).fit(texts)
        self.matrix = self.embedder.encode(texts)

    def rank(self, question: str) -> list[tuple]:
        q = self.embedder.encode([question])[0]
        sims = self.matrix @ q
        order = np.lexsort((np.arange(len(sims)), -sims))
        return [(self.passages[i].iri, float(sims[i])) for i in order]

    def retrieve(self, question: str, k: int = 20) -> RetrievalResult:
        return passages_to_facts(self.prov, self.rank(question), k, self.name)
