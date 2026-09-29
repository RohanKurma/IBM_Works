"""Embedding-based semantic search utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer

from .metrics import normalize_matrix


@dataclass
class SearchResult:
    rank: int
    document: str
    score: float


class SemanticSearchEngine:
    """Small in-memory semantic search engine for educational experiments."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
        metric: str = "cosine",
    ):
        self.model_name = model_name
        self.metric = metric
        self.model = SentenceTransformer(model_name)
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index(self, documents: Iterable[str]) -> None:
        self.documents = [doc.strip() for doc in documents if doc.strip()]
        if not self.documents:
            raise ValueError("At least one non-empty document is required.")
        self.embeddings = np.asarray(
            self.model.encode(self.documents, convert_to_numpy=True)
        )

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Index documents before searching.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = np.asarray(
            self.model.encode([query], convert_to_numpy=True)
        )

        if self.metric == "cosine":
            docs = normalize_matrix(self.embeddings)
            query_vec = normalize_matrix(query_embedding)[0]
            scores = docs @ query_vec
            order = np.argsort(-scores)

        elif self.metric == "dot":
            scores = self.embeddings @ query_embedding[0]
            order = np.argsort(-scores)

        elif self.metric == "euclidean":
            distances = np.linalg.norm(
                self.embeddings - query_embedding[0],
                axis=1,
            )
            scores = -distances
            order = np.argsort(distances)

        else:
            raise ValueError(
                "metric must be one of: cosine, dot, euclidean"
            )

        top_k = min(max(int(top_k), 1), len(self.documents))

        return [
            SearchResult(
                rank=rank,
                document=self.documents[idx],
                score=float(scores[idx]),
            )
            for rank, idx in enumerate(order[:top_k], start=1)
        ]
