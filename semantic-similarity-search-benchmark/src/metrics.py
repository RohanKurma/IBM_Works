"""Manual and vectorized similarity/distance metrics."""

from __future__ import annotations

import math
import numpy as np


def euclidean_distance_manual(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2/Euclidean distance using the definition."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")
    squared_sum = sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))
    return math.sqrt(squared_sum)


def dot_product_manual(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a dot product using an explicit Python loop."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality.")
    return sum(float(x) * float(y) for x, y in zip(a, b))


def l2_norm_manual(a: np.ndarray) -> float:
    """Compute the L2 norm of one vector."""
    return math.sqrt(sum(float(x) ** 2 for x in a))


def normalize_manual(a: np.ndarray) -> np.ndarray:
    """Normalize one vector to unit length."""
    norm = l2_norm_manual(a)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return np.asarray(a, dtype=float) / norm


def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity from the dot product and vector norms."""
    denom = l2_norm_manual(a) * l2_norm_manual(b)
    if denom == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    return dot_product_manual(a, b) / denom


def pairwise_euclidean(embeddings: np.ndarray) -> np.ndarray:
    """Vectorized all-pairs Euclidean distance matrix."""
    diff = embeddings[:, None, :] - embeddings[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def pairwise_dot_product(embeddings: np.ndarray) -> np.ndarray:
    """Vectorized all-pairs dot-product similarity matrix."""
    return embeddings @ embeddings.T


def normalize_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Normalize each row vector to unit L2 norm."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize matrices containing zero vectors.")
    return embeddings / norms


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Vectorized all-pairs cosine-similarity matrix."""
    normalized = normalize_matrix(embeddings)
    return normalized @ normalized.T


def pairwise_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance = 1 - cosine similarity."""
    return 1.0 - pairwise_cosine_similarity(embeddings)
