"""Test fixtures: a tiny in-memory stand-in for a Chroma collection.

It mimics `collection.query(...)` and `collection.count()` with bag-of-words cosine
similarity and Chroma's `where` operators, so the search/RAG logic can be tested
without downloading an embedding model.
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from food_rag.data import build_document, build_metadata, load_food_data, unique_ids  # noqa: E402


def _vec(text):
    return Counter(re.findall(r"[a-z]+", text.lower()))


def _cosine(a, b):
    dot = sum(a[k] * b[k] for k in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def _matches(meta, where):
    if where is None:
        return True
    if "$and" in where:
        return all(_matches(meta, w) for w in where["$and"])
    (key, cond), = where.items()
    value = meta.get(key)
    if not isinstance(cond, dict):
        return value == cond
    (op, target), = cond.items()
    return {
        "$in": lambda: value in target,
        "$lte": lambda: value <= target,
        "$gte": lambda: value >= target,
        "$eq": lambda: value == target,
    }[op]()


class FakeCollection:
    def __init__(self, foods):
        self.ids = unique_ids(foods)
        self.docs = [build_document(f) for f in foods]
        self.metas = [build_metadata(f) for f in foods]
        self.vecs = [_vec(d) for d in self.docs]
        self.last_query = None

    def count(self):
        return len(self.ids)

    def query(self, query_texts, n_results, where=None):
        self.last_query = {"n_results": n_results, "where": where}
        q = _vec(query_texts[0])
        scored = [(1 - _cosine(q, v), i) for i, v in enumerate(self.vecs) if _matches(self.metas[i], where)]
        scored.sort()
        top = scored[:n_results]
        return {
            "ids": [[self.ids[i] for _, i in top]],
            "metadatas": [[self.metas[i] for _, i in top]],
            "distances": [[d for d, _ in top]],
            "documents": [[self.docs[i] for _, i in top]],
        }


@pytest.fixture(scope="session")
def foods():
    return load_food_data(ROOT / "data" / "FoodDataSet.json")


@pytest.fixture()
def collection(foods):
    return FakeCollection(foods)


@pytest.fixture(scope="session")
def cuisines(foods):
    return sorted({f["cuisine_type"] for f in foods})
