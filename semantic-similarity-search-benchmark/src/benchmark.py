"""Simple benchmark utilities for comparing models and similarity metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
import pandas as pd

from .search import SemanticSearchEngine


@dataclass
class BenchmarkRow:
    model: str
    metric: str
    query: str
    expected_document: str
    top_result: str
    hit_at_1: int
    latency_ms: float


def evaluate_configuration(
    documents,
    cases,
    model_name: str,
    metric: str,
) -> pd.DataFrame:
    engine = SemanticSearchEngine(model_name=model_name, metric=metric)

    start = time.perf_counter()
    engine.index(documents)
    index_ms = (time.perf_counter() - start) * 1000

    rows = []
    for query, expected_document in cases:
        t0 = time.perf_counter()
        result = engine.search(query, top_k=1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        rows.append(
            BenchmarkRow(
                model=model_name,
                metric=metric,
                query=query,
                expected_document=expected_document,
                top_result=result.document,
                hit_at_1=int(result.document == expected_document),
                latency_ms=latency_ms,
            )
        )

    df = pd.DataFrame([asdict(row) for row in rows])
    df["index_latency_ms"] = index_ms
    return df
