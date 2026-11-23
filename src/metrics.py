"""Retrieval metrics. Values follow trec_eval conventions; nan marks a query
without relevant documents, which trec_eval would skip."""
import math
from collections.abc import Sequence

import numpy as np


def semantic_gains(query_cats: set[int], gallery_cats: list[set[int]]) -> np.ndarray:
    """gain_i = share of the query's categories that gallery image i also has."""
    if not query_cats:
        return np.zeros(len(gallery_cats), dtype=np.float64)
    n = len(query_cats)
    return np.array([len(query_cats & g) / n for g in gallery_cats], dtype=np.float64)


def _dcg(gains: Sequence[float]) -> float:
    return float(sum(g / math.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked_gains: Sequence[float], all_gains: Sequence[float], k: int) -> float:
    ideal = _dcg(sorted(all_gains, reverse=True)[:k])
    if ideal == 0:
        return float("nan")
    return _dcg(list(ranked_gains)[:k]) / ideal


def average_precision(ranked_rel: Sequence[bool], n_relevant: int) -> float:
    if n_relevant == 0:
        return float("nan")
    hits, total = 0, 0.0
    for i, rel in enumerate(ranked_rel):
        if rel:
            hits += 1
            total += hits / (i + 1)
    return total / n_relevant


def recall_at_k(ranked_ids: Sequence[int], target_id: int, k: int) -> float:
    return 1.0 if target_id in list(ranked_ids)[:k] else 0.0


def reciprocal_rank(ranked_ids: Sequence[int], target_id: int) -> float:
    for i, d in enumerate(ranked_ids):
        if d == target_id:
            return 1.0 / (i + 1)
    return 0.0


def mean_ignore_nan(values: Sequence[float]) -> tuple[float, int]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    skipped = int(arr.size - finite.size)
    return (float(finite.mean()) if finite.size else float("nan")), skipped
