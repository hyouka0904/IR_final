"""Reciprocal Rank Fusion（Cormack et al. 2009）：融合向量檢索與 tag BM25 的排序。

score(d) = Σ_r 1 / (k + rank_r(d))，rank 從 1 起算；沒出現在某個排序中的文件，該排序不貢獻分數。
只用名次、不用原始分數，因此 cosine 與 BM25 的尺度不同也能直接融合。
"""
import numpy as np


def rrf(rankings: list[np.ndarray], k: int = 60, depth: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for r, d in enumerate(np.asarray(ranking)[:depth]):
            d = int(d)
            if d >= 0:
                fused[d] = fused.get(d, 0.0) + 1.0 / (k + r + 1)
    if not fused:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    ids = np.fromiter(fused, dtype=np.int64)
    scores = np.fromiter(fused.values(), dtype=np.float64)
    order = np.lexsort((ids, -scores))
    return ids[order], scores[order].astype(np.float32)
