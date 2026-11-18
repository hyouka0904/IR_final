"""評估指標：與 trec_eval 定義一致的 nDCG@10、AP、Recall@k、RR，以及 recall 損失、kappa、分階段計時。

為什麼不直接呼叫 trec_eval：semantic 的 qrels 由共享類別推得，展開後會有上千萬列
（person 類別出現在將近一半的 COCO 圖片中），因此在 numpy 上直接算。
定義與 trec_eval 9.x 相同（tests/test_metrics.py 以 pytrec_eval 驗證）：
  - relevant = 等級 >= 1；R = qrels 中 relevant 的總數（不只是被檢索到的）
  - map          = Σ_{第 i 名 relevant} P@i / R
  - ndcg_cut_10  = DCG@10 / IDCG@10，gain = 等級本身、discount = log2(rank + 1)，IDCG 由全部 qrels 排出
  - recall_k     = 前 k 名中的 relevant 數 / R
  - recip_rank   = 1 / 第一個 relevant 的名次（沒有則為 0）
  - R = 0 的 query 不計入平均（trec_eval 只評估出現在 qrels 中的 query）
"""
import time
from contextlib import contextmanager

import numpy as np

METRICS = ("ndcg_cut_10", "map", "recall_1", "recall_10", "recip_rank")


def query_metrics(ranking: np.ndarray, grades: np.ndarray) -> dict[str, float]:
    """ranking：圖庫索引（依名次）；grades：整個圖庫的等級陣列。"""
    ranking = np.asarray(ranking, dtype=np.int64)
    n_rel = int((grades >= 1).sum())
    if n_rel == 0:
        return {m: np.nan for m in METRICS}
    g = grades[ranking].astype(np.float64)
    rel = g >= 1
    hits = np.cumsum(rel)
    ranks = np.arange(1, len(ranking) + 1)
    ap = float((hits[rel] / ranks[rel]).sum() / n_rel)
    disc = 1.0 / np.log2(np.arange(2, 12))
    dcg = float((g[:10] * disc[:len(g[:10])]).sum())
    ideal = np.sort(grades[grades >= 1])[::-1][:10].astype(np.float64)
    idcg = float((ideal * disc[:len(ideal)]).sum())
    first = np.flatnonzero(rel)
    return {
        "ndcg_cut_10": dcg / idcg,
        "map": ap,
        "recall_1": float(rel[:1].sum() / n_rel),
        "recall_10": float(rel[:10].sum() / n_rel),
        "recip_rank": float(1.0 / (first[0] + 1)) if len(first) else 0.0,
    }


def mean_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for m in METRICS:
        vals = np.array([q[m] for q in per_query], dtype=np.float64)
        out[m] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return out


def category_coverage(ranking: np.ndarray, q_cats: np.ndarray, g_cats: np.ndarray, k: int = 10) -> float:
    """前 k 名涵蓋了 query 的幾成類別。衡量 MMR 是否讓結果涵蓋多種意圖（nDCG 看不出多樣性）。"""
    need = q_cats.astype(bool)
    if not need.any():
        return np.nan
    got = g_cats[np.asarray(ranking[:k], dtype=np.int64)].astype(bool).any(0)
    return float((got & need).sum() / need.sum())


def recall_loss(ann_ids: list[np.ndarray], flat_ids: list[np.ndarray]) -> float:
    """1 − mean_q |ANN top-K ∩ Flat top-K| / |Flat top-K|。"""
    overlaps = [len(np.intersect1d(a[a >= 0], f)) / len(f) for a, f in zip(ann_ids, flat_ids) if len(f)]
    return 1.0 - float(np.mean(overlaps)) if overlaps else np.nan


def cohen_kappa(a, b, weights: str | None = None) -> float:
    """兩位評審的一致性。weights=None 為一般 Cohen's kappa；"linear" 為線性加權（適合 0/1/2 分級）。"""
    a, b = np.asarray(a), np.asarray(b)
    labels = np.union1d(a, b)
    n = len(labels)
    pos = {v: i for i, v in enumerate(labels)}
    obs = np.zeros((n, n))
    for x, y in zip(a, b):
        obs[pos[x], pos[y]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    if weights is None:
        w = 1 - np.eye(n)
    elif weights == "linear":
        idx = np.arange(n)
        w = np.abs(idx[:, None] - idx[None, :]) / max(n - 1, 1)
    else:
        raise ValueError(weights)
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)


class StageTimer:
    """分階段累計耗時（ms）；同一階段可進出多次。"""

    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t) * 1000)

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + ms

    def as_dict(self) -> dict[str, float]:
        return dict(self.ms)
