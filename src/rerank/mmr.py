"""MMR（Maximal Marginal Relevance, Carbonell & Goldstein 1998）多樣化重排。

依序挑選 argmax_d [ λ·rel(d) − (1 − λ)·max_{s ∈ 已選} sim(d, s) ]。
rel 用前一階段的分數（先做 min-max 正規化，讓 RRF 等不同尺度的分數也能和 cosine 相比），
sim 用同一個模型的 embedding cosine。
query 有多種可能意圖時（the cat or the sofa?），MMR 讓前幾名涵蓋不同意圖。
"""
import numpy as np


def mmr(ids: np.ndarray, scores: np.ndarray, db_embs: np.ndarray, lam: float, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """回傳重排後的 (ids, scores)；只重排前 depth 名，其後維持原順序。分數為 MMR 選取時的值。"""
    ids, scores = np.asarray(ids), np.asarray(scores, dtype=np.float32)
    head, tail = ids[:depth], ids[depth:]
    n = len(head)
    if n <= 1 or lam >= 1.0:
        return ids, scores
    s = scores[:depth]
    rel = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones(n, dtype=np.float32)
    emb = db_embs[head]
    sim = emb @ emb.T
    chosen = [int(np.argmax(rel))]
    max_sim = sim[chosen[0]].copy()
    remaining = np.ones(n, dtype=bool)
    remaining[chosen[0]] = False
    picked_scores = [lam * rel[chosen[0]]]
    for _ in range(n - 1):
        val = np.where(remaining, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(val))
        chosen.append(j)
        picked_scores.append(val[j])
        remaining[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    order = np.asarray(chosen)
    new_ids = np.concatenate([head[order], tail])
    new_scores = np.concatenate([np.asarray(picked_scores, dtype=np.float32),
                                 np.full(len(tail), min(picked_scores) - 1, dtype=np.float32)])
    return new_ids, new_scores
