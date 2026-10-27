"""Ensemble：候選取聯集 → 用 cache/embeddings 精確重算 cosine → 加權融合。

先取聯集再精確重算，是為了避免某個候選只出現在一個模型的索引結果中、在其他模型沒有分數。
"""
import numpy as np


def union_candidates(per_model_ids: dict[str, np.ndarray]) -> np.ndarray:
    ids = np.concatenate([np.asarray(v).ravel() for v in per_model_ids.values()])
    return np.unique(ids[ids >= 0])     # IVF-PQ 不足 K 個時會回傳 -1


def rescore(query_embs: dict[str, np.ndarray], db_embs: dict[str, np.ndarray], candidates: np.ndarray) -> dict[str, np.ndarray]:
    """單一 query：每個模型對每個候選的精確 cosine 分數（向量皆已 L2-normalized）。"""
    return {m: db_embs[m][candidates] @ q for m, q in query_embs.items()}


def fuse(scores: dict[str, np.ndarray], weights: dict[str, float], candidates: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    fused = sum(weights[m] * s for m, s in scores.items())
    order = np.argsort(-fused, kind="stable")[:k]
    return candidates[order], fused[order]
