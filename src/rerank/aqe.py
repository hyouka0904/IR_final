"""alpha-QE：每個模型各自以初次檢索的前 n 名擴展 query 向量。

q' = normalize(q + sum_{i<n} max(s_i, 0)^alpha * x_i)，s_i 為 q 與第 i 名的 cosine。
分數用精確 embedding 重算，避免 IVF-PQ 的近似分數影響權重。
"""
import numpy as np


def alpha_qe(query_emb: np.ndarray, db_embs: np.ndarray, top_ids: np.ndarray, n: int, alpha: float) -> np.ndarray:
    ids = top_ids[top_ids >= 0][:n]
    neighbors = db_embs[ids]
    weights = np.clip(neighbors @ query_emb, 0, None) ** alpha
    q = query_emb + weights @ neighbors
    return (q / np.linalg.norm(q)).astype(np.float32)
