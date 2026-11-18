"""Rocchio relevance feedback（在 embedding 空間）。

q' = normalize(α·q + β·mean(relevant) − γ·mean(non-relevant))

demo：使用者勾選結果中相關 / 不相關的圖。
評估：以 qrels 模擬使用者判斷前 feedback_depth 名（見 src/eval/protocol.py 的 simulate_feedback），
並用 frozen ranking 評估：已判斷的前幾名維持原位，只重排其後的結果，避免把「使用者已經看過的答案」算成進步。
"""
import numpy as np


def rocchio(query_emb: np.ndarray, db_embs: np.ndarray, rel_ids, nonrel_ids, alpha: float, beta: float, gamma: float) -> np.ndarray:
    q = alpha * query_emb
    rel_ids, nonrel_ids = np.asarray(rel_ids, dtype=np.int64), np.asarray(nonrel_ids, dtype=np.int64)
    if len(rel_ids):
        q = q + beta * db_embs[rel_ids].mean(0)
    if len(nonrel_ids):
        q = q - gamma * db_embs[nonrel_ids].mean(0)
    norm = np.linalg.norm(q)
    return (q / norm if norm > 0 else query_emb).astype(np.float32)
