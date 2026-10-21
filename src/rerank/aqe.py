"""alpha-QE：每個模型各自以初次檢索的前 n 名擴展 query 向量。"""


def alpha_qe(query_emb: "np.ndarray", db_embs: "np.ndarray", top_ids: "np.ndarray", top_scores: "np.ndarray", n: int, alpha: float) -> "np.ndarray":
    ...
