"""Ensemble：候選取聯集 → 用 cache/embeddings 精確重算 cosine → 加權融合。"""


def union_candidates(per_model_ids: dict[str, "np.ndarray"]) -> "np.ndarray":
    ...


def rescore(query_embs: dict[str, "np.ndarray"], db_embs: dict[str, "np.ndarray"], candidates: "np.ndarray") -> dict[str, "np.ndarray"]:
    ...


def fuse(scores: dict[str, "np.ndarray"], weights: dict[str, float], candidates: "np.ndarray", k: int) -> tuple["np.ndarray", "np.ndarray"]:
    ...
