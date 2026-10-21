"""Flat / HNSW / IVF-PQ 的建立、存檔與載入。"""


def build_index(kind: str, db_embs: "np.ndarray", cfg: dict):
    ...


def save_index(index, path: str) -> None:
    ...


def load_index(path: str):
    ...


def search(index, query_embs: "np.ndarray", k: int) -> tuple["np.ndarray", "np.ndarray"]:
    ...
