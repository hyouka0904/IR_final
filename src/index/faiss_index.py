"""Flat / HNSW / IVF-PQ 的建立、存檔與載入。

全部使用 inner product（embedding 已 L2-normalized，等同 cosine），且全部在 CPU 上，
因為 FAISS 的 HNSW 只支援 CPU，三種索引放在同一個裝置上量延遲才公平。
IVF-PQ 的分群與 codebook 直接以圖庫本身訓練，不屬於模型訓練。
"""
import faiss
import numpy as np

INDEX_KINDS = ("flat", "hnsw", "ivfpq")


def build_index(kind: str, db_embs: np.ndarray, cfg: dict) -> faiss.Index:
    x = np.ascontiguousarray(db_embs, dtype=np.float32)
    d = x.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        c = cfg["index"]["hnsw"]
        index = faiss.IndexHNSWFlat(d, c["m"], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = c["ef_construction"]
    elif kind == "ivfpq":
        c = cfg["index"]["ivfpq"]
        if d % c["m"]:
            raise ValueError(f"IVF-PQ m={c['m']} must divide dim={d}")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, c["nlist"], c["m"], c["nbits"], faiss.METRIC_INNER_PRODUCT)
        index.train(x)
    else:
        raise ValueError(f"Unknown index kind: {kind}")
    index.add(x)
    set_search_params(index, kind, cfg)
    return index


def set_search_params(index: faiss.Index, kind: str, cfg: dict) -> None:
    # 搜尋參數不一定會跟著 .index 存檔，載入後一律重新設定
    if kind == "hnsw":
        index.hnsw.efSearch = cfg["index"]["hnsw"]["ef_search"]
    elif kind == "ivfpq":
        index.nprobe = cfg["index"]["ivfpq"]["nprobe"]


def save_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_index(path: str, kind: str, cfg: dict) -> faiss.Index:
    index = faiss.read_index(path)
    set_search_params(index, kind, cfg)
    return index


def search(index: faiss.Index, query_embs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """query_embs: (Q, D) → (scores, ids)，形狀皆為 (Q, k)；不足 k 個時 id 為 -1。"""
    q = np.ascontiguousarray(np.atleast_2d(query_embs), dtype=np.float32)
    if isinstance(index, faiss.IndexHNSW):
        # efSearch 必須 >= k，否則 HNSW 回傳的結果會少於 k 個
        index.hnsw.efSearch = max(index.hnsw.efSearch, k)
    return index.search(q, k)
