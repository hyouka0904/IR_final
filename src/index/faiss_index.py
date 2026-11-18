"""索引的建立、存檔、載入與搜尋：FAISS Flat / HNSW / IVF-PQ、HW2 cluster pruning、pHash 的二元索引。

向量索引全部用 inner product（embedding 已 L2-normalized，等同 cosine），且全部在 CPU 上，
因為 FAISS 的 HNSW 只支援 CPU，所有索引在同一個裝置上量延遲才公平。
IVF-PQ 的分群與 codebook 直接以圖庫本身訓練，不屬於模型訓練。
pHash 只有 Flat（IndexBinaryFlat，Hamming 距離），分數換算成 1 − hamming / bits。
"""
import os

import faiss
import numpy as np

from src.index.cluster_pruning import ClusterPruning
from src.models.phash import similarity

INDEX_KINDS = ("flat", "hnsw", "ivfpq", "cp")


def index_file(cache_root: str, gallery: str, rep: str, kind: str) -> str:
    ext = {"cp": ".npz"}.get(kind, ".index")
    return os.path.join(cache_root, "indexes", gallery, f"{rep}_{kind}{ext}")


def build_index(kind: str, db: np.ndarray, cfg: dict, seed: int = 0):
    if db.dtype == np.uint8:                      # pHash
        if kind != "flat":
            raise ValueError("pHash only supports the flat index")
        index = faiss.IndexBinaryFlat(db.shape[1] * 8)
        index.add(np.ascontiguousarray(db))
        return index
    x = np.ascontiguousarray(db, dtype=np.float32)
    d = x.shape[1]
    if kind == "cp":
        c = cfg["index"]["cp"]
        return ClusterPruning.build(x, c["n_leaders"], c["b1"], c["b2"], seed)
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


def set_search_params(index, kind: str, cfg: dict) -> None:
    # 搜尋參數不一定會跟著存檔，載入後一律重新設定
    if kind == "hnsw":
        index.hnsw.efSearch = cfg["index"]["hnsw"]["ef_search"]
    elif kind == "ivfpq":
        index.nprobe = cfg["index"]["ivfpq"]["nprobe"]
    elif kind == "cp":
        index.b2 = cfg["index"]["cp"]["b2"]


def save_index(index, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(index, ClusterPruning):
        index.save(path)
    elif isinstance(index, faiss.IndexBinary):
        faiss.write_index_binary(index, path)
    else:
        faiss.write_index(index, path)


def load_index(path: str, kind: str, cfg: dict, binary: bool = False):
    if kind == "cp":
        index = ClusterPruning.load(path)
    elif binary:
        index = faiss.read_index_binary(path)
    else:
        index = faiss.read_index(path)
    set_search_params(index, kind, cfg)
    return index


def search(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(Q, D) → (scores, ids)，形狀皆為 (Q, k)，分數越大越相似；不足 k 個時 id 為 -1。"""
    k = min(k, index.ntotal)
    if isinstance(index, ClusterPruning):
        return index.search(queries, k)
    if isinstance(index, faiss.IndexBinary):
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.uint8)
        dist, ids = index.search(q, k)
        return similarity(dist, index.d), ids
    q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
    if isinstance(index, faiss.IndexHNSW):
        # efSearch 必須 >= k，否則 HNSW 回傳的結果會少於 k 個
        index.hnsw.efSearch = max(index.hnsw.efSearch, k)
    return index.search(q, k)


def index_bytes(path: str) -> int:
    return os.path.getsize(path)
