"""Search indexes over gallery vectors, mapped to COCO image ids.

Kinds (README: Indexing): `flat` (exact, FAISS), `cluster` (HW2 cluster pruning, Manning et al.
IIR 7.1.6), `ivfpq` (FAISS IVF + product quantization) and `hnsw` (FAISS graph index). pHash codes
only get `flat` (binary Hamming). Search always runs on CPU (README: Efficiency); HNSW only exists
on CPU in FAISS, so this keeps the latency comparison fair. IVF-PQ is trained on the gallery
itself, which is index construction, not model training.

Every index has the same interface: ntotal, ids, metric, kind, add(), search(), save(), load().
"""
import json
import math
from pathlib import Path

import faiss
import numpy as np

from src.schema import INDEX_KINDS

METRICS = ("ip", "hamming")


class FlatIndex:
    kind = "flat"

    def __init__(self, index, ids: np.ndarray, metric: str):
        self.index = index
        self.ids = np.asarray(ids, dtype=np.int64)
        self.metric = metric

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    @classmethod
    def build(cls, vectors: np.ndarray, ids: np.ndarray, metric: str) -> "FlatIndex":
        if metric == "ip":
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            index = faiss.IndexFlatIP(vectors.shape[1])
        elif metric == "hamming":
            vectors = np.ascontiguousarray(vectors, dtype=np.uint8)
            index = faiss.IndexBinaryFlat(vectors.shape[1] * 8)
        else:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        index.add(vectors)
        return cls(index, ids, metric)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors (same dtype rules as build) under the given image ids."""
        ids = np.asarray(ids, dtype=np.int64)
        dtype = np.float32 if self.metric == "ip" else np.uint8
        vectors = np.ascontiguousarray(vectors, dtype=dtype)
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors differ in length")
        self.index.add(vectors)
        self.ids = np.concatenate([self.ids, ids])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, image ids), higher score = better; k is clipped to ntotal."""
        k = min(k, self.ntotal)
        if self.metric == "ip":
            scores, pos = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        else:
            dist, pos = self.index.search(np.ascontiguousarray(queries, dtype=np.uint8), k)
            scores = -dist.astype(np.float32)
        return scores, self.ids[pos]

    def vectors(self) -> np.ndarray:
        """Every stored vector in id order (float indexes only); used by the re-rankers."""
        if self.metric != "ip":
            raise ValueError("binary (hamming) indexes have no float vectors")
        return self.index.reconstruct_n(0, self.ntotal)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = str(directory / "index.faiss")
        if self.metric == "ip":
            faiss.write_index(self.index, path)
        else:
            faiss.write_index_binary(self.index, path)
        np.save(directory / "ids.npy", self.ids)
        (directory / "meta.json").write_text(json.dumps({"metric": self.metric}))

    @classmethod
    def load(cls, directory: Path) -> "FlatIndex":
        directory = Path(directory)
        metric = json.loads((directory / "meta.json").read_text())["metric"]
        path = str(directory / "index.faiss")
        index = faiss.read_index(path) if metric == "ip" else faiss.read_index_binary(path)
        return cls(index, np.load(directory / "ids.npy"), metric)


def _float32(vectors: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _check_ip(metric: str, kind: str) -> None:
    if metric != "ip":
        raise ValueError(f"the {kind} index needs float vectors (metric ip); "
                         f"pHash codes only support the flat index")


class FaissANNIndex:
    """IVF-PQ or HNSW over L2-normalized vectors, inner product (= cosine)."""
    metric = "ip"

    def __init__(self, index, ids: np.ndarray, kind: str, params: dict):
        self.index = index
        self.ids = np.asarray(ids, dtype=np.int64)
        self.kind = kind
        self.params = dict(params)
        self._apply_search_params()

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def _apply_search_params(self) -> None:
        # search-time parameters are not always stored in the file; set them after every load
        if self.kind == "hnsw":
            self.index.hnsw.efSearch = int(self.params["ef_search"])
        else:
            self.index.nprobe = int(self.params["nprobe"])

    @classmethod
    def build(cls, vectors: np.ndarray, ids: np.ndarray, kind: str, params: dict,
              metric: str = "ip") -> "FaissANNIndex":
        _check_ip(metric, kind)
        x = _float32(vectors)
        if len(ids) != len(x):
            raise ValueError("ids and vectors differ in length")
        n, d = x.shape
        params = dict(params)
        if kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, int(params["m"]), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(params["ef_construction"])
        elif kind == "ivfpq":
            m = int(params["m"])
            if d % m:
                raise ValueError(f"IVF-PQ m={m} must divide the vector dimension {d}")
            # nlist null = round(sqrt(N)), the same order as cluster pruning's leaders
            nlist = int(params.get("nlist") or max(1, round(math.sqrt(n))))
            nlist = min(nlist, n)
            # 2^nbits PQ centroids need at least that many training vectors
            nbits = min(int(params["nbits"]), max(1, int(math.log2(n))))
            index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, m, nbits,
                                     faiss.METRIC_INNER_PRODUCT)
            index.train(x)
            params.update(nlist=nlist, nbits=nbits)
        else:
            raise ValueError(f"FaissANNIndex kind must be ivfpq or hnsw, got {kind!r}")
        index.add(x)
        return cls(index, ids, kind, params)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        x = _float32(vectors)
        if len(ids) != len(x):
            raise ValueError("ids and vectors differ in length")
        self.index.add(x)
        self.ids = np.concatenate([self.ids, ids])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, self.ntotal)
        if self.kind == "hnsw":  # efSearch must be >= k or HNSW returns fewer than k results
            self.index.hnsw.efSearch = max(int(self.params["ef_search"]), k)
        scores, pos = self.index.search(_float32(queries), k)
        return _drop_missing(scores, pos, self.ids)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        np.save(directory / "ids.npy", self.ids)
        (directory / "meta.json").write_text(json.dumps(
            {"metric": "ip", "kind": self.kind, "params": self.params}))

    @classmethod
    def load(cls, directory: Path, params: dict | None = None) -> "FaissANNIndex":
        """params (from the config) override the stored search-time parameters."""
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        merged = {**meta["params"], **_search_params(meta["kind"], params or {})}
        return cls(faiss.read_index(str(directory / "index.faiss")), np.load(directory / "ids.npy"),
                   meta["kind"], merged)


def _search_params(kind: str, params: dict) -> dict:
    keys = {"hnsw": ("ef_search",), "ivfpq": ("nprobe",), "cluster": ("b2",)}[kind]
    return {k: params[k] for k in keys if k in params}


def _drop_missing(scores: np.ndarray, pos: np.ndarray, ids: np.ndarray):
    """ANN indexes may return fewer than k hits (position -1, always at the end of a row).
    A single query (how the engine searches) is cut to its real hits; with several queries the
    misses stay in place as id -1 and score -inf."""
    if (pos >= 0).all():
        return scores, ids[pos]
    if len(pos) == 1:
        n = int((pos[0] >= 0).sum())
        return scores[:, :n], ids[pos[:, :n]]
    return np.where(pos >= 0, scores, -np.inf).astype(np.float32), np.where(pos >= 0, ids[pos], -1)


class ClusterPruningIndex:
    """HW2 cluster pruning (IIR 7.1.6): √N random leaders; every vector (follower) is attached to
    its b1 nearest leaders; a query scores exactly only the followers of its b2 nearest leaders.
    Unlike FAISS IVF, leaders are sampled at random instead of trained with k-means."""
    metric = "ip"
    kind = "cluster"

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, leaders: np.ndarray,
                 followers: list[list[int]], params: dict):
        self._vectors = _float32(vectors)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.leaders = np.asarray(leaders, dtype=np.int64)  # rows of _vectors
        self.followers = [list(f) for f in followers]  # per leader, rows (leader included)
        self.params = dict(params)

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def _nearest_leaders(self, x: np.ndarray, b: int) -> np.ndarray:
        sims = x @ self._vectors[self.leaders].T
        return np.argsort(-sims, axis=1, kind="stable")[:, :b]

    @classmethod
    def build(cls, vectors: np.ndarray, ids: np.ndarray, params: dict, seed: int = 0,
              metric: str = "ip") -> "ClusterPruningIndex":
        _check_ip(metric, "cluster")
        x = _float32(vectors)
        if len(ids) != len(x):
            raise ValueError("ids and vectors differ in length")
        n = len(x)
        n_leaders = min(n, int(params.get("leaders") or max(1, round(math.sqrt(n)))))
        rng = np.random.default_rng(seed)
        leaders = np.sort(rng.choice(n, size=n_leaders, replace=False))
        idx = cls(x, ids, leaders, [[] for _ in range(n_leaders)], params)
        b1 = int(params["b1"])
        for start in range(0, n, 4096):  # batches keep the N x sqrt(N) similarity matrix small
            near = idx._nearest_leaders(x[start:start + 4096], b1)
            for row, ls in enumerate(near, start=start):
                for j in ls:
                    idx.followers[j].append(row)
        for j, row in enumerate(leaders):  # a leader always belongs to its own cluster
            if row not in idx.followers[j]:
                idx.followers[j].append(int(row))
        return idx

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """New vectors become followers of their b1 nearest leaders (the leaders stay fixed)."""
        x = _float32(vectors)
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) != len(x):
            raise ValueError("ids and vectors differ in length")
        start = self.ntotal
        self._vectors = np.concatenate([self._vectors, x])
        self.ids = np.concatenate([self.ids, ids])
        for row, ls in enumerate(self._nearest_leaders(x, int(self.params["b1"])), start=start):
            for j in ls:
                self.followers[j].append(row)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _float32(np.atleast_2d(queries))
        k = min(k, self.ntotal)
        b2 = min(int(self.params["b2"]), len(self.leaders))
        all_scores, all_rows = [], []
        for qi, near in enumerate(self._nearest_leaders(q, b2)):
            cand = np.unique(np.concatenate([np.asarray(self.followers[j], dtype=np.int64)
                                             for j in near]))
            s = self._vectors[cand] @ q[qi]
            top = np.lexsort((cand, -s))[:k]
            all_scores.append(s[top])
            all_rows.append(cand[top])
        n = min(len(r) for r in all_rows)  # candidates may be fewer than k
        scores = np.stack([s[:n] for s in all_scores]).astype(np.float32)
        rows = np.stack([r[:n] for r in all_rows])
        return scores, self.ids[rows]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        lens = np.array([len(f) for f in self.followers], dtype=np.int64)
        flat = np.concatenate([np.asarray(f, dtype=np.int64) for f in self.followers]) \
            if self.followers else np.zeros(0, dtype=np.int64)
        np.savez(directory / "cluster.npz", vectors=self._vectors, leaders=self.leaders,
                 lens=lens, followers=flat)
        np.save(directory / "ids.npy", self.ids)
        (directory / "meta.json").write_text(json.dumps(
            {"metric": "ip", "kind": "cluster", "params": self.params}))

    @classmethod
    def load(cls, directory: Path, params: dict | None = None) -> "ClusterPruningIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        z = np.load(directory / "cluster.npz")
        followers = np.split(z["followers"], np.cumsum(z["lens"])[:-1])
        merged = {**meta["params"], **_search_params("cluster", params or {})}
        return cls(z["vectors"], np.load(directory / "ids.npy"), z["leaders"],
                   [f.tolist() for f in followers], merged)


def build_index(kind: str, vectors: np.ndarray, ids: np.ndarray, metric: str, cfg: dict,
                seed: int = 0):
    """Build any index kind with its parameters from cfg['index']."""
    if kind == "flat":
        return FlatIndex.build(vectors, ids, metric)
    if kind == "cluster":
        return ClusterPruningIndex.build(vectors, ids, cfg["index"]["cluster"], seed, metric)
    if kind in ("ivfpq", "hnsw"):
        return FaissANNIndex.build(vectors, ids, kind, cfg["index"][kind], metric)
    raise ValueError(f"index kind must be one of {INDEX_KINDS}, got {kind!r}")


def load_index(directory: Path, cfg: dict | None = None):
    """Load any saved index; meta.json without a kind is a flat index (the week-1 layout)."""
    directory = Path(directory)
    kind = json.loads((directory / "meta.json").read_text()).get("kind", "flat")
    params = (cfg or {}).get("index", {}).get(kind)
    if kind == "flat":
        return FlatIndex.load(directory)
    if kind == "cluster":
        return ClusterPruningIndex.load(directory, params)
    return FaissANNIndex.load(directory, params)


def index_size_bytes(directory: Path) -> int:
    """Bytes on disk of one saved index (all files in its folder)."""
    return sum(p.stat().st_size for p in Path(directory).iterdir() if p.is_file())
