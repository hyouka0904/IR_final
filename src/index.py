"""FAISS Flat (exact) index over gallery vectors, mapped to COCO image ids.
Search always runs on CPU (README: Efficiency)."""
import json
from pathlib import Path

import faiss
import numpy as np

METRICS = ("ip", "hamming")


class FlatIndex:
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
