"""HW2 cluster pruning（Manning et al., Introduction to IR, 7.1.6）。

建索引：隨機選 √N 篇文件當 leader，其餘每篇（follower）掛到最近的 b1 個 leader。
查詢：先找最近的 b2 個 leader，只在它們的 follower 中精確計算分數。
與 FAISS IVF 的差別在於 leader 是隨機選的、不做 k-means 訓練。
向量皆已 L2-normalized，「最近」以 inner product（= cosine）判斷。
"""
import numpy as np


class ClusterPruning:
    def __init__(self, vectors: np.ndarray, leaders: np.ndarray, followers: list[np.ndarray], b2: int):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.leaders = leaders.astype(np.int64)                 # leader 的文件編號
        self.followers = followers                              # 每個 leader 的 follower（含 leader 本身）
        self.b2 = b2

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    @classmethod
    def build(cls, vectors: np.ndarray, n_leaders: int | None, b1: int, b2: int, seed: int = 0) -> "ClusterPruning":
        n = len(vectors)
        n_leaders = n_leaders or max(1, round(np.sqrt(n)))
        rng = np.random.default_rng(seed)
        leaders = np.sort(rng.choice(n, size=n_leaders, replace=False))
        lv = vectors[leaders]
        assign = []
        for i in range(0, n, 4096):                             # 分批，避免 N × √N 的矩陣一次佔太多記憶體
            sims = vectors[i:i + 4096] @ lv.T
            assign.append(np.argsort(-sims, axis=1, kind="stable")[:, :b1])
        assign = np.concatenate(assign)                          # (N, b1)
        # leader 一定屬於自己的 cluster（隨機 leader 之間可能非常接近）
        assign[leaders, 0] = np.arange(n_leaders)
        docs = np.repeat(np.arange(n), b1)
        flat = assign.ravel()
        order = np.argsort(flat, kind="stable")
        bounds = np.searchsorted(flat[order], np.arange(n_leaders + 1))
        followers = [np.unique(docs[order[bounds[j]:bounds[j + 1]]]) for j in range(n_leaders)]
        return cls(vectors, leaders, followers, b2)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """與 faiss 相同的介面：(scores, ids)，形狀 (Q, k)；不足 k 個時 id 為 -1、分數為 -inf。"""
        queries = np.atleast_2d(queries).astype(np.float32)
        out_s = np.full((len(queries), k), -np.inf, dtype=np.float32)
        out_i = np.full((len(queries), k), -1, dtype=np.int64)
        lsims = queries @ self.vectors[self.leaders].T
        b2 = min(self.b2, len(self.leaders))
        for qi, q in enumerate(queries):
            near = np.argsort(-lsims[qi], kind="stable")[:b2]
            cand = np.unique(np.concatenate([self.followers[j] for j in near]))
            s = self.vectors[cand] @ q
            top = np.lexsort((cand, -s))[:k]
            out_s[qi, :len(top)], out_i[qi, :len(top)] = s[top], cand[top]
        return out_s, out_i

    def save(self, path: str) -> None:
        lens = np.array([len(f) for f in self.followers], dtype=np.int64)
        np.savez(path, vectors=self.vectors, leaders=self.leaders, lens=lens,
                 followers=np.concatenate(self.followers) if self.followers else np.zeros(0, np.int64), b2=self.b2)

    @classmethod
    def load(cls, path: str, b2: int | None = None) -> "ClusterPruning":
        z = np.load(path)
        followers = np.split(z["followers"], np.cumsum(z["lens"])[:-1])
        return cls(z["vectors"], z["leaders"], followers, int(z["b2"]) if b2 is None else b2)
