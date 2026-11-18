"""HW1 inverted index：term → postings（依 doc id 排序的 (doc, tf)）。

另外保留 forward index（doc → {term: tf}），demo 用來說明「這張圖是因為哪些 tag 被找到」。
postings 以 numpy 陣列存放，BM25 可以整段向量化累加分數。
"""
import pickle

import numpy as np


class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # term → (doc ids int32, tf int32)
        self.doc_len = np.zeros(0, dtype=np.int32)
        self.forward: list[dict[str, int]] = []

    @classmethod
    def build(cls, docs: list[dict[str, int]]) -> "InvertedIndex":
        idx = cls()
        lists: dict[str, tuple[list[int], list[int]]] = {}
        for d, terms in enumerate(docs):
            for t, tf in terms.items():
                ids, tfs = lists.setdefault(t, ([], []))
                ids.append(d)                      # d 遞增，postings 自然有序
                tfs.append(tf)
        idx.postings = {t: (np.asarray(ids, dtype=np.int32), np.asarray(tfs, dtype=np.int32)) for t, (ids, tfs) in lists.items()}
        idx.doc_len = np.asarray([sum(terms.values()) for terms in docs], dtype=np.int32)
        idx.forward = [dict(terms) for terms in docs]
        return idx

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avg_doc_len(self) -> float:
        return float(self.doc_len.mean()) if self.n_docs else 0.0

    def df(self, term: str) -> int:
        p = self.postings.get(term)
        return 0 if p is None else len(p[0])

    def vocabulary(self) -> list[str]:
        return sorted(self.postings)

    # 布林檢索（HW1）：postings 有序，以合併方式求交集 / 聯集
    def boolean_and(self, terms: list[str]) -> np.ndarray:
        lists = sorted((self.postings.get(t, (np.zeros(0, np.int32),))[0] for t in terms), key=len)
        if not lists:
            return np.zeros(0, dtype=np.int32)
        out = lists[0]
        for p in lists[1:]:
            out = np.intersect1d(out, p, assume_unique=True)
        return out

    def boolean_or(self, terms: list[str]) -> np.ndarray:
        lists = [self.postings[t][0] for t in terms if t in self.postings]
        return np.unique(np.concatenate(lists)) if lists else np.zeros(0, dtype=np.int32)

    def matched_terms(self, query_terms: dict[str, int], doc: int) -> list[str]:
        """query 與某文件共有的 term，依 query tf 由高到低。"""
        doc_terms = self.forward[doc]
        return sorted((t for t in query_terms if t in doc_terms), key=lambda t: -query_terms[t])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"postings": self.postings, "doc_len": self.doc_len, "forward": self.forward}, f)

    @classmethod
    def load(cls, path: str) -> "InvertedIndex":
        with open(path, "rb") as f:
            d = pickle.load(f)
        idx = cls()
        idx.postings, idx.doc_len, idx.forward = d["postings"], d["doc_len"], d["forward"]
        return idx
