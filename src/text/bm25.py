"""HW2 BM25（Okapi），在 inverted index 上以 term-at-a-time 累加分數。

score(q, d) = Σ_t idf(t) · qtf'(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
idf(t)      = ln(1 + (N − df + 0.5) / (df + 0.5))       # 不會是負數（df > N/2 時仍為正）
qtf'(t)     = (k3+1)·qtf / (k3 + qtf)                   # query 端 tf 飽和；weighted tag 的 query 也有 tf

只回傳分數 > 0 的文件（沒有任何共同 term 的文件不算被檢索到），因此可能少於 k 個。
"""
import numpy as np

from src.text.inverted_index import InvertedIndex


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75, k3: float = 8.0):
        self.index, self.k1, self.b, self.k3 = index, k1, b, k3
        self._norm = k1 * (1 - b + b * index.doc_len / max(index.avg_doc_len, 1e-9))   # 每篇文件的分母常數

    def idf(self, term: str) -> float:
        n, df = self.index.n_docs, self.index.df(term)
        return float(np.log(1 + (n - df + 0.5) / (df + 0.5)))

    def score_all(self, query_terms: dict[str, int]) -> np.ndarray:
        scores = np.zeros(self.index.n_docs, dtype=np.float32)
        for t, qtf in query_terms.items():
            p = self.index.postings.get(t)
            if p is None:
                continue
            ids, tf = p
            qw = (self.k3 + 1) * qtf / (self.k3 + qtf)
            scores[ids] += self.idf(t) * qw * tf * (self.k1 + 1) / (tf + self._norm[ids])
        return scores

    def search(self, query_terms: dict[str, int], k: int) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (ids, scores)，依分數由高到低；同分時 doc id 小的在前。"""
        scores = self.score_all(query_terms)
        hit = np.flatnonzero(scores > 0)
        order = np.lexsort((hit, -scores[hit]))[:k]
        ids = hit[order]
        return ids.astype(np.int64), scores[ids]
