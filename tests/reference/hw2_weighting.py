# Vendored unchanged from team27_hw2/weighting.py (IR HW2, team 27) for the BM25 parity test only.
# Do not edit below this line.
"""SMART weighting scheme (ddd.qqq) and Okapi BM25 scoring engine.

900 combinations of SMART notation on both document and query, alongside parameterized Okapi BM25:
5 term-frequency variants ('n', 'l', 'a', 'b', 'L')
x 3 inverse document frequency variants ('n', 't', 'p')
x 2 normalization modes ('n', 'c')
Lazily loads precomputed per-document statistics and vector norms from doc_stats.tsv when required by the active scheme.
"""

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

# SMART letters that are implemented (IIR Fig. 6.15)
TF_LETTERS = "nlabL"
IDF_LETTERS = "ntp"
NORM_LETTERS = "nc"
DEFAULT_SCHEME = "ltc.ltc"
# BM25 defaults (the values BM25 champion lists were built with)
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
# per-document statistics and vector lengths written by prep_extra.py
DOC_STATS_FILE = "doc_stats.tsv"

HELP = """weighting scheme: SMART notation ddd.qqq (document.query) or bm25.
  tf   n natural   l 1+log tf   a 0.5+0.5 tf/max tf   b boolean   L (1+log tf)/(1+log avg tf)
  idf  n none      t log(N/df)  p max(0, log((N-df)/df))
  norm n none      c cosine
  examples: ltc.ltc (default, Slide 4), lnc.ltc (textbook), nnc.ntc, anc.atc, Lpc.ltc, bm25"""


# raised for a scheme name that cannot be understood
class SchemeError(ValueError):
    pass


class Scheme:
    """Encapsulates document weighting, query weighting, and vector normalization for a scheme.

    Computes score(q, d) = sum_t wq(t) * wd(t, d) / (query_norm * doc_norm(d)).
    With 'ltc.ltc', reuses IndexReader.doc_norms directly without loading doc_stats.tsv.

    Attributes:
        idx (IndexReader): Underlying index reader.
        spec (str): Canonical scheme specification ('ddd.qqq' or 'bm25').
        is_bm25 (bool): True if the scheme is Okapi BM25.
        k1 (float): BM25 tf saturation constant (default 1.2).
        b (float): BM25 document length normalization constant (default 0.75).
    """

    def __init__(self, idx, spec: str = DEFAULT_SCHEME, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        # index access (N, df, log base)
        self.idx = idx
        # name as typed
        self.spec = spec
        self.is_bm25 = spec.lower() == "bm25"
        self.k1, self.b = k1, b
        if self.is_bm25:
            self.spec = "bm25"
            # BM25 has no vector normalization on either side
            self.doc, self.query = "bm25", "bm25"
        else:
            parts = spec.split(".")
            # "lnc" alone means the same letters on both sides
            if len(parts) == 1:
                parts = parts * 2
            if len(parts) != 2 or any(len(p) != 3 for p in parts):
                raise SchemeError(f"scheme {spec!r} is not of the form ddd.qqq or bm25")
            for p in parts:
                if p[0] not in TF_LETTERS or p[1] not in IDF_LETTERS or p[2] not in NORM_LETTERS:
                    raise SchemeError(f"scheme {spec!r}: tf must be one of {TF_LETTERS}, idf one of {IDF_LETTERS}, "
                                      f"normalization one of {NORM_LETTERS}")
            self.doc, self.query = parts
            self.spec = ".".join(parts)
        # True for the Slide 4 formula
        self.is_default = self.spec == DEFAULT_SCHEME
        # per-document tables, loaded only when a scheme needs them
        self._max_tf: Optional[List[int]] = None
        self._avg_tf: Optional[List[float]] = None
        self._norms: Optional[List[float]] = None
        self._bm_k: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # per-document tables
    # ------------------------------------------------------------------

    # read the wanted columns of doc_stats.tsv: column name -> list indexed by doc_id
    def _load_stats(self, columns: List[str]) -> Dict[str, List[float]]:
        path = os.path.join(self.idx.index_dir, DOC_STATS_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; build it with 'python prep_extra.py' first")
        out = {c: [0.0] * (self.idx.N + 1) for c in columns}
        with open(path, encoding="utf-8", newline="\n") as f:
            header = f.readline().rstrip("\n").split("\t")
            where = {c: header.index(c) for c in columns}
            for line in f:
                cells = line.rstrip("\n").split("\t")
                d = int(cells[0])
                for c, k in where.items():
                    out[c][d] = float(cells[k])
        return out

    # max tf of every document (augmented tf)
    def _doc_max_tf(self) -> List[float]:
        if self._max_tf is None:
            self._max_tf = self._load_stats(["max_tf"])["max_tf"]
        return self._max_tf

    # average tf of every document = |d| / number of distinct terms (log-average tf)
    def _doc_avg_tf(self) -> List[float]:
        if self._avg_tf is None:
            t = self._load_stats(["length", "distinct"])
            self._avg_tf = [(l / u if u else 0.0) for l, u in zip(t["length"], t["distinct"])]
        return self._avg_tf

    # ||d|| of every document under the document side of this scheme (all 1 when it is not cosine-normalized)
    def doc_norms(self) -> List[float]:
        if self._norms is None:
            if self.is_bm25 or self.doc[2] == "n":
                self._norms = [1.0] * (self.idx.N + 1)
            # Slide 4 weights: the table the index reader already holds
            elif self.doc[:2] == "lt":
                self._norms = self.idx.doc_norms
            else:
                self._norms = self._load_stats(["norm_" + self.doc[:2]])["norm_" + self.doc[:2]]
        return self._norms

    # BM25 denominator part k1 (1 - b + b |d| / avgdl) of every document
    def bm25_k(self) -> List[float]:
        if self._bm_k is None:
            avgdl = self.idx.meta["avgdl"]
            k = [self.k1 * (1.0 - self.b)] * (self.idx.N + 1)
            with open(os.path.join(self.idx.index_dir, "doc_lengths.tsv"), encoding="utf-8", newline="\n") as f:
                for line in f:
                    d_s, l_s = line.rstrip("\n").split("\t")
                    k[int(d_s)] = self.k1 * (1.0 - self.b + self.b * int(l_s) / avgdl)
            self._bm_k = k
        return self._bm_k

    # ------------------------------------------------------------------
    # idf
    # ------------------------------------------------------------------

    # idf of a processed term under one SMART idf letter
    def idf(self, term: str, letter: str) -> float:
        df = self.idx.df(term)
        if df == 0:
            return 0.0
        if letter == "n":
            return 1.0
        if letter == "t":
            return self.idx.idf(term)
        # probabilistic idf, clipped at 0 for terms in more than half of the documents
        n = self.idx.N
        return max(0.0, math.log((n - df) / df, self.idx.log_base)) if df < n else 0.0

    # BM25 idf: ln(1 + (N - df + 0.5) / (df + 0.5)), always positive
    def bm25_idf(self, term: str) -> float:
        df = self.idx.df(term)
        return math.log(1.0 + (self.idx.N - df + 0.5) / (df + 0.5)) if df else 0.0

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------

    # function (doc_id, tf) -> un-normalized document weight of one term
    def doc_weight_fn(self, term: str) -> Callable[[int, int], float]:
        if self.is_bm25:
            idf, k1p, bm_k = self.bm25_idf(term), self.k1 + 1.0, self.bm25_k()
            return lambda d, tf: idf * (tf * k1p / (tf + bm_k[d]))
        letter = self.doc[0]
        idf = self.idf(term, self.doc[1])
        tfw = self.idx.tf_weight
        if letter == "l":
            # same expression as the original ranker: (1 + log tf) * idf
            return lambda d, tf: tfw(tf) * idf
        if letter == "n":
            return lambda d, tf: tf * idf
        if letter == "b":
            return lambda d, tf: idf
        if letter == "a":
            max_tf = self._doc_max_tf()
            return lambda d, tf: (0.5 + 0.5 * tf / max_tf[d]) * idf
        # "L": log tf relative to the document's average tf
        avg_tf = self._doc_avg_tf()
        return lambda d, tf: tfw(tf) / tfw(avg_tf[d]) * idf

    # (term -> query weight, query vector length) for the query term frequencies
    def query_weights(self, qtf: Dict[str, int]) -> Tuple[Dict[str, float], float]:
        if self.is_bm25:
            # a repeated query word counts once per occurrence
            return {t: float(c) for t, c in qtf.items()}, 1.0
        letter = self.query[0]
        tfw = self.idx.tf_weight
        max_q = max(qtf.values(), default=1)
        avg_q = sum(qtf.values()) / len(qtf) if qtf else 1.0
        w: Dict[str, float] = {}
        for t, c in qtf.items():
            idf = self.idf(t, self.query[1])
            if letter == "l":
                w[t] = tfw(c) * idf
            elif letter == "n":
                w[t] = c * idf
            elif letter == "b":
                w[t] = idf
            elif letter == "a":
                w[t] = (0.5 + 0.5 * c / max_q) * idf
            else:
                w[t] = tfw(c) / tfw(avg_q) * idf
        norm = math.sqrt(sum(v ** 2 for v in w.values())) if self.query[2] == "c" else 1.0
        return w, norm

    # which precomputed max-score bound of tf_index.bin is valid for this scheme: "ltc", "bm25" or None
    def bound_kind(self, tf_index_meta: dict) -> Optional[str]:
        if self.is_bm25:
            bm = tf_index_meta.get("bm25", {})
            return "bm25" if (bm.get("k1"), bm.get("b")) == (self.k1, self.b) else None
        return "ltc" if self.doc == "ltc" else None
