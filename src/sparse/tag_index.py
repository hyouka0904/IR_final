"""Positional inverted index over image tags, scored with HW2's Okapi BM25.

An image is a document; its tags (in descending confidence) are written as a token sequence where
each tag repeats tf times, so positions are token slots. Files use the HW1 Slide-5 format.
"""
import math
import re
from pathlib import Path

from src.sparse.hw1_format import PostingEntry, format_posting_standard, parse_posting_line


_TERM = re.compile(r"[a-z0-9_]+")


class TagIndex:
    def __init__(self, k1: float, b: float):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, list[int]]] = {}  # term -> doc -> positions
        self._doc_len: dict[int, int] = {}
        self._doc_terms: dict[int, list[str]] = {}
        self._total_len = 0

    # ---- building -------------------------------------------------------
    def add(self, image_id: int, tags: list[tuple[str, int]]) -> None:
        """Index one image. Raises ValueError (and changes nothing) for a repeated id, no tags,
        a tf below 1, a repeated term, or a term outside [a-z0-9_]."""
        image_id = int(image_id)
        if image_id in self._doc_len:
            raise ValueError(f"image id {image_id} is already in the tag index")
        if not tags:
            raise ValueError(f"image {image_id} has no tags; it could not be stored")
        seen = set()
        for term, tf in tags:
            if not _TERM.fullmatch(term):
                raise ValueError(f"tag {term!r} must match [a-z0-9_]+ (HW1 record format)")
            if int(tf) < 1:
                raise ValueError(f"tag {term!r} has tf {tf}; tf must be >= 1")
            if term in seen:
                raise ValueError(f"tag {term!r} appears twice for image {image_id}")
            seen.add(term)
        pos = 0
        terms = []
        for term, tf in tags:
            tf = int(tf)
            self.postings.setdefault(term, {})[image_id] = list(range(pos, pos + tf))
            pos += tf
            terms.append(term)
        self._doc_len[image_id] = pos
        self._doc_terms[image_id] = terms
        self._total_len += pos

    @classmethod
    def from_docs(cls, docs: dict[int, list[tuple[str, int]]], k1: float, b: float) -> "TagIndex":
        idx = cls(k1, b)
        for image_id in sorted(docs):
            idx.add(image_id, docs[image_id])
        return idx

    # ---- statistics -----------------------------------------------------
    @property
    def N(self) -> int:
        return len(self._doc_len)

    @property
    def avgdl(self) -> float:
        return self._total_len / self.N if self.N else 0.0

    @property
    def ids(self) -> list[int]:
        return sorted(self._doc_len)

    def doc_len(self, image_id: int) -> int:
        return self._doc_len[int(image_id)]

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def doc_terms(self, image_id: int) -> list[str]:
        return list(self._doc_terms.get(int(image_id), []))

    # ---- scoring (HW2 weighting.py, scheme "bm25") ----------------------
    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5)) if df else 0.0

    def score(self, query_tf: dict[str, int]) -> dict[int, float]:
        """BM25 score of every image sharing at least one query term; query weight = raw qtf."""
        scores: dict[int, float] = {}
        avgdl, k1p = self.avgdl, self.k1 + 1.0
        for term, qtf in query_tf.items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for d, positions in plist.items():
                tf = len(positions)
                bm_k = self.k1 * (1.0 - self.b + self.b * self._doc_len[d] / avgdl)
                scores[d] = scores.get(d, 0.0) + float(qtf) * (idf * (tf * k1p / (tf + bm_k)))
        return scores

    def search(self, query_tf: dict[str, int], k: int) -> list[tuple[int, float]]:
        """Top-k (image id, score), score descending, id ascending on ties."""
        hits = sorted(self.score(query_tf).items(), key=lambda x: (-x[1], x[0]))
        return hits[:k]

    # ---- files (HW1 format) ---------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.postings)
        with open(directory / "dictionary.txt", "w", encoding="utf-8", newline="\n") as f:
            for t in terms:
                f.write(f"{t}, {self.df(t)}\n")
        with open(directory / "postings.txt", "w", encoding="utf-8", newline="\n") as f:
            for t in terms:
                entries = [PostingEntry(d, len(p), p) for d, p in sorted(self.postings[t].items())]
                f.write(format_posting_standard(t, entries) + "\n")

    @classmethod
    def load(cls, directory: Path, k1: float, b: float) -> "TagIndex":
        """Rebuild from postings.txt; each image's term order comes from its first positions."""
        docs: dict[int, list[tuple[int, str, int]]] = {}
        with open(Path(directory) / "postings.txt", encoding="utf-8", newline="\n") as f:
            for line in f:
                if not line.strip():
                    continue
                term, _, entries = parse_posting_line(line, attach_skips=False)
                for e in entries:
                    docs.setdefault(e.doc_id, []).append((min(e.positions), term, e.tf))
        return cls.from_docs({d: [(t, tf) for _, t, tf in sorted(v)] for d, v in docs.items()}, k1, b)
