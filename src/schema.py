"""Data passed between modules (README: agreed by both members)."""
from dataclasses import dataclass, field

from PIL import Image

TASKS = ("semantic", "similar")
INDEX_KINDS = ("flat", "cluster", "ivfpq", "hnsw")
RERANKERS = ("aqe", "rocchio", "mmr")  # applied in this order


@dataclass
class SearchRequest:
    image: Image.Image
    task: str
    k: int = 10
    representation: str | None = None
    crop: tuple[int, int, int, int] | None = None  # (left, top, right, bottom) pixels
    index: str = "flat"                             # README: Indexing
    rerank: tuple[str, ...] = ()                    # any of RERANKERS; run in RERANKERS order
    # Rocchio: image ids the user marked (relevant, non-relevant); ignored without "rocchio"
    feedback: tuple[list[int], list[int]] | None = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.index not in INDEX_KINDS:
            raise ValueError(f"index must be one of {INDEX_KINDS}, got {self.index!r}")
        unknown = [r for r in self.rerank if r not in RERANKERS]
        if unknown:
            raise ValueError(f"rerank must be drawn from {RERANKERS}, got {unknown}")
        self.rerank = tuple(r for r in RERANKERS if r in self.rerank)


@dataclass
class SearchResult:
    image_id: int
    score: float
    rank: int
    shared_tags: list[str] = field(default_factory=list)  # tags/fusion: query tags it also has


@dataclass
class SearchResponse:
    results: list[SearchResult]
    representation: str
    timings_ms: dict[str, float] = field(default_factory=dict)
    query_tags: list[str] = field(default_factory=list)  # tags/fusion: the query image's tags
    index: str = "flat"
    rerank: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)       # e.g. Rocchio asked for without feedback
