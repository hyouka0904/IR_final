"""Data passed between modules (README: agreed by both members)."""
from dataclasses import dataclass, field

from PIL import Image

TASKS = ("semantic", "similar")


@dataclass
class SearchRequest:
    image: Image.Image
    task: str
    k: int = 10
    representation: str | None = None
    crop: tuple[int, int, int, int] | None = None  # (left, top, right, bottom) pixels

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")


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
