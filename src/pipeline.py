"""Online search, shared by evaluation, the demo and the pairs script."""
import time

import numpy as np
from PIL import Image

from src.config import resolve_path
from src.encoders import Encoder, get_encoder
from src.index import FlatIndex
from src.preprocess import preprocess
from src.rerank import rrf, shared_tags
from src.schema import SearchRequest, SearchResponse, SearchResult
from src.sparse.tag_index import TagIndex
from src.tagging import Tagger

DENSE = ("clip", "dinov2", "phash")
REPRESENTATIONS = ("clip", "dinov2", "phash", "tags", "fusion", "tags_nococo", "fusion_nococo")
_FUSION = {"fusion": "tags", "fusion_nococo": "tags_nococo"}


def tag_variant(rep: str) -> str | None:
    """The tag cache a representation reads: fusion -> tags, fusion_nococo -> tags_nococo."""
    if rep in _FUSION:
        return _FUSION[rep]
    return rep if rep.startswith("tags") else None


class SearchEngine:
    def __init__(self, cfg: dict, encoders: dict[str, Encoder], indexes: dict[str, FlatIndex],
                 load_encoders: bool = False, taggers: dict[str, Tagger] | None = None,
                 tag_indexes: dict[str, TagIndex] | None = None):
        self.cfg = cfg
        self.encoders = dict(encoders)
        self.indexes = dict(indexes)
        self.taggers = dict(taggers or {})
        self.tag_indexes = dict(tag_indexes or {})
        self._load_encoders = load_encoders

    @classmethod
    def from_cache(cls, cfg: dict, representations: list[str] | None = None) -> "SearchEngine":
        """Load every cached dense index and tag variant (or only those `representations` need)."""
        cache = resolve_path(cfg, "cache")
        wanted = set(representations or REPRESENTATIONS)
        dense = {r for r in DENSE if r in wanted}
        variants = {tag_variant(r) for r in wanted} - {None}
        if wanted & set(_FUSION):
            dense.add("clip")  # fusion reads the clip kNN line
        indexes = {r: FlatIndex.load(cache / r / "flat") for r in sorted(dense)
                   if (cache / r / "flat" / "meta.json").exists()}
        taggers, tag_indexes = {}, {}
        k1, b = cfg["bm25"]["k1"], cfg["bm25"]["b"]
        for v in cfg["tags"]["variants"]:
            if v in variants and (cache / v / "postings.txt").exists():
                taggers[v] = Tagger.load(cache / v, cfg)
                tag_indexes[v] = TagIndex.load(cache / v, k1, b)
        return cls(cfg, {}, indexes, load_encoders=True, taggers=taggers, tag_indexes=tag_indexes)

    def encoder(self, rep: str) -> Encoder:
        if rep not in self.encoders and self._load_encoders:
            self.encoders[rep] = get_encoder(rep, self.cfg)
        return self.encoders[rep]

    # ---- what is loaded -------------------------------------------------
    def _has_clip_encoder(self) -> bool:
        return "clip" in self.encoders or self._load_encoders

    def available(self) -> list[str]:
        """Representations whose parts are all loaded, in REPRESENTATIONS order."""
        out = []
        for rep in REPRESENTATIONS:
            v = tag_variant(rep)
            if v is None:
                ok = rep in self.indexes
            else:
                ok = v in self.taggers and v in self.tag_indexes and self._has_clip_encoder()
                if rep in _FUSION:
                    ok = ok and "clip" in self.indexes
            if ok:
                out.append(rep)
        return out

    def gallery_ids(self, rep: str) -> set[int]:
        """Image ids that `rep` searches over."""
        v = tag_variant(rep)
        if v is None:
            return set(int(i) for i in self.indexes[rep].ids)
        tag_ids = set(self.tag_indexes[v].ids)
        if rep in _FUSION:
            clip_ids = set(int(i) for i in self.indexes["clip"].ids)
            if clip_ids != tag_ids:
                raise ValueError(f"{rep}: the clip index ({len(clip_ids)} images) and the {v} index "
                                 f"({len(tag_ids)} images) differ; rerun 03_tag.py")
        return tag_ids

    # ---- search ---------------------------------------------------------
    def _check(self, task: str, rep: str) -> None:
        allowed = self.cfg["tasks"][task]["allowed"]
        if rep not in allowed:
            raise ValueError(f"representation {rep!r} is not allowed for the {task} task; "
                             f"allowed: {', '.join(allowed)}")
        if rep not in self.available():
            raise ValueError(f"representation {rep!r} is not loaded; "
                             f"available: {', '.join(self.available())}")

    def search(self, req: SearchRequest) -> SearchResponse:
        rep = req.representation or self.cfg["tasks"][req.task]["representation"]
        self._check(req.task, rep)
        v = tag_variant(rep)
        encoder = self.encoder("clip" if v else rep)  # lazy model load stays out of the timings

        t0 = time.perf_counter()
        img = preprocess(req.image, self.cfg["preprocess"]["max_side"], req.crop)
        query = encoder.encode([img])
        query_terms: list[str] = []
        if v:
            qtags = self.taggers[v].tag(query)[0]
            qtf = {t: tf for t, _, tf in qtags}
            query_terms = [t for t, _, _ in qtags]
        t1 = time.perf_counter()

        if v is None:
            scores, ids = self.indexes[rep].search(query, req.k)
            hits = list(zip(ids[0].tolist(), scores[0].tolist()))
        elif rep not in _FUSION:
            hits = self.tag_indexes[v].search(qtf, req.k)
        else:
            depth = self.cfg["fusion"]["depth"]
            _, clip_ids = self.indexes["clip"].search(query, depth)
            tag_ids = [d for d, _ in self.tag_indexes[v].search(qtf, depth)]
            hits = rrf([clip_ids[0].tolist(), tag_ids], self.cfg["fusion"]["rrf_k"])[:req.k]
        results = []
        for r, (i, s) in enumerate(hits):
            shared = shared_tags(query_terms, self.tag_indexes[v].doc_terms(i)) if v else []
            results.append(SearchResult(image_id=int(i), score=float(s), rank=r + 1,
                                        shared_tags=shared))
        t2 = time.perf_counter()  # building results (incl. shared tags) counts as search

        # re-ranking (Rocchio / alpha-QE / MMR) arrives in week 3; timed as a stage already
        t3 = time.perf_counter()
        return SearchResponse(results=results, representation=rep,
                              timings_ms={"embed": (t1 - t0) * 1e3, "search": (t2 - t1) * 1e3,
                                          "rerank": (t3 - t2) * 1e3},
                              query_tags=query_terms)

    # ---- growing the gallery -------------------------------------------
    def all_ids(self) -> set[int]:
        """Every image id in any loaded index."""
        ids = set()
        for index in self.indexes.values():
            ids |= set(int(i) for i in index.ids)
        for index in self.tag_indexes.values():
            ids |= set(index.ids)
        return ids

    def add_image(self, img: Image.Image, image_id: int | None = None) -> int:
        """Add one image to every loaded dense index and tag index; returns its id.
        New ids start at user.first_id and count up from the largest user id."""
        existing = self.all_ids()
        first = int(self.cfg["user"]["first_id"])
        if image_id is None:
            user_ids = [i for i in existing if i >= first]
            image_id = max(user_ids) + 1 if user_ids else first
        image_id = int(image_id)
        if image_id in existing:
            raise ValueError(f"image id {image_id} is already in the gallery")
        img = preprocess(img, self.cfg["preprocess"]["max_side"])
        vecs: dict[str, np.ndarray] = {}
        for rep in self.indexes:
            vecs[rep] = self.encoder(rep).encode([img])
        if self.taggers and "clip" not in vecs:
            vecs["clip"] = self.encoder("clip").encode([img])
        for rep, index in self.indexes.items():
            index.add(vecs[rep], np.array([image_id]))
        for v, tagger in self.taggers.items():
            if v in self.tag_indexes:
                tags = tagger.tag(vecs["clip"])[0]
                self.tag_indexes[v].add(image_id, [(t, tf) for t, _, tf in tags])
        return image_id
