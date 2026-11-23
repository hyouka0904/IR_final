"""Online search, shared by evaluation, the demo and the pairs script."""
import time

import numpy as np
from PIL import Image

from src.config import resolve_path
from src.encoders import Encoder, get_encoder
from src.index import FlatIndex, load_index
from src.preprocess import preprocess
from src.rerank import alpha_qe, mmr, rocchio, rrf, shared_tags
from src.schema import INDEX_KINDS, SearchRequest, SearchResponse, SearchResult
from src.sparse.tag_index import TagIndex
from src.tagging import Tagger

DENSE = ("clip", "siglip", "dinov2", "phash")
VECTOR = ("clip", "siglip", "dinov2")  # float vectors: ANN indexes and re-ranking apply
REPRESENTATIONS = ("clip", "siglip", "dinov2", "phash", "tags", "fusion", "tags_nococo",
                   "fusion_nococo")
_FUSION = {"fusion": "tags", "fusion_nococo": "tags_nococo"}


def tag_variant(rep: str) -> str | None:
    """The tag cache a representation reads: fusion -> tags, fusion_nococo -> tags_nococo."""
    if rep in _FUSION:
        return _FUSION[rep]
    return rep if rep.startswith("tags") else None


def vector_line(rep: str) -> str | None:
    """The float-vector representation behind `rep` (fusion's dense line is clip), or None for
    pHash and tags-only, which have no vectors to index approximately or to re-rank with."""
    if rep in _FUSION:
        return "clip"
    return rep if rep in VECTOR else None


class SearchEngine:
    def __init__(self, cfg: dict, encoders: dict[str, Encoder], indexes: dict[str, FlatIndex],
                 load_encoders: bool = False, taggers: dict[str, Tagger] | None = None,
                 tag_indexes: dict[str, TagIndex] | None = None,
                 ann_indexes: dict[tuple[str, str], object] | None = None):
        self.cfg = cfg
        self.encoders = dict(encoders)
        self.indexes = dict(indexes)  # flat index per dense representation
        self.taggers = dict(taggers or {})
        self.tag_indexes = dict(tag_indexes or {})
        self.ann_indexes = dict(ann_indexes or {})  # (rep, kind) -> cluster / ivfpq / hnsw index
        self._index_dirs: dict[tuple[str, str], object] = {}  # built on disk, loaded on first use
        self._added: list[tuple[int, dict[str, np.ndarray]]] = []  # replayed into lazy indexes
        self._vectors: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
        self._load_encoders = load_encoders
        self._cache = None  # cache folder when loaded from_cache (index sizes on disk)

    @classmethod
    def from_cache(cls, cfg: dict, representations: list[str] | None = None) -> "SearchEngine":
        """Load every cached dense index and tag variant (or only those `representations` need).
        Non-flat indexes (cluster / ivfpq / hnsw) are found here and loaded on first use."""
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
        engine = cls(cfg, {}, indexes, load_encoders=True, taggers=taggers, tag_indexes=tag_indexes)
        engine._cache = cache
        for rep in indexes:
            for kind in INDEX_KINDS[1:]:
                if (cache / rep / kind / "meta.json").exists():
                    engine._index_dirs[(rep, kind)] = cache / rep / kind
        return engine

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

    def index_kinds(self, rep: str) -> list[str]:
        """Index kinds `rep` can search with: flat, plus the built ANN indexes of its vector line."""
        line = vector_line(rep)
        if line is None:
            return ["flat"]
        return ["flat"] + [k for k in INDEX_KINDS[1:]
                           if (line, k) in self.ann_indexes or (line, k) in self._index_dirs]

    def can_rerank(self, rep: str) -> bool:
        return vector_line(rep) is not None

    def index_for(self, rep: str, kind: str):
        if kind == "flat":
            return self.indexes[rep]
        key = (rep, kind)
        if key not in self.ann_indexes:
            if key not in self._index_dirs:
                raise ValueError(f"no {kind} index for {rep}; run "
                                 f"scripts/04_build_index.py --index {kind} --models {rep}")
            index = load_index(self._index_dirs[key], self.cfg)
            for image_id, vecs in self._added:  # photos added before this index was loaded
                if rep in vecs:
                    index.add(vecs[rep], np.array([image_id]))
            self.ann_indexes[key] = index
        return self.ann_indexes[key]

    def index_dir(self, rep: str, kind: str):
        """The folder a saved index was loaded from, or None (built in memory)."""
        if (rep, kind) in self._index_dirs:
            return self._index_dirs[(rep, kind)]
        if self._cache is not None and (self._cache / rep / kind / "meta.json").exists():
            return self._cache / rep / kind
        return None

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

    def vectors_of(self, rep: str, image_ids) -> np.ndarray:
        """Gallery vectors of `rep` for the given image ids (from its exact flat index)."""
        if rep not in self._vectors:
            flat = self.indexes[rep]
            self._vectors[rep] = (flat.vectors(), {int(i): p for p, i in enumerate(flat.ids)})
        vecs, pos = self._vectors[rep]
        rows = [pos[int(i)] for i in image_ids]
        return vecs[rows] if rows else np.zeros((0, vecs.shape[1]), dtype=np.float32)

    # ---- search ---------------------------------------------------------
    def _check(self, req: SearchRequest, rep: str) -> None:
        allowed = self.cfg["tasks"][req.task]["allowed"]
        if rep not in allowed:
            raise ValueError(f"representation {rep!r} is not allowed for the {req.task} task; "
                             f"allowed: {', '.join(allowed)}")
        if rep not in self.available():
            raise ValueError(f"representation {rep!r} is not loaded; "
                             f"available: {', '.join(self.available())}")
        if req.index != "flat" and vector_line(rep) is None:
            raise ValueError(f"the {req.index} index needs float vectors; {rep} only supports flat")
        if req.index not in self.index_kinds(rep):
            raise ValueError(f"no {req.index} index for {vector_line(rep)}; built: "
                             f"{', '.join(self.index_kinds(rep))} (scripts/04_build_index.py)")
        if req.rerank and not self.can_rerank(rep):
            raise ValueError(f"re-ranking ({', '.join(req.rerank)}) needs a vector representation "
                             f"(clip, siglip, dinov2 or fusion); {rep} has none")

    def _retrieve(self, rep: str, kind: str, qvec: np.ndarray, qtf: dict[str, int],
                  depth: int) -> list[tuple[int, float]]:
        v = tag_variant(rep)
        if v is None:
            scores, ids = self.index_for(rep, kind).search(qvec[None], depth)
            return list(zip(ids[0].tolist(), scores[0].tolist()))
        if rep not in _FUSION:
            return self.tag_indexes[v].search(qtf, depth)
        fdepth = max(self.cfg["fusion"]["depth"], depth)
        _, clip_ids = self.index_for("clip", kind).search(qvec[None], fdepth)
        tag_ids = [d for d, _ in self.tag_indexes[v].search(qtf, fdepth)]
        return rrf([clip_ids[0].tolist(), tag_ids], self.cfg["fusion"]["rrf_k"])[:depth]

    def _rerank(self, req: SearchRequest, rep: str, qvec: np.ndarray, qtf: dict[str, int],
                hits: list[tuple[int, float]], depth: int, notes: list[str]):
        line = vector_line(rep)
        r = self.cfg["rerank"]
        if "aqe" in req.rerank:
            c = r["aqe"]
            top = [d for d, _ in hits[:c["n"]]]
            qvec = alpha_qe(qvec, self.vectors_of(line, top), c["n"], c["alpha"])
            hits = self._retrieve(rep, req.index, qvec, qtf, depth)
        if "rocchio" in req.rerank:
            known = self.gallery_ids(rep)
            rel, nonrel = req.feedback or ([], [])
            rel = [int(i) for i in rel if int(i) in known]
            nonrel = [int(i) for i in nonrel if int(i) in known]
            if rel or nonrel:
                c = r["rocchio"]
                qvec = rocchio(qvec, self.vectors_of(line, rel), self.vectors_of(line, nonrel),
                               c["alpha"], c["beta"], c["gamma"])
                hits = self._retrieve(rep, req.index, qvec, qtf, depth)
            else:
                notes.append("rocchio: no results marked yet, ranking unchanged")
        if "mmr" in req.rerank:
            c = r["mmr"]
            head = hits[:c["depth"]]
            hits = mmr(hits, self.vectors_of(line, [d for d, _ in head]), c["lambda"], c["depth"])
        return hits

    def search(self, req: SearchRequest) -> SearchResponse:
        rep = req.representation or self.cfg["tasks"][req.task]["representation"]
        self._check(req, rep)
        v = tag_variant(rep)
        encoder = self.encoder("clip" if v else rep)  # lazy model load stays out of the timings
        if req.index != "flat":
            self.index_for(vector_line(rep), req.index)  # lazy index load too
        if req.rerank:
            self.vectors_of(vector_line(rep), [])

        t0 = time.perf_counter()
        img = preprocess(req.image, self.cfg["preprocess"]["max_side"], req.crop)
        query = encoder.encode([img])
        query_terms: list[str] = []
        qtf: dict[str, int] = {}
        if v:
            qtags = self.taggers[v].tag(query)[0]
            qtf = {t: tf for t, _, tf in qtags}
            query_terms = [t for t, _, _ in qtags]
        t1 = time.perf_counter()

        # MMR needs a candidate pool deeper than k to pick diverse results from
        depth = max(req.k, self.cfg["rerank"]["mmr"]["depth"]) if "mmr" in req.rerank else req.k
        hits = self._retrieve(rep, req.index, query[0], qtf, depth)
        t2 = time.perf_counter()

        notes: list[str] = []
        if req.rerank:
            hits = self._rerank(req, rep, query[0], qtf, hits, depth, notes)
        t3 = time.perf_counter()

        results = []
        for r, (i, s) in enumerate(hits[:req.k]):
            shared = shared_tags(query_terms, self.tag_indexes[v].doc_terms(i)) if v else []
            results.append(SearchResult(image_id=int(i), score=float(s), rank=r + 1,
                                        shared_tags=shared))
        t4 = time.perf_counter()  # building results (incl. shared tags) counts as search
        return SearchResponse(results=results, representation=rep,
                              timings_ms={"embed": (t1 - t0) * 1e3,
                                          "search": ((t2 - t1) + (t4 - t3)) * 1e3,
                                          "rerank": (t3 - t2) * 1e3},
                              query_tags=query_terms, index=req.index, rerank=req.rerank,
                              notes=notes)

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
        """Add one image to every loaded index (flat, ANN, tags); returns its id.
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
        for (rep, _), index in self.ann_indexes.items():
            if rep in vecs:
                index.add(vecs[rep], np.array([image_id]))
        self._added.append((image_id, vecs))
        self._vectors.clear()  # rebuilt from the flat indexes on next use
        for v, tagger in self.taggers.items():
            if v in self.tag_indexes:
                tags = tagger.tag(vecs["clip"])[0]
                self.tag_indexes[v].add(image_id, [(t, tf) for t, _, tf in tags])
        return image_id
