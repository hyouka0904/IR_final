"""Shared test doubles: a 4-image color gallery with clip/dinov2 indexes and a 3-term tagger."""
import numpy as np
from PIL import Image

from src.index import FlatIndex
from src.pipeline import SearchEngine
from src.sparse.tag_index import TagIndex
from src.tagging import Tagger

COLORS = {1: (250, 0, 0), 2: (0, 250, 0), 3: (0, 0, 250), 4: (200, 200, 0)}
TERMS = ["red", "green", "blue"]


class MeanColorEncoder:
    """A unit vector of the mean RGB color; counts encode calls."""
    metric = "ip"

    def __init__(self, name="clip"):
        self.name = name
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        v = np.array([np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0
                      for im in images], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def color_tagger(top_n=2):
    return Tagger(TERMS, np.eye(3, dtype=np.float32), top_n=top_n, tf_max=5, logit_scale=100)


def solid(color, size=32):
    return Image.new("RGB", (size, size), color)


def tag_engine(cfg, tag_docs=None, dense=("clip", "dinov2")):
    """Engine over COLORS. tag_docs overrides the gallery's tags (default: tagged by the tagger)."""
    imgs = [solid(c) for c in COLORS.values()]
    ids = np.array(list(COLORS))
    encoders, indexes = {}, {}
    for n in dense:
        enc = MeanColorEncoder(n)
        encoders[n] = enc
        indexes[n] = FlatIndex.build(enc.encode(imgs), ids, enc.metric)
    if "clip" not in encoders:
        encoders["clip"] = MeanColorEncoder("clip")
    tagger = color_tagger()
    if tag_docs is None:
        tags = tagger.tag(encoders["clip"].encode(imgs))
        tag_docs = {int(i): [(t, tf) for t, _, tf in tg] for i, tg in zip(ids, tags)}
    for enc in encoders.values():
        enc.calls = 0
    k1, b = cfg["bm25"]["k1"], cfg["bm25"]["b"]
    taggers = {v: tagger for v in ("tags", "tags_nococo")}
    tag_indexes = {v: TagIndex.from_docs(tag_docs, k1, b) for v in ("tags", "tags_nococo")}
    return SearchEngine(cfg, encoders, indexes, taggers=taggers, tag_indexes=tag_indexes)
