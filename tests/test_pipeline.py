import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.embed import embed_paths
from src.index import FlatIndex
from src.pipeline import SearchEngine
from src.schema import SearchRequest

COLORS = {1: (250, 0, 0), 2: (0, 250, 0), 3: (0, 0, 250), 4: (200, 200, 0)}


class MeanColorEncoder:
    """Test double: a unit vector of the mean RGB color, robust to most edits."""
    metric = "ip"

    def __init__(self, name="fake"):
        self.name = name

    def encode(self, images):
        v = np.array([np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) + 1.0
                      for im in images], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _engine(cfg, names=("clip", "dinov2")):
    imgs = [Image.new("RGB", (32, 32), c) for c in COLORS.values()]
    ids = np.array(list(COLORS))
    encoders, indexes = {}, {}
    for n in names:
        enc = MeanColorEncoder(n)
        encoders[n] = enc
        indexes[n] = FlatIndex.build(enc.encode(imgs), ids, enc.metric)
    return SearchEngine(cfg, encoders, indexes)


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["tasks"]["semantic"]["representation"] = "clip"  # this gallery has no tag index
    return cfg


def test_embed_paths_shape(tmp_path):
    paths = []
    for i, c in COLORS.items():
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (900, 40), c).save(p)
        paths.append(p)
    v = embed_paths(MeanColorEncoder(), paths, batch_size=3, max_side=512)
    assert v.shape == (4, 3) and v.dtype == np.float32


def test_search_returns_k_ranked_results(cfg):
    resp = _engine(cfg).search(SearchRequest(Image.new("RGB", (20, 20), (0, 240, 0)),
                                             task="similar", k=3))
    assert [r.rank for r in resp.results] == [1, 2, 3]
    assert resp.results[0].image_id == 2
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)


def test_search_k_larger_than_gallery(cfg):
    resp = _engine(cfg).search(SearchRequest(Image.new("RGB", (8, 8)), task="similar", k=50))
    assert len(resp.results) == 4


def test_search_uses_task_default_representation(cfg):
    eng = _engine(cfg)
    img = Image.new("RGB", (8, 8))
    assert eng.search(SearchRequest(img, task="semantic")).representation == "clip"
    assert eng.search(SearchRequest(img, task="similar")).representation == "dinov2"
    assert eng.search(SearchRequest(img, task="similar", representation="clip")
                      ).representation == "clip"


def test_search_applies_crop(cfg):
    img = Image.new("RGB", (40, 20), (250, 0, 0))
    img.paste((0, 0, 250), (20, 0, 40, 20))  # right half blue
    resp = _engine(cfg).search(SearchRequest(img, task="similar", crop=(20, 0, 40, 20)))
    assert resp.results[0].image_id == 3


def test_search_records_three_timings(cfg):
    resp = _engine(cfg).search(SearchRequest(Image.new("RGB", (8, 8)), task="semantic"))
    assert set(resp.timings_ms) == {"embed", "search", "rerank"}
    assert all(v >= 0 for v in resp.timings_ms.values())


def test_unknown_representation_raises(cfg):
    with pytest.raises(ValueError):
        _engine(cfg).search(SearchRequest(Image.new("RGB", (8, 8)), task="semantic",
                                          representation="siglip"))
