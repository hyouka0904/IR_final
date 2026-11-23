import pytest

from src.config import load_config
from src.pipeline import tag_variant
from src.rerank import rrf
from src.schema import SearchRequest
from tests.fakes import COLORS, solid, tag_engine

RED = COLORS[1]


@pytest.fixture
def cfg():
    return load_config()


def _search(eng, rep, k=10, task="semantic", color=RED):
    return eng.search(SearchRequest(solid(color), task=task, k=k, representation=rep))


def test_tag_variant():
    assert tag_variant("fusion") == "tags" and tag_variant("fusion_nococo") == "tags_nococo"
    assert tag_variant("tags_nococo") == "tags_nococo" and tag_variant("clip") is None


def test_available_lists_tag_and_fusion(cfg):
    eng = tag_engine(cfg)
    assert set(eng.available()) == {"clip", "dinov2", "tags", "fusion", "tags_nococo",
                                    "fusion_nococo"}
    assert "fusion" not in tag_engine(cfg, dense=("dinov2",)).available()


def test_tags_rep_returns_bm25_ranking_with_shared_tags(cfg):
    eng = tag_engine(cfg)
    resp = _search(eng, "tags")
    idx = eng.tag_indexes["tags"]
    expected = idx.search({"red": 5, "green": 1}, 10)
    assert resp.query_tags == ["red", "green"]
    assert [(r.image_id, pytest.approx(r.score)) for r in resp.results] == expected
    # yellow (red 5, green 5) outranks red (red 5, green 1); blue (blue 5, red 1) is last
    assert [r.image_id for r in resp.results][:2] == [4, 1]
    assert resp.results[0].shared_tags == ["red", "green"]
    for r in resp.results:
        assert r.shared_tags == [t for t in resp.query_tags if t in idx.doc_terms(r.image_id)]


def test_fusion_rrf_of_clip_and_tags(cfg):
    eng = tag_engine(cfg)
    clip_ids = [r.image_id for r in _search(eng, "clip", k=100).results]
    tag_ids = [r.image_id for r in _search(eng, "tags", k=100).results]
    resp = _search(eng, "fusion", k=3)
    expected = rrf([clip_ids, tag_ids], cfg["fusion"]["rrf_k"])[:3]
    assert [(r.image_id, pytest.approx(r.score)) for r in resp.results] == expected
    assert [r.rank for r in resp.results] == [1, 2, 3]
    assert resp.representation == "fusion"


def test_fusion_falls_back_to_clip_when_no_tag_overlap(cfg):
    eng = tag_engine(cfg, tag_docs={i: [("zzz", 1)] for i in COLORS})
    assert _search(eng, "tags").results == []
    clip_ids = [r.image_id for r in _search(eng, "clip").results]
    fused = _search(eng, "fusion").results
    assert [r.image_id for r in fused] == clip_ids
    assert all(r.shared_tags == [] for r in fused)


def test_fusion_small_gallery_no_duplicates(cfg):
    ids = [r.image_id for r in _search(tag_engine(cfg), "fusion", k=50).results]
    assert len(ids) == len(set(ids)) <= 4


def test_disallowed_task_rep_raises(cfg):
    with pytest.raises(ValueError, match="allowed"):
        _search(tag_engine(cfg), "tags", task="similar")


def test_unavailable_rep_raises(cfg):
    with pytest.raises(ValueError, match="phash"):
        _search(tag_engine(cfg), "phash")


def test_add_image_searchable_in_every_rep(cfg):
    eng = tag_engine(cfg)
    cyan = (0, 200, 200)
    first = cfg["user"]["first_id"]
    assert eng.add_image(solid(cyan)) == first
    for rep in ("clip", "dinov2", "tags", "fusion", "tags_nococo"):
        assert _search(eng, rep, color=cyan).results[0].image_id == first, rep
    assert eng.tag_indexes["tags"].N == 5
    assert eng.add_image(solid((90, 90, 90))) == first + 1
    assert first + 1 in eng.gallery_ids("fusion")


def test_add_image_explicit_id_and_duplicate_raises(cfg):
    eng = tag_engine(cfg)
    assert eng.add_image(solid((9, 9, 9)), image_id=1_000_000_042) == 1_000_000_042
    assert eng.add_image(solid((8, 8, 8))) == 1_000_000_043
    with pytest.raises(ValueError):
        eng.add_image(solid((7, 7, 7)), image_id=3)
    with pytest.raises(ValueError):
        eng.add_image(solid((7, 7, 7)), image_id=1_000_000_042)


def test_query_tags_and_single_clip_encode(cfg):
    eng = tag_engine(cfg)
    resp = _search(eng, "fusion")
    assert resp.query_tags == ["red", "green"]
    assert eng.encoders["clip"].calls == 1
    assert set(resp.timings_ms) == {"embed", "search", "rerank"}


def test_gallery_ids_per_rep(cfg):
    eng = tag_engine(cfg)
    assert eng.gallery_ids("clip") == eng.gallery_ids("tags") == eng.gallery_ids("fusion") \
        == set(COLORS)


def test_shared_tags_are_timed_as_search_not_rerank(cfg, monkeypatch):
    import time

    import src.pipeline as pipeline
    real = pipeline.shared_tags

    def slow(q, d):
        time.sleep(0.02)
        return real(q, d)

    monkeypatch.setattr(pipeline, "shared_tags", slow)
    resp = _search(tag_engine(cfg), "tags", k=4)
    assert resp.timings_ms["search"] >= 4 * 20
    assert resp.timings_ms["rerank"] < 20
