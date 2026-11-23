"""Index kinds and re-ranking through SearchEngine.search."""
import numpy as np
import pytest

from src.config import load_config
from src.index import build_index
from src.schema import SearchRequest
from tests.fakes import COLORS, solid, tag_engine

RED = COLORS[1]


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["index"]["cluster"].update(leaders=2, b2=2)
    cfg["rerank"]["mmr"]["depth"] = 4
    return cfg


def _with_ann(cfg, kinds=("cluster", "hnsw")):
    eng = tag_engine(cfg)
    for rep in ("clip", "dinov2"):
        flat = eng.indexes[rep]
        for kind in kinds:
            eng.ann_indexes[(rep, kind)] = build_index(kind, flat.vectors(), flat.ids, "ip", cfg)
    return eng


def _search(eng, rep="clip", task="semantic", color=RED, **kw):
    return eng.search(SearchRequest(solid(color), task=task, representation=rep, **kw))


def test_index_kinds_listed_per_rep(cfg):
    eng = _with_ann(cfg)
    assert eng.index_kinds("clip") == ["flat", "cluster", "hnsw"]
    assert eng.index_kinds("fusion") == ["flat", "cluster", "hnsw"]  # fusion's dense line is clip
    assert eng.index_kinds("tags") == ["flat"]


@pytest.mark.parametrize("kind", ["cluster", "hnsw"])
def test_ann_search_matches_flat_top1(cfg, kind):
    eng = _with_ann(cfg)
    for rep in ("clip", "fusion"):
        flat = _search(eng, rep, k=3)
        ann = _search(eng, rep, k=3, index=kind)
        assert ann.index == kind and ann.results[0].image_id == flat.results[0].image_id


def test_missing_or_unsupported_index_raises(cfg):
    eng = tag_engine(cfg)
    with pytest.raises(ValueError, match="hnsw"):
        _search(eng, "clip", index="hnsw")
    with pytest.raises(ValueError, match="flat"):
        _search(_with_ann(cfg), "tags", index="hnsw")


def test_rerank_needs_vectors(cfg):
    with pytest.raises(ValueError, match="vector"):
        _search(tag_engine(cfg), "tags", rerank=("aqe",))


def test_aqe_and_mmr_run_and_are_timed(cfg):
    eng = tag_engine(cfg)
    for rep in ("clip", "fusion"):
        resp = _search(eng, rep, k=4, rerank=("aqe", "mmr"))
        assert resp.rerank == ("aqe", "mmr") and len(resp.results) == 4
        assert len({r.image_id for r in resp.results}) == 4
        assert resp.timings_ms["rerank"] > 0


def test_rocchio_without_feedback_is_a_noop_with_a_note(cfg):
    eng = tag_engine(cfg)
    plain = [r.image_id for r in _search(eng, k=4).results]
    resp = _search(eng, k=4, rerank=("rocchio",))
    assert [r.image_id for r in resp.results] == plain and resp.notes


def test_rocchio_feedback_moves_ranking(cfg):
    eng = tag_engine(cfg)
    base = [r.image_id for r in _search(eng, k=4).results]
    # red query; marking blue (3) relevant and the others non-relevant should lift blue
    resp = _search(eng, k=4, rerank=("rocchio",), feedback=([3], [1, 4]))
    got = [r.image_id for r in resp.results]
    assert got.index(3) < base.index(3)
    # unknown ids in the feedback are ignored, not an error
    _search(eng, k=4, rerank=("rocchio",), feedback=([999], []))


def test_add_image_reaches_ann_indexes_loaded_before_and_after(cfg, tmp_path):
    eng = _with_ann(cfg, kinds=("hnsw",))
    flat = eng.indexes["clip"]
    later = build_index("cluster", flat.vectors(), flat.ids, "ip", cfg)
    later.save(tmp_path / "cluster")
    eng._index_dirs[("clip", "cluster")] = tmp_path / "cluster"  # lazily loaded after the add
    cyan = (0, 200, 200)
    new_id = eng.add_image(solid(cyan))
    for kind in ("flat", "hnsw", "cluster"):
        assert _search(eng, "clip", task="similar", color=cyan, k=1, index=kind
                       ).results[0].image_id == new_id, kind
    np.testing.assert_allclose(np.linalg.norm(eng.vectors_of("clip", [new_id]), axis=1), 1, rtol=1e-5)
