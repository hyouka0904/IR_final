import numpy as np
import pytest

from src.config import load_config
from src.index import (ClusterPruningIndex, FaissANNIndex, FlatIndex, build_index, index_size_bytes,
                       load_index)
from src.schema import INDEX_KINDS

CFG = load_config()


def _gallery(n=600, d=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v, np.arange(1000, 1000 + n)


@pytest.mark.parametrize("kind", INDEX_KINDS)
def test_every_kind_finds_itself(kind):
    v, ids = _gallery()
    idx = build_index(kind, v, ids, "ip", CFG)
    assert idx.kind == kind and idx.ntotal == len(v)
    scores, got = idx.search(v[:20], 5)
    assert got.shape[1] <= 5
    hit = np.mean([ids[i] in got[i] for i in range(20)])
    assert hit >= 0.9, kind  # a stored vector is its own nearest neighbour
    assert (np.diff(scores, axis=1) <= 1e-5).all()  # scores descending


@pytest.mark.parametrize("kind", ["cluster", "ivfpq", "hnsw"])
def test_ann_recall_against_flat(kind):
    v, ids = _gallery(n=800)
    rng = np.random.default_rng(1)
    q = rng.standard_normal((30, v.shape[1])).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    _, exact = FlatIndex.build(v, ids, "ip").search(q, 10)
    _, approx = build_index(kind, v, ids, "ip", CFG).search(q, 10)
    recall = np.mean([len(set(a) & set(e)) / 10 for a, e in zip(approx, exact)])
    assert recall > 0.3, (kind, recall)  # approximate, but far better than random (~0.0125)


@pytest.mark.parametrize("kind", INDEX_KINDS)
def test_save_load_same_results_and_add(tmp_path, kind):
    v, ids = _gallery(n=400)
    idx = build_index(kind, v, ids, "ip", CFG)
    idx.save(tmp_path / kind)
    loaded = load_index(tmp_path / kind, CFG)
    assert type(loaded) is type(idx) and loaded.kind == kind
    s1, i1 = idx.search(v[:5], 7)
    s2, i2 = loaded.search(v[:5], 7)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(s1, s2, rtol=1e-5)
    assert index_size_bytes(tmp_path / kind) > 0
    new = v[:1] * -1  # far from everything else
    loaded.add(new, np.array([42]))
    assert loaded.ntotal == 401 and 42 in loaded.search(new, 3)[1][0]


def test_cluster_pruning_structure():
    v, ids = _gallery(n=100)
    idx = ClusterPruningIndex.build(v, ids, {"leaders": None, "b1": 1, "b2": 2}, seed=3)
    assert len(idx.leaders) == 10  # round(sqrt(100))
    members = sorted(r for f in idx.followers for r in f)
    assert members == list(range(100))  # b1 = 1: every vector in exactly one cluster
    for j, row in enumerate(idx.leaders):
        assert row in idx.followers[j]


def test_config_search_params_override_on_load(tmp_path):
    v, ids = _gallery(n=300)
    FaissANNIndex.build(v, ids, "hnsw", CFG["index"]["hnsw"]).save(tmp_path)
    loaded = load_index(tmp_path, {"index": {"hnsw": {"ef_search": 7}}})
    assert loaded.index.hnsw.efSearch == 7


def test_phash_only_flat():
    codes = np.zeros((10, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="flat"):
        build_index("hnsw", codes, np.arange(10), "hamming", CFG)
    assert build_index("flat", codes, np.arange(10), "hamming", CFG).metric == "hamming"


def test_ivfpq_m_must_divide_dim():
    v, ids = _gallery(n=300, d=30)
    with pytest.raises(ValueError, match="divide"):
        build_index("ivfpq", v, ids, "ip", CFG)


def test_week1_flat_cache_still_loads(tmp_path):
    v, ids = _gallery(n=50)
    FlatIndex.build(v, ids, "ip").save(tmp_path)  # meta.json has no "kind"
    assert isinstance(load_index(tmp_path), FlatIndex)
