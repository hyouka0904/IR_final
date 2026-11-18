"""inverted index、BM25、索引（cluster pruning / pHash / FAISS）、後處理（alpha-QE / Rocchio / MMR / RRF）。"""
import numpy as np
import pytest
from PIL import Image

from src.index import faiss_index
from src.index.cluster_pruning import ClusterPruning
from src.models import phash
from src.rerank.aqe import alpha_qe
from src.rerank.mmr import mmr
from src.rerank.rocchio import rocchio
from src.rerank.rrf import rrf
from src.text.bm25 import BM25
from src.text.inverted_index import InvertedIndex

CFG = {"index": {"hnsw": {"m": 16, "ef_construction": 40, "ef_search": 32},
                 "ivfpq": {"nlist": 8, "m": 8, "nbits": 4, "nprobe": 8},
                 "cp": {"n_leaders": None, "b1": 1, "b2": 4}}}


def _unit(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


DOCS = [{"cat": 3, "sofa": 1}, {"dog": 2}, {"cat": 1, "dog": 1, "ball": 1}, {"sofa": 2}]


def test_inverted_index(tmp_path):
    idx = InvertedIndex.build(DOCS)
    assert idx.df("cat") == 2 and idx.df("zebra") == 0
    assert idx.postings["dog"][0].tolist() == [1, 2]
    assert idx.boolean_and(["cat", "dog"]).tolist() == [2]
    assert idx.boolean_or(["sofa", "ball"]).tolist() == [0, 2, 3]
    assert idx.matched_terms({"sofa": 1, "cat": 5}, 0) == ["cat", "sofa"]
    idx.save(tmp_path / "i.pkl")
    assert InvertedIndex.load(tmp_path / "i.pkl").df("cat") == 2


def test_bm25_matches_formula():
    idx = InvertedIndex.build(DOCS)
    bm = BM25(idx, k1=1.2, b=0.75, k3=8.0)
    q = {"cat": 1, "sofa": 2}
    avgdl = np.mean([4, 2, 3, 2])

    def term(tf, dl, df, qtf):
        idf = np.log(1 + (4 - df + 0.5) / (df + 0.5))
        return idf * (9 * qtf / (8 + qtf)) * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))

    expected = [term(3, 4, 2, 1) + term(1, 4, 2, 2), 0.0, term(1, 3, 2, 1), term(2, 2, 2, 2)]
    assert bm.score_all(q) == pytest.approx(np.array(expected, dtype=np.float32), rel=1e-5)
    ids, scores = bm.search(q, 10)
    assert 1 not in ids.tolist()                       # 沒有共同 term 的文件不算被檢索到
    assert ids.tolist() == np.argsort(-np.array(expected))[:3].tolist()


def test_cluster_pruning(tmp_path):
    rng = np.random.default_rng(0)
    x = _unit(rng, 900, 16)
    cp = ClusterPruning.build(x, None, b1=1, b2=4, seed=0)
    assert len(cp.leaders) == 30
    assert sorted(np.concatenate(cp.followers).tolist()) == list(range(900))   # b1 = 1：每篇恰好一個 cluster
    s, ids = cp.search(x[:5], 10)
    assert ids[:, 0].tolist() == list(range(5))      # 自己一定在自己 leader 的 cluster 裡附近
    full = ClusterPruning(x, cp.leaders, cp.followers, b2=30)
    _, exact = full.search(x[5:10], 10)
    assert np.array_equal(exact, np.argsort(-(x[5:10] @ x.T), axis=1)[:, :10])   # 檢查全部 leader = 精確搜尋
    cp.save(str(tmp_path / "cp.npz"))
    cp2 = ClusterPruning.load(str(tmp_path / "cp.npz"))
    assert np.array_equal(cp2.search(x[:3], 5)[1], cp.search(x[:3], 5)[1])


@pytest.mark.parametrize("kind", faiss_index.INDEX_KINDS)
def test_vector_indexes(kind, tmp_path):
    rng = np.random.default_rng(0)
    x = _unit(rng, 500, 32)
    idx = faiss_index.build_index(kind, x, CFG)
    path = str(tmp_path / f"v_{kind}{'.npz' if kind == 'cp' else '.index'}")
    faiss_index.save_index(idx, path)
    idx = faiss_index.load_index(path, kind, CFG)
    s, ids = faiss_index.search(idx, x[:4], 20)
    assert s.shape == ids.shape == (4, 20)
    if kind in ("flat", "hnsw", "cp"):
        assert ids[:, 0].tolist() == [0, 1, 2, 3]


def test_phash_binary_index(tmp_path):
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(20)]
    db = phash.compute(images)
    assert db.shape == (20, 8) and db.dtype == np.uint8
    idx = faiss_index.build_index("flat", db, CFG)
    faiss_index.save_index(idx, str(tmp_path / "p.index"))
    idx = faiss_index.load_index(str(tmp_path / "p.index"), "flat", CFG, binary=True)
    s, ids = faiss_index.search(idx, db[3], 5)
    assert ids[0, 0] == 3 and s[0, 0] == 1.0
    assert s[0, 1] == pytest.approx(1 - phash.hamming(db[3], db[ids[0, 1]]) / 64)
    with pytest.raises(ValueError):
        faiss_index.build_index("hnsw", db, CFG)


def test_aqe_rocchio():
    rng = np.random.default_rng(0)
    db = _unit(rng, 50, 8)
    q = db[0]
    q2 = alpha_qe(q, db, np.array([0, 1, -1]), 3, 3.0)
    assert np.linalg.norm(q2) == pytest.approx(1)
    r = rocchio(q, db, [1, 2], [3], 1.0, 0.75, 0.15)
    expected = q + 0.75 * db[[1, 2]].mean(0) - 0.15 * db[3]
    assert r == pytest.approx(expected / np.linalg.norm(expected), abs=1e-6)
    assert rocchio(q, db, [], [], 1, 1, 1) == pytest.approx(q, abs=1e-6)


def test_mmr_diversifies():
    # 0、1 幾乎相同，2 不同：λ 小時第二名應該換成 2
    db = np.array([[1, 0], [0.999, 0.0447], [0, 1]], dtype=np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    ids, _ = mmr(np.array([0, 1, 2]), np.array([0.9, 0.89, 0.5]), db, lam=0.5, depth=3)
    assert ids.tolist() == [0, 2, 1]
    ids, _ = mmr(np.array([0, 1, 2]), np.array([0.9, 0.89, 0.5]), db, lam=1.0, depth=3)
    assert ids.tolist() == [0, 1, 2]
    ids, _ = mmr(np.array([0, 1, 2, 7]), np.array([0.9, 0.89, 0.5, 0.1]), np.vstack([db, np.zeros((5, 2))]), 0.5, 3)
    assert ids[-1] == 7                              # depth 之後維持原順序


def test_rrf():
    ids, scores = rrf([np.array([1, 2, 3]), np.array([3, 1, -1])], k=60)
    assert ids.tolist() == [1, 3, 2]
    assert scores[0] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf([np.array([-1])])[0].size == 0
