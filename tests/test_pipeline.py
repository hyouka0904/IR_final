"""共用 pipeline 與評估流程：以合成的小圖庫與快取特徵跑完整流程（不需要 torch 與模型權重）。"""
import copy

import numpy as np
import pytest

from src import features
from src.config import load_config
from src.data.sets import CategoryQrels, load_set, save_set
from src.eval.protocol import frozen_ranking, run_queries, simulate_feedback
from src.features import TagSet
from src.index import faiss_index
from src.pipeline import Pipeline, parse_config_id, text_index_path, without_post
from src.schema import SearchRequest
from src.text.inverted_index import InvertedIndex

N, D, V, NQ = 80, 16, 12, 6


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cache")
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["cache_root"] = str(tmp)
    vocab = [f"concept {i}" for i in range(V)]
    (tmp / "vocab.txt").write_text("\n".join(vocab))
    cfg["tags"].update(vocab=str(tmp / "vocab.txt"), top_n=4)
    cfg["index"]["ivfpq"].update(nlist=4, m=4, nbits=4, nprobe=4)
    cfg["mmr"]["depth"] = 20
    rng = np.random.default_rng(0)

    save_set(str(tmp), "g", [{"id": f"g{i}", "path": f"g{i}.jpg"} for i in range(N)], {"kind": "gallery"})
    db = rng.standard_normal((N, D)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    (tmp / "features" / "g").mkdir(parents=True)
    np.save(features.feature_path(str(tmp), "g", "clip"), db)
    ph = rng.integers(0, 255, (N, 8), dtype=np.uint8)
    np.save(features.feature_path(str(tmp), "g", "phash"), ph)
    top = np.stack([rng.permutation(V)[:4] for _ in range(N)]).astype(np.int32)
    probs = np.sort(rng.random((N, 4)).astype(np.float32), axis=1)[:, ::-1] / 2
    TagSet.save(str(tmp), "g", top, probs, cfg["tags"])
    tags = TagSet.load(str(tmp), "g", vocab, cfg["tags"])
    import os
    os.makedirs(os.path.dirname(text_index_path(str(tmp), "g")))
    InvertedIndex.build([tags.terms(i) for i in range(N)]).save(text_index_path(str(tmp), "g"))
    for kind in faiss_index.INDEX_KINDS:
        faiss_index.save_index(faiss_index.build_index(kind, db, cfg), faiss_index.index_file(str(tmp), "g", "clip", kind))
    faiss_index.save_index(faiss_index.build_index("flat", ph, cfg), faiss_index.index_file(str(tmp), "g", "phash", "flat"))

    # query：圖庫圖片加一點雜訊；第 0 個 query 排除自己（模擬 Holidays）
    qidx = list(range(NQ))
    save_set(str(tmp), "q", [{"id": f"q{i}", "path": "x", **({"exclude_idx": 0} if i == 0 else {})} for i in qidx],
             {"kind": "queries", "gallery": "g"})
    q = db[qidx] + 0.05 * rng.standard_normal((NQ, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qfeats = [{"clip": q[i], "phash": ph[i], "tags": (top[i], probs[i])} for i in qidx]
    g_cats = (rng.random((N, 5)) < 0.3).astype(np.uint8)
    qrels = CategoryQrels(g_cats[qidx], g_cats, 2)
    return Pipeline("g", cfg), load_set(str(tmp), "q"), qfeats, qrels


CONFIGS = ["clip-flat-none", "clip-hnsw-none", "clip-ivfpq-none", "clip-cp-none", "phash-flat-none", "bm25-inv-none",
           "clip+bm25-flat-none", "clip-flat-aqe", "clip-flat-mmr", "clip+bm25-flat-aqe+mmr"]


def test_parse_config_id():
    assert parse_config_id("siglip+bm25-flat-aqe+mmr")["vec"] == "siglip"
    assert parse_config_id("bm25-inv-none")["vec"] is None
    for bad in ["bm25-flat-none", "phash-hnsw-none", "phash-flat-aqe", "clip-flat-mmr+aqe", "bm25-inv-mmr", "foo-flat-none", "clip-flat"]:
        with pytest.raises(ValueError):
            parse_config_id(bad)
    assert without_post("clip-flat-aqe+rocchio", "rocchio") == "clip-flat-aqe"
    assert without_post("clip-flat-rocchio", "rocchio") == "clip-flat-none"


@pytest.mark.parametrize("cid", CONFIGS)
def test_search(env, cid):
    pipe, queries, qfeats, _ = env
    res = pipe.search(SearchRequest(cid, k=10), qfeats[1])
    assert 0 < len(res.ids) <= 10 and len(set(res.ids.tolist())) == len(res.ids)
    assert res.ids.min() >= 0 and len(res.scores) == len(res.ids)
    assert "search" in res.latency_ms and "embed" not in res.latency_ms
    if cid in ("clip-flat-none", "clip-hnsw-none", "phash-flat-none"):
        assert res.ids[0] == 1                       # 加了雜訊的自己應排第一
    if "bm25" in cid:
        assert res.query_tags is not None and len(res.query_tags) == 4


def test_exclude(env):
    pipe, _, qfeats, _ = env
    res = pipe.search(SearchRequest("clip-flat-none", k=N, exclude=[0]), qfeats[0])
    assert 0 not in res.ids.tolist() and len(res.ids) == N - 1


def test_rocchio_feedback_changes_query(env):
    pipe, _, qfeats, _ = env
    base = pipe.search(SearchRequest("clip-flat-rocchio", k=10), qfeats[1])
    assert base.extra["rocchio"] == "skipped: no feedback"
    fb = pipe.search(SearchRequest("clip-flat-rocchio", k=10, feedback=([50, 51], [1])), qfeats[1])
    assert not np.array_equal(base.extra["query_vector"], fb.extra["query_vector"])


def test_matched_tags(env):
    pipe, _, qfeats, _ = env
    res = pipe.search(SearchRequest("bm25-inv-none", k=5), qfeats[2])
    terms = res.extra["query_terms"]
    assert pipe.matched_tags(terms, int(res.ids[0]))      # 第一名至少有一個共同 tag


def test_feedback_protocol():
    grades = np.array([0, 2, 0, 1, 0])
    assert simulate_feedback(np.array([0, 1, 2, 3]), grades, 3, 1) == ([1], [0, 2])
    assert frozen_ranking(np.array([4, 3, 2]), np.array([1, 4, 0, 3]), 2, 4).tolist() == [4, 3, 1, 0]


@pytest.mark.parametrize("cid", ["clip-flat-none", "clip-flat-rocchio", "clip+bm25-flat-mmr"])
def test_run_queries(env, cid):
    pipe, queries, qfeats, qrels = env
    rankings, per_query, lat = run_queries(pipe, queries, cid, qfeats, qrels, depth=30)
    assert len(rankings) == len(per_query) == len(lat) == NQ
    assert 0 not in rankings[0].tolist()                   # exclude_idx 生效
    assert all("cat_coverage_10" in m and "ndcg_cut_10" in m for m in per_query)
    if cid == "clip-flat-rocchio":
        init, _, _ = run_queries(pipe, queries, "clip-flat-none", qfeats, qrels, depth=30)
        d = pipe.cfg["rocchio"]["feedback_depth"]
        assert all(np.array_equal(a[:d], b[:d]) for a, b in zip(init, rankings))   # frozen：前 d 名不變
