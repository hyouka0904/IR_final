import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.edits import EDIT_TYPES
from src.evaluate import evaluate_semantic, evaluate_similar, write_semantic_qrels
from src.index import FlatIndex
from src.pipeline import SearchEngine
from src.trec import TREC_REL_LEVEL
from tests.test_pipeline import MeanColorEncoder

GALLERY = [
    {"image_id": 1, "file_name": "1.png", "categories": [1]},
    {"image_id": 2, "file_name": "2.png", "categories": [2]},
    {"image_id": 3, "file_name": "3.png", "categories": [1, 2]},
    {"image_id": 4, "file_name": "4.png", "categories": [3]},
    {"image_id": 5, "file_name": "5.png", "categories": [4]},
]
COLORS = {1: (250, 0, 0), 2: (0, 250, 0), 3: (200, 200, 0), 4: (0, 0, 250), 5: (90, 90, 90)}


@pytest.fixture
def setup(tmp_path):
    cfg = load_config()
    gdir, qdir = tmp_path / "train", tmp_path / "val"
    gdir.mkdir(), qdir.mkdir()
    imgs = []
    for r in GALLERY:
        img = Image.new("RGB", (64, 64), COLORS[r["image_id"]])
        img.save(gdir / r["file_name"])
        imgs.append(img)
    enc = MeanColorEncoder("clip")
    idx = FlatIndex.build(enc.encode(imgs), np.array([r["image_id"] for r in GALLERY]), "ip")
    engine = SearchEngine(cfg, {"clip": enc}, {"clip": idx})
    return cfg, engine, gdir, qdir


BLOCKS = {1: (0, 0, 32, 32), 2: (32, 0, 64, 32), 3: (0, 32, 32, 64), 4: (32, 32, 64, 64),
          5: (16, 16, 48, 48)}


class LayoutEncoder:
    """Test double robust to the synthetic edits: coarse 4x4 grayscale layout."""
    name = "dinov2"
    metric = "ip"

    def encode(self, images):
        v = np.array([np.asarray(im.convert("L").resize((4, 4), Image.Resampling.BOX),
                                 dtype=np.float32).ravel() for im in images])
        v -= v.mean(axis=1, keepdims=True)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def layout_setup(tmp_path):
    cfg = load_config()
    gdir = tmp_path / "layout"
    gdir.mkdir()
    imgs = []
    for r in GALLERY:
        img = Image.new("RGB", (64, 64), (30, 30, 30))
        img.paste((230, 230, 230), BLOCKS[r["image_id"]])
        img.save(gdir / r["file_name"])
        imgs.append(img)
    enc = LayoutEncoder()
    idx = FlatIndex.build(enc.encode(imgs), np.array([r["image_id"] for r in GALLERY]), "ip")
    return cfg, SearchEngine(cfg, {"dinov2": enc}, {"dinov2": idx}), gdir


def _query(qdir, qid, color, cats):
    Image.new("RGB", (40, 40), color).save(qdir / f"q{qid}.png")
    return {"image_id": qid, "file_name": f"q{qid}.png", "categories": cats}


def test_semantic_perfect_ranking_scores_one(setup):
    cfg, engine, _, qdir = setup
    queries = [_query(qdir, 100, (250, 0, 0), [1])]
    summary, rows = evaluate_semantic(engine, queries, GALLERY, qdir, "clip", cfg)
    # relevant (gain >= 0.5) for query cats {1}: images 1 and 3; red query ranks 1 first.
    assert rows[0]["ndcg@10"] == pytest.approx(1.0, abs=0.05)
    assert summary["n_queries"] == 1 and summary["n_skipped"] == 0
    assert 0 < summary["map"] <= 1
    assert {"embed_ms", "search_ms", "rerank_ms"} <= set(summary)


def test_semantic_skips_queries_with_no_relevant(setup):
    cfg, engine, _, qdir = setup
    queries = [_query(qdir, 100, (250, 0, 0), [1]), _query(qdir, 101, (0, 250, 0), [99])]
    summary, rows = evaluate_semantic(engine, queries, GALLERY, qdir, "clip", cfg)
    assert summary["n_queries"] == 2 and summary["n_skipped"] == 1
    assert np.isnan(rows[1]["ap"])
    assert np.isfinite(summary["map"])


def test_semantic_writes_trec_run(setup, tmp_path):
    cfg, engine, _, qdir = setup
    queries = [_query(qdir, 100, (250, 0, 0), [1])]
    evaluate_semantic(engine, queries, GALLERY, qdir, "clip", cfg, runs_dir=tmp_path / "runs")
    write_semantic_qrels(tmp_path / "runs" / "semantic.qrels", queries, GALLERY,
                         cfg["eval"]["semantic_binary_threshold"])
    run = (tmp_path / "runs" / "semantic_clip.run").read_text().splitlines()
    qrels = (tmp_path / "runs" / "semantic.qrels").read_text().splitlines()
    assert run[0].split()[:4] == ["100", "Q0", "1", "1"]
    assert "100 0 1 3" in qrels and "100 0 3 3" in qrels  # grade = round(3 * gain)
    assert not any(line.split()[2] in ("2", "4", "5") for line in qrels)  # gain 0 omitted


def test_similar_edited_original_found_at_rank1(layout_setup):
    cfg, engine, gdir = layout_setup
    summary, rows = evaluate_similar(engine, GALLERY, gdir, "dinov2", cfg, n=5)
    assert summary["recall@1"] == 1.0 and summary["mrr"] == 1.0
    assert len(rows) == 5 and {r["edit"] for r in rows} == set(EDIT_TYPES)


def test_similar_summary_has_per_edit_keys(layout_setup):
    cfg, engine, gdir = layout_setup
    summary, _ = evaluate_similar(engine, GALLERY, gdir, "dinov2", cfg, n=5)
    for e in EDIT_TYPES:
        assert {f"recall@1_{e}", f"recall@10_{e}", f"mrr_{e}"} <= set(summary)
    assert summary["task"] == "similar" and summary["representation"] == "dinov2"


def _trec_map(run_path, qrels_path, level):
    """trec_eval-style MAP: docs ranked by score desc (ties: docno desc), rel = grade >= level,
    queries without any relevant doc are skipped."""
    rel = {}
    for line in qrels_path.read_text().splitlines():
        q, _, d, g = line.split()
        if int(g) >= level:
            rel.setdefault(q, set()).add(d)
    runs = {}
    for line in run_path.read_text().splitlines():
        q, _, d, _, s, _ = line.split()
        runs.setdefault(q, []).append((float(s), d))
    aps = []
    for q, docs in rel.items():
        ranked = [d for _, d in sorted(runs.get(q, []), key=lambda x: (x[0], x[1]), reverse=True)]
        hits, total = 0, 0.0
        for i, d in enumerate(ranked):
            if d in docs:
                hits += 1
                total += hits / (i + 1)
        aps.append(total / len(docs))
    return sum(aps) / len(aps)


@pytest.mark.parametrize("threshold", [0.5, 0.6])
def test_trec_files_reproduce_map(setup, tmp_path, threshold):
    cfg, engine, _, qdir = setup
    cfg["eval"]["semantic_binary_threshold"] = threshold
    queries = [_query(qdir, 100, (250, 0, 0), [1, 2]), _query(qdir, 101, (0, 250, 0), [1, 2, 3])]
    runs = tmp_path / "runs"
    summary, _ = evaluate_semantic(engine, queries, GALLERY, qdir, "clip", cfg, runs_dir=runs)
    write_semantic_qrels(runs / "semantic.qrels", queries, GALLERY, threshold)
    assert _trec_map(runs / "semantic_clip.run", runs / "semantic.qrels",
                     TREC_REL_LEVEL) == pytest.approx(summary["map"])


def test_evaluate_semantic_does_not_write_qrels(setup, tmp_path):
    cfg, engine, _, qdir = setup
    queries = [_query(qdir, 100, (250, 0, 0), [1])]
    evaluate_semantic(engine, queries, GALLERY, qdir, "clip", cfg, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs" / "semantic.qrels").exists()


def test_gallery_index_mismatch_raises(setup, layout_setup):
    cfg, engine, _, qdir = setup
    bigger = GALLERY + [{"image_id": 6, "file_name": "6.png", "categories": [1]}]
    with pytest.raises(ValueError, match="index"):
        evaluate_semantic(engine, [_query(qdir, 100, (250, 0, 0), [1])], bigger, qdir, "clip", cfg)
    cfg2, engine2, gdir = layout_setup
    with pytest.raises(ValueError, match="index"):
        evaluate_similar(engine2, GALLERY[:4], gdir, "dinov2", cfg2, n=4)


def test_semantic_eval_runs_for_fusion(tmp_path):
    from tests.fakes import COLORS as TAG_COLORS, solid, tag_engine
    cfg = load_config()
    engine = tag_engine(cfg)
    gallery = [{"image_id": i, "file_name": f"{i}.png", "categories": [i]} for i in TAG_COLORS]
    qdir = tmp_path / "val"
    qdir.mkdir()
    queries = []
    for i, c in TAG_COLORS.items():
        solid(c).save(qdir / f"q{i}.png")
        queries.append({"image_id": 100 + i, "file_name": f"q{i}.png", "categories": [i]})
    for rep in ("fusion", "tags"):
        s, rows = evaluate_semantic(engine, queries, gallery, qdir, rep, cfg)
        assert s["representation"] == rep and s["n_queries"] == 4
        assert 0 < s["ndcg@10"] <= 1 and len(rows) == 4
