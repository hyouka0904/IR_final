"""Configurations, the Rocchio protocol, recall loss and the box / Holidays / phone sets."""
import numpy as np
import pytest

from src.config import load_config
from src.evaluate import (EvalQuery, config_label, evaluate_box, evaluate_holidays,
                          evaluate_phone, evaluate_semantic, parse_config, run_query)
from src.index import build_index
from tests.fakes import COLORS, solid, tag_engine


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg["search"]["eval_depth"] = 4
    cfg["rerank"]["rocchio"]["feedback_depth"] = 2
    cfg["rerank"]["mmr"]["depth"] = 4
    return cfg


def _gallery(tmp_path):
    gallery = [{"image_id": i, "file_name": f"{i}.png", "categories": [i]} for i in COLORS]
    d = tmp_path / "imgs"
    d.mkdir()
    for i, c in COLORS.items():
        solid(c).save(d / f"{i}.png")
    return gallery, d


def test_config_labels_roundtrip():
    assert config_label("clip") == "clip"
    assert config_label("fusion", "hnsw", ("aqe", "mmr")) == "fusion@hnsw+aqe+mmr"
    assert parse_config("fusion@hnsw+mmr+aqe") == ("fusion", "hnsw", ("aqe", "mmr"))
    assert parse_config("tags_nococo") == ("tags_nococo", "flat", ())
    with pytest.raises(ValueError):
        parse_config("clip@lsh")
    with pytest.raises(ValueError):
        parse_config("clip+boost")


def test_rocchio_protocol_freezes_judged_results(cfg):
    eng = tag_engine(cfg)
    img = solid(COLORS[1])
    q = EvalQuery("q", lambda: img, relevant=lambda d: d == 3)
    plain, _ = run_query(eng, "semantic", q, "clip", "flat", (), 4, cfg)
    hits, t = run_query(eng, "semantic", q, "clip", "flat", ("rocchio",), 4, cfg)
    assert [d for d, _ in hits[:2]] == [d for d, _ in plain[:2]]  # the judged top 2 stay put
    assert sorted(d for d, _ in hits) == sorted(COLORS)
    assert t["search"] > 0  # both rounds counted


def test_exclude_drops_the_query_itself(cfg):
    eng = tag_engine(cfg)
    q = EvalQuery("q", lambda: solid(COLORS[2]), relevant=lambda d: False, exclude=frozenset({2}))
    hits, _ = run_query(eng, "similar", q, "dinov2", "flat", (), 4, cfg)
    assert 2 not in [d for d, _ in hits] and len(hits) == 3


def test_semantic_reports_coverage_and_config(cfg, tmp_path):
    eng = tag_engine(cfg)
    gallery, d = _gallery(tmp_path)
    queries = [{"image_id": 100, "file_name": "1.png", "categories": [1, 4]}]
    s, rows = evaluate_semantic(eng, queries, gallery, d, "fusion", cfg, rerank=("mmr",))
    assert s["config"] == "fusion+mmr" and s["rerank"] == "mmr" and s["recall_loss@10"] == 0.0
    assert 0 <= s["coverage@10"] <= 1 and rows[0]["coverage@10"] == 1.0  # top 4 = whole gallery


def test_ann_config_reports_recall_loss(cfg, tmp_path):
    eng = tag_engine(cfg)
    flat = eng.indexes["clip"]
    eng.ann_indexes[("clip", "hnsw")] = build_index("hnsw", flat.vectors(), flat.ids, "ip", cfg)
    gallery, d = _gallery(tmp_path)
    queries = [{"image_id": 100, "file_name": "1.png", "categories": [1]}]
    s, _ = evaluate_semantic(eng, queries, gallery, d, "clip", cfg, index="hnsw")
    assert s["config"] == "clip@hnsw" and s["recall_loss@10"] == pytest.approx(0.0)


def test_box_vs_full_and_binary_relevance(cfg, tmp_path):
    eng = tag_engine(cfg)
    gallery, d = _gallery(tmp_path)
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    img[:, :20] = COLORS[1]
    img[:, 20:] = COLORS[3]
    from PIL import Image
    Image.fromarray(img).save(d / "q.png")
    boxes = [{"image_id": 100, "file_name": "q.png", "category": 3, "crop": [20, 0, 40, 20]}]
    box, rows = evaluate_box(eng, boxes, gallery, d, "clip", cfg, crop=True)
    full, _ = evaluate_box(eng, boxes, gallery, d, "clip", cfg, crop=False)
    assert box["task"] == "box" and full["task"] == "box_full"
    assert rows[0]["qid"] == "100_c3" and box["map"] == 1.0  # the crop is all blue
    assert full["map"] <= box["map"]


def test_holidays_groups_and_self_exclusion(cfg, tmp_path):
    eng = tag_engine(cfg)
    gallery = [{"image_id": i, "file_name": f"{i}.png", "source": "holidays",
                "group": "a" if i in (1, 4) else f"g{i}", "categories": []} for i in COLORS]
    d = tmp_path / "h"
    d.mkdir()
    for i, c in COLORS.items():
        solid(c).save(d / f"{i}.png")
    queries = [gallery[0]]  # red; its group-mate 4 is yellow (red + green)
    s, rows = evaluate_holidays(eng, queries, gallery, "dinov2", cfg, lambda r: d / r["file_name"])
    assert s["task"] == "holidays" and rows[0]["qid"] == "1"
    assert rows[0]["recall@10"] == 1.0 and 0 < s["map"] <= 1


def test_phone_graded_qrels(cfg, tmp_path):
    eng = tag_engine(cfg)
    gallery, d = _gallery(tmp_path)
    photo = tmp_path / "IMG_1.png"
    solid(COLORS[1]).save(photo)
    s, rows = evaluate_phone(eng, [photo], {"IMG_1": {1: 2, 4: 1}}, gallery, "clip", cfg)
    assert s["task"] == "phone" and rows[0]["qid"] == "IMG_1"
    assert rows[0]["ap"] == 1.0 and rows[0]["ndcg@10"] == pytest.approx(1.0)
