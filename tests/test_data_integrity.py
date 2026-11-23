"""資料完整性檢查：先用合成資料確認檢查本身抓得到問題，再（資料存在時）檢查真實資料。

真實資料的測試在對應資料集不存在時會自動略過，所以沒有資料的開發環境也能跑整個 tests/。
"""
import json
import pickle

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.data import integrity
from src.data.sets import CategoryQrels, SparseQrels, save_set, set_dir


def write_jpeg(path, seed=0, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path, "JPEG")
    return path


def levels(sec):
    return [level for level, _ in sec.items]


# ---------------- 圖片 ----------------

def test_image_problem_detects_damage(tmp_path):
    good = write_jpeg(tmp_path / "good.jpg")
    assert integrity.image_problem(str(good)) is None
    assert integrity.image_problem(str(good), full=True) is None

    truncated = tmp_path / "truncated.jpg"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])      # 下載或解壓被截斷
    assert integrity.image_problem(str(truncated)) is not None

    (tmp_path / "empty.jpg").write_bytes(b"")
    assert integrity.image_problem(str(tmp_path / "empty.jpg")) == "大小為 0"
    (tmp_path / "junk.jpg").write_bytes(b"<html>404 Not Found</html>")         # 下載到錯誤頁面
    assert integrity.image_problem(str(tmp_path / "junk.jpg")) is not None
    assert integrity.image_problem(str(tmp_path / "missing.jpg")) == "不存在"


def test_image_problem_accepts_trailing_bytes(tmp_path):
    """有些正常的 JPEG 在 EOI 之後還有資料：快速檢查要解碼確認，不能直接判定損毀。"""
    p = write_jpeg(tmp_path / "a.jpg")
    p.write_bytes(p.read_bytes() + b"extra trailer data")
    assert integrity.image_problem(str(p)) is None


def test_check_images_reports_examples(tmp_path):
    paths = [str(write_jpeg(tmp_path / f"{i}.jpg", i)) for i in range(3)] + [str(tmp_path / "gone.jpg")]
    sec = integrity.Section("x")
    integrity.check_images(sec, "imgs", paths)
    assert sec.failed and "1 / 4" in sec.items[0][1] and "gone.jpg" in sec.items[0][1]


# ---------------- 原始資料集 ----------------

def test_check_partials(tmp_path):
    assert not integrity.check_partials(str(tmp_path)).failed
    (tmp_path / "holidays").mkdir()
    (tmp_path / "holidays" / "jpg1.tar.gz.part").write_bytes(b"x")
    assert integrity.check_partials(str(tmp_path)).failed


def make_holidays(root, n_groups=3, per_group=2):
    for g in range(n_groups):
        for k in range(per_group):
            write_jpeg(root / "holidays" / "jpg" / f"{1000 + g}{k:02d}.jpg", g * 10 + k)


def test_check_holidays(tmp_path, monkeypatch):
    assert levels(integrity.check_holidays(str(tmp_path))) == ["skip"]
    make_holidays(tmp_path)
    monkeypatch.setattr(integrity, "HOLIDAYS_N", 6)
    monkeypatch.setattr(integrity, "HOLIDAYS_N_QUERIES", 3)
    assert not integrity.check_holidays(str(tmp_path)).failed

    (tmp_path / "holidays" / "jpg" / "100101.jpg").unlink()           # 數量不符、出現只有 1 張的組
    assert integrity.check_holidays(str(tmp_path)).failed


def test_check_revisited(tmp_path, monkeypatch):
    ds = "roxford5k"
    assert levels(integrity.check_revisited(str(tmp_path), ds)) == ["skip"]
    d = tmp_path / "datasets" / ds
    names = [f"img{i}" for i in range(4)]
    for i, n in enumerate(names):
        write_jpeg(d / "jpg" / f"{n}.jpg", i)
    gnd = {"imlist": names[:3], "qimlist": names[3:], "gnd": [{"bbx": [0, 0, 10, 10], "easy": [0], "hard": [], "junk": []}]}
    with open(d / f"gnd_{ds}.pkl", "wb") as f:
        pickle.dump(gnd, f)
    monkeypatch.setitem(integrity.REVISITED_N, ds, (3, 1))
    assert not integrity.check_revisited(str(tmp_path), ds).failed

    (d / "jpg" / "img1.jpg").write_bytes(b"")
    assert integrity.check_revisited(str(tmp_path), ds).failed


def test_check_coco_annotations(tmp_path, monkeypatch):
    assert integrity.check_coco_annotations(str(tmp_path)).failed               # 還沒下載算失敗（主流程必需）
    ann = tmp_path / "coco" / "annotations"
    ann.mkdir(parents=True)
    cats = [{"id": i + 1, "name": f"c{i}"} for i in range(80)]
    for split in ("train2017", "val2017"):
        inst = {"images": [{"id": 1, "file_name": "1.jpg", "width": 10, "height": 10, "coco_url": "u"}], "categories": cats,
                "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0,
                                 "segmentation": [[0, 0, 1, 1]]}]}
        (ann / f"instances_{split}.json").write_text(json.dumps(inst))
    monkeypatch.setattr(integrity, "COCO_N_IMAGES", {"train2017": 1, "val2017": 1})
    assert not integrity.check_coco_annotations(str(tmp_path)).failed
    monkeypatch.setattr(integrity, "COCO_N_IMAGES", {"train2017": 118287, "val2017": 5000})
    assert integrity.check_coco_annotations(str(tmp_path)).failed


# ---------------- cache/sets ----------------

def make_sets(root):
    cache = root / "cache"
    g_paths = [write_jpeg(root / "img" / f"g{i}.jpg", i) for i in range(4)]
    gallery = [{"id": f"g{i}", "path": str(p)} for i, p in enumerate(g_paths)]
    save_set(str(cache), "coco40k", gallery, {"kind": "gallery"})
    q_path = write_jpeg(root / "img" / "q0.jpg", 99)
    sem = [{"id": "q0", "path": str(q_path)}, {"id": "q1", "path": str(q_path), "box": [1, 1, 20, 20]}]
    d = save_set(str(cache), "coco_semantic", sem, {"kind": "queries", "gallery": "coco40k", "task": "semantic"})
    CategoryQrels(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1], [1, 1], [0, 0]]), 2).save(f"{d}/qrels.npz")
    sim = [{"id": "e0", "path": str(q_path), "source_idx": 2}]
    d = save_set(str(cache), "coco_similar", sim, {"kind": "queries", "gallery": "coco40k", "task": "similar"})
    SparseQrels([{2: 1}], 4).save(f"{d}/qrels.txt", ["e0"], [g["id"] for g in gallery])
    cfg = {"paths": {"cache_root": str(cache)},
           "data": {"coco": {"gallery_size": 4, "n_semantic_queries": 2, "n_similar_queries": 1}}}
    return cfg, cache


def test_check_sets_ok(tmp_path):
    cfg, _ = make_sets(tmp_path)
    secs = integrity.check_sets(cfg)
    assert [s.name for s in secs] == ["set coco40k", "set coco_semantic", "set coco_similar"]
    assert not any(s.failed for s in secs), [s.items for s in secs]


def test_check_sets_empty(tmp_path):
    secs = integrity.check_sets({"paths": {"cache_root": str(tmp_path)}, "data": {}})
    assert levels(secs[0]) == ["skip"]


@pytest.mark.parametrize("damage", ["size", "image", "qrels_shape", "qrels_unknown_doc", "source_idx", "bad_box"])
def test_check_sets_detects(tmp_path, damage):
    cfg, cache = make_sets(tmp_path)
    if damage == "size":
        cfg["data"]["coco"]["gallery_size"] = 40000
    elif damage == "image":
        (tmp_path / "img" / "g1.jpg").unlink()
    elif damage == "qrels_shape":
        CategoryQrels(np.array([[1, 0]]), np.array([[1, 0]]), 2).save(f"{set_dir(str(cache), 'coco_semantic')}/qrels.npz")
    elif damage == "qrels_unknown_doc":
        with open(f"{set_dir(str(cache), 'coco_similar')}/qrels.txt", "a", encoding="utf-8") as f:
            f.write("e0 0 nonexistent 1\n")
    elif damage == "source_idx":
        save_set(str(cache), "coco_similar", [{"id": "e0", "path": str(tmp_path / "img" / "q0.jpg"), "source_idx": 9}],
                 {"kind": "queries", "gallery": "coco40k"})
    elif damage == "bad_box":
        save_set(str(cache), "coco_semantic", [{"id": "q0", "path": str(tmp_path / "img" / "q0.jpg")},
                                                {"id": "q1", "path": str(tmp_path / "img" / "q0.jpg"), "box": [20, 1, 5, 20]}],
                 {"kind": "queries", "gallery": "coco40k"})
    assert any(s.failed for s in integrity.check_sets(cfg))


# ---------------- 真實資料（存在時才檢查） ----------------

CFG = load_config()
DATA, CACHE = CFG["paths"]["data_root"], CFG["paths"]["cache_root"]


def assert_ok(secs):
    failures = [f"[{s.name}] {msg}" for s in secs for level, msg in s.items if level == "fail"]
    assert not failures, "\n".join(failures)


def test_real_partials():
    assert_ok([integrity.check_partials(DATA)])


@pytest.mark.skipif(not (integrity.Path(DATA) / "coco" / "annotations").is_dir(), reason="COCO 標註尚未下載")
def test_real_coco_annotations():
    assert_ok([integrity.check_coco_annotations(DATA)])


@pytest.mark.skipif(not (integrity.Path(DATA) / "holidays" / "jpg").is_dir(), reason="Holidays 尚未下載")
def test_real_holidays():
    assert_ok([integrity.check_holidays(DATA)])


@pytest.mark.parametrize("ds", sorted(integrity.REVISITED_N))
def test_real_revisited(ds):
    sec = integrity.check_revisited(DATA, ds)
    if levels(sec) == ["skip"]:
        pytest.skip(f"{ds} 尚未下載")
    assert_ok([sec])


@pytest.mark.skipif(not (integrity.Path(CACHE) / "sets").is_dir(), reason="cache/sets 尚未產生")
def test_real_sets():
    assert_ok(integrity.check_sets(CFG))
