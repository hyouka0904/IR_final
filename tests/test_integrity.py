"""scripts/check_data.py checks on a tiny synthetic data tree."""
import json
import pickle

import numpy as np
import pytest
from PIL import Image

from src import integrity
from src.config import load_config
from src.data import HOLIDAYS_ID_BASE, write_manifest
from src.index import build_index


def _img(path, color=(1, 2, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


@pytest.fixture
def tree(tmp_path):
    cfg = load_config()
    for k in ("data", "cache", "holidays", "revisited", "phone"):
        cfg["paths"][k] = str(tmp_path / k)
    cfg["data"].update(gallery_size=3, semantic_queries=2)
    cfg["integrity"].update(holidays={"images": 2, "queries": 1}, revisited={"roxford5k": [2, 1]})
    data = tmp_path / "data"
    gallery = [{"image_id": i, "file_name": f"{i}.jpg", "categories": [1]} for i in (1, 2, 3)]
    for r in gallery:
        _img(data / "train2017" / r["file_name"])
    hol = [{"image_id": HOLIDAYS_ID_BASE + 100000 + j, "file_name": f"10000{j}.jpg",
            "source": "holidays", "group": "1000", "categories": []} for j in (0, 1)]
    for r in hol:
        _img(tmp_path / "holidays" / "jpg" / r["file_name"])
    write_manifest(data / "gallery.json", gallery + hol)
    queries = [{"image_id": 10 + i, "file_name": f"q{i}.jpg", "categories": [1, 2]} for i in (0, 1)]
    for q in queries:
        _img(data / "val2017" / q["file_name"])
    write_manifest(data / "queries_semantic.json", queries)
    write_manifest(data / "queries_box.json",
                   [{"image_id": 10, "file_name": "q0.jpg", "category": 2, "crop": [0, 0, 8, 8]}])
    write_manifest(data / "queries_holidays.json", hol[:1])
    return cfg, tmp_path


def _failed(sections):
    return [f"[{s.name}] {m}" for s in sections for lv, m in s.items if lv == "fail"]


def test_clean_tree_passes(tree):
    cfg, _ = tree
    secs = integrity.check_manifests(cfg) + [integrity.check_partials(cfg)]
    assert _failed(secs) == []


@pytest.mark.parametrize("damage", ["broken_image", "count", "duplicate_id", "query_in_gallery",
                                    "box_category", "lonely_holidays", "part_file"])
def test_damage_is_reported(tree, damage):
    cfg, root = tree
    data = root / "data"
    gallery = json.loads((data / "gallery.json").read_text())
    if damage == "broken_image":
        (data / "train2017" / "2.jpg").write_bytes(b"\xff\xd8 truncated")
    elif damage == "count":
        cfg["data"]["gallery_size"] = 40000
    elif damage == "duplicate_id":
        write_manifest(data / "gallery.json", gallery + gallery[:1])
    elif damage == "query_in_gallery":
        write_manifest(data / "gallery.json", gallery + [{"image_id": 10, "file_name": "1.jpg",
                                                          "categories": [1]}])
    elif damage == "box_category":
        write_manifest(data / "queries_box.json",
                       [{"image_id": 10, "file_name": "q0.jpg", "category": 9, "crop": [0, 0, 8, 8]}])
    elif damage == "lonely_holidays":
        write_manifest(data / "gallery.json", gallery[:-1])
    elif damage == "part_file":
        (root / "holidays" / "jpg1.tar.gz.part").write_bytes(b"x")
    assert _failed(integrity.check_manifests(cfg) + [integrity.check_partials(cfg)])


def test_revisited_and_phone_skip_or_check(tree):
    cfg, root = tree
    assert [lv for s in integrity.check_revisited(cfg) for lv, _ in s.items] == ["skip"]
    assert integrity.check_phone(cfg).items[0][0] == "skip"
    d = root / "revisited" / "roxford5k"
    for n in ("a", "b", "q"):
        _img(d / "jpg" / f"{n}.jpg")
    with open(d / "gnd_roxford5k.pkl", "wb") as f:
        pickle.dump({"imlist": ["a", "b"], "qimlist": ["q"], "gnd": [{}]}, f)
    assert _failed(integrity.check_revisited(cfg)) == []
    (d / "jpg" / "b.jpg").unlink()
    assert _failed(integrity.check_revisited(cfg))


def test_cache_must_cover_the_gallery(tree):
    cfg, root = tree
    ids = np.array(sorted(r["image_id"] for r in json.loads((root / "data/gallery.json").read_text())))
    d = root / "cache" / "clip"
    d.mkdir(parents=True)
    v = np.eye(len(ids), 8, dtype=np.float32)
    np.save(d / "vectors.npy", v)
    np.save(d / "ids.npy", ids)
    build_index("flat", v, ids, "ip", cfg).save(d / "flat")
    secs = integrity.check_cache(cfg)
    assert _failed(secs) == []
    assert any(lv == "skip" for s in secs if s.name == "cache dinov2" for lv, _ in s.items)
    build_index("hnsw", v[:-1], ids[:-1], "ip", cfg).save(d / "hnsw")  # stale index
    assert _failed(integrity.check_cache(cfg))
