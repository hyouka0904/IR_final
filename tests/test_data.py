import json

from src.data import (download_images, load_coco_instances, read_manifest, select_subset,
                      write_manifest)


def _coco_json(tmp_path):
    data = {
        "images": [
            {"id": 3, "file_name": "c.jpg", "coco_url": "http://x/c.jpg"},
            {"id": 1, "file_name": "a.jpg", "coco_url": "http://x/a.jpg"},
            {"id": 2, "file_name": "b.jpg", "coco_url": "http://x/b.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 5}, {"image_id": 1, "category_id": 2},
            {"image_id": 1, "category_id": 5}, {"image_id": 3, "category_id": 7},
        ],
    }
    p = tmp_path / "instances.json"
    p.write_text(json.dumps(data))
    return p


def test_load_coco_instances_filters_unannotated(tmp_path):
    recs = load_coco_instances(_coco_json(tmp_path))
    assert [r["image_id"] for r in recs] == [1, 3]
    assert recs[0] == {"image_id": 1, "file_name": "a.jpg", "url": "http://x/a.jpg",
                       "categories": [2, 5]}


def test_select_subset_deterministic():
    recs = [{"image_id": i} for i in range(100)]
    a = select_subset(recs, 10, seed=27)
    b = select_subset(list(reversed(recs)), 10, seed=27)
    assert a == b and len(a) == 10
    assert select_subset(recs, 10, seed=28) != a
    assert len(select_subset(recs, 500, seed=27)) == 100


def test_download_skips_existing_and_records_failures(tmp_path):
    recs = [{"image_id": 1, "file_name": "a.jpg", "url": "u/a"},
            {"image_id": 2, "file_name": "b.jpg", "url": "u/b"},
            {"image_id": 3, "file_name": "c.jpg", "url": "u/bad"}]
    calls = []

    def fetch(url):
        calls.append(url)
        if url == "u/bad":
            raise OSError("connection reset")
        return b"jpegbytes"

    (tmp_path / "a.jpg").write_bytes(b"already")
    stats = download_images(recs, tmp_path, fetch)
    assert stats == {"downloaded": 1, "skipped": 1, "failed": [3]}
    assert (tmp_path / "b.jpg").read_bytes() == b"jpegbytes"
    assert not (tmp_path / "c.jpg").exists() and not list(tmp_path.glob("*.part"))

    calls.clear()
    stats = download_images(recs[:2], tmp_path, fetch)
    assert calls == [] and stats["skipped"] == 2


def test_manifest_roundtrip(tmp_path):
    recs = [{"image_id": 1, "file_name": "a.jpg", "url": "u", "categories": [1]}]
    write_manifest(tmp_path / "m.json", recs)
    assert read_manifest(tmp_path / "m.json") == recs


def test_load_coco_instances_with_boxes_drops_segmentation(tmp_path):
    data = {"images": [{"id": 1, "file_name": "a.jpg", "coco_url": "u", "width": 100, "height": 80}],
            "annotations": [
                {"image_id": 1, "category_id": 5, "bbox": [1, 2, 30, 40], "area": 900,
                 "iscrowd": 0, "segmentation": [[0, 0, 1, 1]]},
                {"image_id": 1, "category_id": 2, "bbox": [0, 0, 99, 79], "area": 7000,
                 "iscrowd": 1, "segmentation": {"counts": "x"}}]}
    p = tmp_path / "i.json"
    p.write_text(json.dumps(data))
    rec = load_coco_instances(p, with_boxes=True)[0]
    assert rec["categories"] == [2, 5] and (rec["width"], rec["height"]) == (100, 80)
    assert rec["boxes"] == [[5, 1, 2, 30, 40, 900]]  # crowd regions have no usable box


def test_holidays_records(tmp_path):
    from src.data import HOLIDAYS_ID_BASE, holidays_records
    for n in ("100000", "100001", "100100", "100101", "100102"):
        (tmp_path / f"{n}.jpg").write_bytes(b"x")
    gallery, queries = holidays_records(tmp_path)
    assert [r["image_id"] for r in queries] == [HOLIDAYS_ID_BASE + 100000, HOLIDAYS_ID_BASE + 100100]
    assert {r["group"] for r in gallery} == {"1000", "1001"} and len(gallery) == 5
    assert all(r["source"] == "holidays" and r["categories"] == [] for r in gallery)
    assert holidays_records(tmp_path / "missing") == ([], [])


def test_box_queries_pick_one_category_box():
    from src.data import box_queries
    cfg = {"box_queries": {"min_categories": 2, "min_area_frac": 0.05, "pad_frac": 0.1}}
    recs = [{"image_id": 1, "file_name": "a.jpg", "categories": [3, 7], "width": 100, "height": 100,
             "boxes": [[3, 10, 10, 40, 40, 1600], [3, 0, 0, 5, 5, 25], [7, 50, 50, 50, 50, 2500]]},
            {"image_id": 2, "file_name": "b.jpg", "categories": [3], "width": 10, "height": 10,
             "boxes": [[3, 0, 0, 9, 9, 81]]}]
    out = box_queries(recs, cfg, seed=27)
    assert len(out) == 1 and out[0]["image_id"] == 1  # single-category queries are skipped
    q = out[0]
    expected = {3: [6, 6, 54, 54], 7: [45, 45, 100, 100]}  # largest box, padded, clipped
    assert q["crop"] == expected[q["category"]]
    assert box_queries(recs, cfg, seed=27) == out


def test_gallery_path_per_source():
    from src.config import load_config, resolve_path
    from src.data import gallery_path
    cfg = load_config()
    assert gallery_path(cfg, {"file_name": "a.jpg"}) == resolve_path(cfg, "data") / "train2017" / "a.jpg"
    assert gallery_path(cfg, {"file_name": "100000.jpg", "source": "holidays"}) == \
        resolve_path(cfg, "holidays") / "jpg" / "100000.jpg"
