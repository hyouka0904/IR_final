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
