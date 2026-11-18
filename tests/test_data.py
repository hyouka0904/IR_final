"""manifest / qrels 的存讀，以及 edit 產生器。"""
import numpy as np
import pytest
from PIL import Image

from src.data import edits
from src.data.sets import CategoryQrels, SparseQrels, list_sets, load_qrels, load_set, save_set, set_dir
from src.features import load_query_image

ECFG = {"types": ["crop", "rotation", "color", "jpeg", "blur"], "crop": {"area": [0.3, 0.7]}, "rotation": {"degrees": [10, 45]},
        "color": {"brightness": [0.6, 1.4], "contrast": [0.6, 1.4], "saturation": [0.3, 1.7], "hue": [-0.1, 0.1]},
        "jpeg": {"quality": [5, 20]}, "blur": {"radius": [2.0, 5.0]}}


def test_set_roundtrip(tmp_path):
    cache = str(tmp_path)
    save_set(cache, "g", [{"id": "a", "path": "x/a.jpg"}, {"id": "b", "path": "x/b.jpg"}], {"kind": "gallery"})
    rows = [{"id": "q1", "path": "x/q.jpg", "box": [1, 2, 30, 40], "exclude_idx": 1}, {"id": "q2", "path": "x/q.jpg"}]
    save_set(cache, "q", rows, {"kind": "queries", "gallery": "g"})
    q = load_set(cache, "q")
    assert q.box(0) == [1, 2, 30, 40] and q.box(1) is None
    assert q.exclude(0) == 1 and q.exclude(1) is None
    assert q.meta["gallery"] == "g" and len(q) == 2
    assert list_sets(cache, "gallery") == ["g"] and list_sets(cache, "queries") == ["q"]

    g = load_set(cache, "g")
    SparseQrels([{1: 1}, {0: 2}], 2).save(f"{set_dir(cache, 'q')}/qrels.txt", q.ids, g.ids)
    qr = load_qrels(cache, q, g, {"eval": {"semantic_max_grade": 2}})
    assert qr.grades(0).tolist() == [0, 1] and qr.grades(1).tolist() == [2, 0]


def test_category_qrels(tmp_path):
    q = np.array([[1, 1, 0], [0, 0, 1]])
    g = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])
    qr = CategoryQrels(q, g, max_grade=1)
    assert qr.grades(0).tolist() == [1, 1, 0]
    qr.save(str(tmp_path / "q.npz"))
    assert CategoryQrels.load(str(tmp_path / "q.npz"), max_grade=2).grades(0).tolist() == [2, 1, 0]


@pytest.mark.parametrize("edit", ECFG["types"])
def test_edits(edit, tmp_path):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    p = edits.sample_params(edit, ECFG, rng)
    out = edits.apply_edit(image, edit, p)
    assert out.mode == "RGB"
    if edit == "crop":
        assert out.width * out.height < image.width * image.height
    else:
        assert out.size == image.size
    path = edits.save_edited(image, edit, p, str(tmp_path / "e"))
    assert path.endswith(".jpg" if edit == "jpeg" else ".png")
    with Image.open(path) as saved:
        assert np.array_equal(np.array(saved.convert("RGB")), np.array(out))   # 存檔結果與 apply_edit 一致


def test_assign_edits_balanced():
    e = edits.assign_edits(1000, ECFG["types"], np.random.default_rng(0))
    assert {t: e.count(t) for t in ECFG["types"]} == {t: 200 for t in ECFG["types"]}


def test_query_box_crop(tmp_path):
    Image.new("RGB", (100, 80)).save(tmp_path / "a.png")
    assert load_query_image(str(tmp_path / "a.png"), [10.2, 5, 50, 45.5]).size == (40, 41)
