import pytest
from PIL import Image

from src.config import load_config
from src.edits import apply_edit_chain
from src.pairs import Pair, evaluate_pairs, load_pairs, make_synthetic_pairs, summarize_pairs
from tests.fakes import solid, tag_engine

PARAMS = {"crop_frac": 0.6, "rotate_deg": 15, "hue_shift": 0.08, "saturation": 1.5,
          "jpeg_quality": 20}


def test_apply_edit_chain_order():
    img = Image.new("RGB", (100, 50), (10, 120, 200))
    out = apply_edit_chain(img, ["crop", "rotation"], PARAMS)
    assert out.size == (60, 30)
    assert apply_edit_chain(img, [], PARAMS).size == (100, 50)
    with pytest.raises(ValueError):
        apply_edit_chain(img, ["crop", "warp"], PARAMS)


def test_load_pairs_skips_incomplete(tmp_path):
    ok = tmp_path / "mug"
    ok.mkdir()
    solid((1, 2, 3)).save(ok / "gallery.png")
    solid((1, 2, 3)).save(ok / "query.jpg")
    half = tmp_path / "half"
    half.mkdir()
    solid((1, 2, 3)).save(half / "gallery.jpg")
    two = tmp_path / "two"
    two.mkdir()
    for name in ("gallery.jpg", "gallery.png", "query.jpg"):
        solid((1, 2, 3)).save(two / name)
    (tmp_path / "notes.txt").write_text("x")
    pairs = load_pairs(tmp_path)
    assert pairs == [Pair("mug", ok / "gallery.png", ok / "query.jpg")]
    assert load_pairs(tmp_path / "missing") == []


def test_make_synthetic_pairs(tmp_path):
    src = tmp_path / "val"
    src.mkdir()
    Image.new("RGB", (80, 60), (200, 30, 30)).save(src / "a.jpg")
    cfg = {"pairs": {"synthetic_edit": {"edits": ["crop", "rotation", "color", "jpeg"], **PARAMS}}}
    pairs = make_synthetic_pairs([{"image_id": 7, "file_name": "a.jpg"}], src, tmp_path / "p", cfg)
    assert [p.name for p in pairs] == ["synth_7"]
    with Image.open(pairs[0].gallery) as g, Image.open(pairs[0].query) as q:
        assert g.size == (80, 60) and q.size == (48, 36)
    mtime = pairs[0].query.stat().st_mtime_ns
    make_synthetic_pairs([{"image_id": 7, "file_name": "a.jpg"}], src, tmp_path / "p", cfg)
    assert pairs[0].query.stat().st_mtime_ns == mtime  # existing pairs are kept


def test_evaluate_pairs_partner_rank1(tmp_path):
    cfg = load_config()
    pairs = []
    for name, color in (("cyan", (0, 200, 200)), ("magenta", (200, 0, 200))):
        d = tmp_path / name
        d.mkdir()
        solid(color).save(d / "gallery.png")
        solid(color).save(d / "query.png")
        pairs.append(Pair(name, d / "gallery.png", d / "query.png"))
    rows = evaluate_pairs(tag_engine(cfg), pairs, cfg)
    reps = {("semantic", r) for r in ("clip", "dinov2", "tags", "fusion", "tags_nococo",
                                      "fusion_nococo")} | {("similar", "clip"),
                                                           ("similar", "dinov2")}
    assert {(r["task"], r["representation"]) for r in rows} == reps
    assert len(rows) == 2 * len(reps)
    assert all(r["rank"] == 1 for r in rows), [r for r in rows if r["rank"] != 1]


def test_summarize_pairs_counts():
    rows = [{"pair": "a", "task": "similar", "representation": "dinov2", "rank": 1},
            {"pair": "b", "task": "similar", "representation": "dinov2", "rank": 7},
            {"pair": "c", "task": "similar", "representation": "dinov2", "rank": None},
            {"pair": "a", "task": "semantic", "representation": "fusion", "rank": 12}]
    assert summarize_pairs(rows) == [
        {"task": "similar", "representation": "dinov2", "n": 3, "at1": 1, "top10": 2},
        {"task": "semantic", "representation": "fusion", "n": 1, "at1": 0, "top10": 0},
    ]
