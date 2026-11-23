import io

import numpy as np
import pytest
from PIL import Image

from src.index import FlatIndex
from src.preprocess import preprocess


def _jpeg_with_orientation(w, h, orientation):
    img = Image.new("RGB", (w, h), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    return Image.open(buf)


def test_preprocess_exif_rotation():
    out = preprocess(_jpeg_with_orientation(40, 20, 6), max_side=512)
    assert out.size == (20, 40)


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "CMYK", "LA"])
def test_preprocess_modes_to_rgb(mode):
    assert preprocess(Image.new(mode, (8, 8)), max_side=512).mode == "RGB"


def test_preprocess_transparent_composited_on_white():
    out = preprocess(Image.new("RGBA", (4, 4), (0, 0, 0, 0)), max_side=512)
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_preprocess_downscale_only():
    assert preprocess(Image.new("RGB", (1024, 256)), max_side=512).size == (512, 128)
    assert preprocess(Image.new("RGB", (100, 50)), max_side=512).size == (100, 50)


def test_crop_clamped():
    out = preprocess(Image.new("RGB", (100, 80)), max_side=512, crop=(-10, 10, 500, 70))
    assert out.size == (100, 60)


def test_crop_zero_area_raises():
    with pytest.raises(ValueError):
        preprocess(Image.new("RGB", (100, 80)), max_side=512, crop=(50, 10, 50, 70))
    with pytest.raises(ValueError):
        preprocess(Image.new("RGB", (100, 80)), max_side=512, crop=(200, 10, 300, 70))


def _unit(rows):
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_ip_index_returns_self_first():
    vecs = _unit([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    idx = FlatIndex.build(vecs, np.array([10, 20, 30]), metric="ip")
    scores, ids = idx.search(vecs[1:2], k=2)
    assert ids[0, 0] == 20 and scores[0, 0] == pytest.approx(1.0)
    assert ids[0, 1] == 30


def test_hamming_index_scores_negative_distance():
    codes = np.array([[0b00000000] * 8, [0b00000011] * 8], dtype=np.uint8)
    idx = FlatIndex.build(codes, np.array([1, 2]), metric="hamming")
    scores, ids = idx.search(codes[:1], k=2)
    assert ids[0].tolist() == [1, 2]
    assert scores[0].tolist() == [0, -16]


def test_k_larger_than_gallery_no_minus_one():
    vecs = _unit([[1, 0], [0, 1]])
    idx = FlatIndex.build(vecs, np.array([5, 6]), metric="ip")
    scores, ids = idx.search(vecs[:1], k=10)
    assert ids.shape == (1, 2) and -1 not in ids


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        FlatIndex.build(_unit([[1, 0]]), np.array([1]), metric="cosine")


@pytest.mark.parametrize("metric", ["ip", "hamming"])
def test_save_load_roundtrip(tmp_path, metric):
    if metric == "ip":
        vecs = _unit([[1, 0], [0, 1], [1, 1]])
    else:
        vecs = np.array([[1] * 8, [2] * 8, [255] * 8], dtype=np.uint8)
    idx = FlatIndex.build(vecs, np.array([7, 8, 9]), metric=metric)
    idx.save(tmp_path / "flat")
    loaded = FlatIndex.load(tmp_path / "flat")
    assert loaded.ntotal == 3 and loaded.metric == metric
    a = idx.search(vecs[2:3], k=3)
    b = loaded.search(vecs[2:3], k=3)
    np.testing.assert_array_equal(a[1], b[1])


def test_flat_index_add():
    vecs = _unit([[1, 0], [0, 1]])
    idx = FlatIndex.build(vecs, np.array([5, 6]), metric="ip")
    idx.add(_unit([[1, 1]]), np.array([1_000_000_000]))
    assert idx.ntotal == 3 and idx.ids.tolist() == [5, 6, 1_000_000_000]
    _, ids = idx.search(_unit([[1, 1]]), k=1)
    assert ids[0, 0] == 1_000_000_000
    codes = np.array([[0] * 8, [255] * 8], dtype=np.uint8)
    hidx = FlatIndex.build(codes, np.array([1, 2]), metric="hamming")
    hidx.add(np.array([[15] * 8], dtype=np.uint8), np.array([3]))
    assert hidx.ntotal == 3 and hidx.search(np.array([[15] * 8], dtype=np.uint8), 1)[1][0, 0] == 3
    with pytest.raises(ValueError):
        idx.add(_unit([[1, 0]]), np.array([7, 8]))
