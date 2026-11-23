import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.encoders import PHashEncoder, get_encoder, resolve_device

CFG = load_config()


def _imgs():
    rng = np.random.default_rng(1)
    return [Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)),
            Image.new("RGB", (50, 50), (10, 200, 30))]


def test_phash_shape_and_dtype():
    enc = PHashEncoder(hash_size=8)
    codes = enc.encode(_imgs())
    assert codes.shape == (2, 8) and codes.dtype == np.uint8
    assert enc.metric == "hamming" and enc.name == "phash"


def test_phash_identical_images_equal():
    img = _imgs()[0]
    codes = PHashEncoder(8).encode([img, img.copy()])
    assert np.array_equal(codes[0], codes[1])


def test_get_encoder_unknown_raises():
    with pytest.raises(ValueError):
        get_encoder("sift", CFG)


def test_resolve_device_cpu_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


@pytest.mark.models
def test_clip_unit_norm():
    enc = get_encoder("clip", CFG)
    v = enc.encode(_imgs())
    assert enc.metric == "ip" and v.shape == (2, 512) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


@pytest.mark.models
def test_dinov2_unit_norm():
    enc = get_encoder("dinov2", CFG)
    v = enc.encode(_imgs())
    assert enc.metric == "ip" and v.shape == (2, 384) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


@pytest.mark.models
def test_clip_text_unit_norm():
    enc = get_encoder("clip", CFG)
    v = enc.encode_text(["a photo of a cat.", "a photo of a dining table."])
    assert v.shape == (2, 512) and v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
