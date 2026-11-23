from collections import Counter

import numpy as np
import pytest
from PIL import Image

from src.config import load_config
from src.edits import EDIT_TYPES, apply_edit, assign_edits

PARAMS = load_config()["edits"]


def _textured(w=100, h=100):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    arr[:, : w // 2, 0] = 220  # strong red left half so the color edit is visible
    return Image.fromarray(arr, "RGB")


@pytest.mark.parametrize("edit", EDIT_TYPES)
def test_each_edit_returns_rgb_and_changes_pixels(edit):
    src = _textured()
    out = apply_edit(src, edit, PARAMS)
    assert out.mode == "RGB"
    if out.size == src.size:
        assert not np.array_equal(np.asarray(out), np.asarray(src))


def test_edit_does_not_mutate_input():
    src = _textured()
    before = np.asarray(src).copy()
    for e in EDIT_TYPES:
        apply_edit(src, e, PARAMS)
    assert np.array_equal(np.asarray(src), before)


def test_crop_size():
    assert apply_edit(_textured(), "crop", PARAMS).size == (70, 70)


def test_unknown_edit_raises():
    with pytest.raises(ValueError):
        apply_edit(_textured(), "flip", PARAMS)


def test_assign_edits_balanced():
    counts = Counter(assign_edits(10))
    assert set(counts) == set(EDIT_TYPES) and set(counts.values()) == {2}
