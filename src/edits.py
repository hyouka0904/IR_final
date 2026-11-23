"""Synthetic edits that turn a gallery image into a similar-image query.
The ground truth for an edited query is its original image."""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

EDIT_TYPES = ("crop", "rotation", "color", "jpeg", "blur")


def _crop(img: Image.Image, frac: float) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, round(w * frac)), max(1, round(h * frac))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _color(img: Image.Image, hue_shift: float, saturation: float) -> Image.Image:
    hsv = np.asarray(img.convert("HSV")).copy()
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(hue_shift * 256)) % 256
    shifted = Image.fromarray(hsv, "HSV").convert("RGB")
    return ImageEnhance.Color(shifted).enhance(saturation)


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_edit(img: Image.Image, edit: str, params: dict) -> Image.Image:
    img = img.convert("RGB")
    if edit == "crop":
        return _crop(img, params["crop_frac"])
    if edit == "rotation":
        return img.rotate(params["rotate_deg"], resample=Image.Resampling.BICUBIC, expand=False)
    if edit == "color":
        return _color(img, params["hue_shift"], params["saturation"])
    if edit == "jpeg":
        return _jpeg(img, params["jpeg_quality"])
    if edit == "blur":
        return img.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
    raise ValueError(f"edit must be one of {EDIT_TYPES}, got {edit!r}")


def assign_edits(n: int) -> list[str]:
    """Round-robin edit type per query, so each type gets an equal share."""
    return [EDIT_TYPES[i % len(EDIT_TYPES)] for i in range(n)]


def apply_edit_chain(img: Image.Image, edits: list[str], params: dict) -> Image.Image:
    """Apply several edits in order (e.g. the strong combined edit of a synthetic photo pair)."""
    img = img.convert("RGB")
    for edit in edits:
        img = apply_edit(img, edit, params)
    return img
