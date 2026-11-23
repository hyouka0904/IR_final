"""Online query preprocessing: EXIF fix, RGB, optional crop, downscale."""
from PIL import Image, ImageOps


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA", "PA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return img.convert("RGB")


def preprocess(img: Image.Image, max_side: int,
               crop: tuple[int, int, int, int] | None = None) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = _to_rgb(img)
    if crop is not None:
        w, h = img.size
        left, top, right, bottom = crop
        left, right = max(0, min(left, w)), max(0, min(right, w))
        top, bottom = max(0, min(top, h)), max(0, min(bottom, h))
        if right <= left or bottom <= top:
            raise ValueError(f"crop box {crop} has no area inside a {w}x{h} image")
        img = img.crop((left, top, right, bottom))
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return img
