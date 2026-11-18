"""Similar-image query：對圖庫圖片做 edit（crop / rotation / color / JPEG / blur）。

參數範圍在 configs/default.yaml 的 data.edits。每張 query 只套一種 edit，
這樣 robustness 分析才能逐 edit 類型比較 DINOv2 與 pHash。
JPEG 以外的 edit 都存 PNG，避免額外的壓縮失真混進其他 edit 類型。
"""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def sample_params(edit: str, ecfg: dict, rng: np.random.Generator) -> dict:
    c = ecfg[edit]
    u = lambda lo_hi: float(rng.uniform(*lo_hi))
    if edit == "crop":
        # 面積比例與長寬比（3/4–4/3）都隨機，位置在圖內隨機
        return {"area": u(c["area"]), "aspect": float(np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))),
                "cx": float(rng.uniform()), "cy": float(rng.uniform())}
    if edit == "rotation":
        return {"degrees": u(c["degrees"]) * float(rng.choice([-1, 1]))}
    if edit == "color":
        return {k: u(c[k]) for k in ("brightness", "contrast", "saturation", "hue")}
    if edit == "jpeg":
        return {"quality": int(round(u(c["quality"])))}
    if edit == "blur":
        return {"radius": u(c["radius"])}
    raise ValueError(f"Unknown edit: {edit}")


def apply_edit(image: Image.Image, edit: str, p: dict) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if edit == "crop":
        cw = min(w, round(np.sqrt(p["area"] * w * h * p["aspect"])))
        ch = min(h, round(np.sqrt(p["area"] * w * h / p["aspect"])))
        x, y = round(p["cx"] * (w - cw)), round(p["cy"] * (h - ch))
        return image.crop((x, y, x + cw, y + ch))
    if edit == "rotation":
        # 不擴大畫布：保持原尺寸，四角補黑
        return image.rotate(p["degrees"], resample=Image.BICUBIC, expand=False)
    if edit == "color":
        image = ImageEnhance.Brightness(image).enhance(p["brightness"])
        image = ImageEnhance.Contrast(image).enhance(p["contrast"])
        image = ImageEnhance.Color(image).enhance(p["saturation"])
        return shift_hue(image, p["hue"])
    if edit == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=p["quality"])
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("RGB")
    if edit == "blur":
        return image.filter(ImageFilter.GaussianBlur(p["radius"]))
    raise ValueError(f"Unknown edit: {edit}")


def shift_hue(image: Image.Image, shift: float) -> Image.Image:
    """shift 以色相環的比例表示（-0.5–0.5）。"""
    hsv = np.array(image.convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + round(shift * 256)) % 256
    return Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")


def save_edited(image: Image.Image, edit: str, p: dict, path_stem: str) -> str:
    """回傳實際檔名。JPEG edit 直接以該品質存檔，與 apply_edit 的結果一致。"""
    if edit == "jpeg":
        path = path_stem + ".jpg"
        image.convert("RGB").save(path, format="JPEG", quality=p["quality"])
    else:
        path = path_stem + ".png"
        apply_edit(image, edit, p).save(path)
    return path


def assign_edits(n: int, types: list[str], rng: np.random.Generator) -> list[str]:
    """每種 edit 數量平均（n 不整除時前幾種多一張），順序打亂。"""
    edits = [types[i % len(types)] for i in range(n)]
    rng.shuffle(edits)
    return edits
