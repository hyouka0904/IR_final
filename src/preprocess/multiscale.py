"""多尺度抽取與合併；線上與離線共用同一份。

模型分兩類：
- 固定解析度（CLIP / SigLIP 的 ViT，position embedding 固定）：輸出永遠是 base_size。
  scale >= 1 時把短邊放大到 base * scale 再中央裁切（看細節）；
  scale < 1 時把長邊縮到 base * scale，再貼到 base 大小的畫布中央（看全景）。
- 彈性解析度（DINOv2）：短邊縮放到 base * scale（取 patch 的整數倍）再中央裁成正方形。
"""
import numpy as np
from PIL import Image


def scale_image(image: Image.Image, base_size: int, scale: float, patch: int | None = None) -> Image.Image:
    w, h = image.size
    target = round(base_size * scale)
    if patch:
        target = max(patch, round(target / patch) * patch)
        return _resize_short_center_crop(image, target, target)
    if scale >= 1:
        return _resize_short_center_crop(image, target, base_size)
    # 縮小：整張圖縮進 target，置中貼到 base_size 畫布
    r = target / max(w, h)
    small = image.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)
    canvas = Image.new("RGB", (base_size, base_size), (124, 116, 104))
    canvas.paste(small, ((base_size - small.width) // 2, (base_size - small.height) // 2))
    return canvas


def _resize_short_center_crop(image: Image.Image, short: int, crop: int) -> Image.Image:
    w, h = image.size
    r = short / min(w, h)
    nw, nh = max(crop, round(w * r)), max(crop, round(h * r))
    image = image.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - crop) // 2, (nh - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def combine(embs: list[np.ndarray]) -> np.ndarray:
    """各尺度的 L2-normalized embedding 相加後再 normalize。embs: [ (N, D), ... ]"""
    x = np.sum(embs, axis=0)
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def extract_multiscale(encoder, images: list[Image.Image], scales: list[float]) -> np.ndarray:
    """回傳 (N, D)。scales == [1.0] 時即為單一尺度。"""
    return combine([encoder.encode(images, scale=s) for s in scales])


def scales_for(scale_mode: str, cfg: dict) -> list[float]:
    """config_id 的 scale 欄位（ss / ms）→ 實際尺度清單。"""
    return [1.0] if scale_mode == "ss" else list(cfg["multiscale"]["scales"])
