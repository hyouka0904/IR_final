"""bbx 裁切與使用者框選裁切；線上與離線共用同一份。"""
import math

from PIL import Image


def _clamp_box(image: Image.Image, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    w, h = image.size
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # 向外取整，避免裁掉 bbx 邊緣的像素
    box = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2)))
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        raise ValueError(f"Empty crop box {box} for image of size {image.size}")
    return box


def crop_bbx(image: Image.Image, bbx: tuple[float, float, float, float]) -> Image.Image:
    """依官方協定用 gnd['gnd'][i]['bbx'] 裁切 query。"""
    return image.crop(_clamp_box(image, *bbx))


def crop_user_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """使用者在 demo 上框選的區域，box 為兩個角點 (x1, y1, x2, y2)，順序不拘。"""
    return image.crop(_clamp_box(image, *box))
