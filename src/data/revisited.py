"""讀取 revisitop 的 gnd、imlist、bbx，並解析圖片路徑。"""
import os
import pickle

from PIL import Image

DATASETS = ("roxford5k", "rparis6k")


def dataset_dir(dataset: str, data_root: str) -> str:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return os.path.join(data_root, "datasets", dataset)


def load_gnd(dataset: str, data_root: str) -> dict:
    """回傳 dict，含 imlist（圖庫）、qimlist（query）、gnd（每個 query 的 bbx / easy / hard / junk）。"""
    path = os.path.join(dataset_dir(dataset, data_root), f"gnd_{dataset}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def db_image_paths(gnd: dict, dataset: str, data_root: str) -> list[str]:
    # 圖庫只包含 imlist；jpg/ 裡的 70 張 query 圖不在其中
    jpg = os.path.join(dataset_dir(dataset, data_root), "jpg")
    return [os.path.join(jpg, name + ".jpg") for name in gnd["imlist"]]


def query_image_paths(gnd: dict, dataset: str, data_root: str) -> list[str]:
    jpg = os.path.join(dataset_dir(dataset, data_root), "jpg")
    return [os.path.join(jpg, name + ".jpg") for name in gnd["qimlist"]]


def query_bbx(gnd: dict, qidx: int) -> tuple[float, float, float, float]:
    """官方 bounding box，格式為 (x1, y1, x2, y2)，原圖像素座標。"""
    return tuple(float(v) for v in gnd["gnd"][qidx]["bbx"])


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def gt_label(gnd: dict, qidx: int, db_idx: int) -> str:
    """Medium 協定下的標記：easy / hard 為 positive，junk 為 junk，其餘為 negative。"""
    g = gnd["gnd"][qidx]
    if db_idx in g["easy"] or db_idx in g["hard"]:
        return "positive"
    if db_idx in g["junk"]:
        return "junk"
    return "negative"
