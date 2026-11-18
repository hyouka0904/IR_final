"""MS-COCO 2017：讀取 instances 標註、抽樣圖庫與 query、依 URL 下載圖片。

目錄：
  data/coco/annotations/instances_{split}.json
  data/coco/{split}/{file_name}
"""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class Coco:
    def __init__(self, data_root: str, split: str):
        self.split = split
        self.image_dir = os.path.join(data_root, "coco", split)
        with open(os.path.join(data_root, "coco", "annotations", f"instances_{split}.json"), encoding="utf-8") as f:
            inst = json.load(f)
        self.images = {im["id"]: im for im in inst["images"]}
        # 類別 id 不連續（1–90 中的 80 個），另外建一個 0–79 的欄位編號
        self.categories = sorted(inst["categories"], key=lambda c: c["id"])
        self.cat_col = {c["id"]: i for i, c in enumerate(self.categories)}
        self.cat_names = [c["name"] for c in self.categories]
        self.anns: dict[int, list[dict]] = {}
        for a in inst["annotations"]:
            if not a.get("iscrowd", 0):
                self.anns.setdefault(a["image_id"], []).append(a)

    def annotated_ids(self) -> list[int]:
        """至少有一個（非 crowd）物件的圖；沒有標註的圖無法定義 semantic 分級。"""
        return sorted(self.anns)

    def path(self, image_id: int) -> str:
        return os.path.join(self.image_dir, self.images[image_id]["file_name"])

    def category_matrix(self, image_ids: list[int]) -> np.ndarray:
        """(len(image_ids), 80) 的 0/1 矩陣。"""
        m = np.zeros((len(image_ids), len(self.categories)), dtype=np.uint8)
        for i, iid in enumerate(image_ids):
            for a in self.anns.get(iid, []):
                m[i, self.cat_col[a["category_id"]]] = 1
        return m

    def pick_box(self, image_id: int, rng: np.random.Generator, min_area_frac: float, pad_frac: float):
        """意圖消歧用的 box query：隨機選一個類別，取該類別最大的物件框（向外擴 pad_frac）。

        回傳 (類別欄位, [x1, y1, x2, y2])；沒有夠大的物件時回傳 None。
        """
        im = self.images[image_id]
        w, h = im["width"], im["height"]
        largest: dict[int, dict] = {}
        for a in self.anns.get(image_id, []):
            if a["area"] >= min_area_frac * w * h:
                c = a["category_id"]
                if c not in largest or a["area"] > largest[c]["area"]:
                    largest[c] = a
        if not largest:
            return None
        cat = sorted(largest)[rng.integers(len(largest))]
        x, y, bw, bh = largest[cat]["bbox"]
        px, py = bw * pad_frac, bh * pad_frac
        box = [max(0.0, x - px), max(0.0, y - py), min(float(w), x + bw + px), min(float(h), y + bh + py)]
        return self.cat_col[cat], [round(v, 1) for v in box]


def sample(ids: list[int], n: int, rng: np.random.Generator) -> list[int]:
    if n > len(ids):
        raise ValueError(f"Cannot sample {n} from {len(ids)} images")
    return sorted(int(i) for i in rng.choice(ids, size=n, replace=False))


def download_images(coco: Coco, image_ids: list[int], workers: int = 16) -> int:
    """只下載抽到的圖（避免下載 18 GB 的 train2017.zip）。回傳下載失敗的張數。"""
    os.makedirs(coco.image_dir, exist_ok=True)
    todo = [i for i in image_ids if not os.path.exists(coco.path(i))]

    def fetch(iid: int) -> bool:
        tmp = coco.path(iid) + ".part"
        try:
            urllib.request.urlretrieve(coco.images[iid]["coco_url"], tmp)
            os.replace(tmp, coco.path(iid))
            return True
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            return False

    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(fetch, todo))
    return len(ok) - sum(ok)
