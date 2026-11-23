"""冒煙測試用的假資料：小型 COCO / Holidays / phone、對應的 config，以及取代 03 的假特徵。

真實流程需要 GPU、模型權重與 10 GB 以上的資料，CI 跑不動；這裡用幾百張隨機圖片與隨機向量，
讓 02 → 04 → 05 → 06 → check_data 的腳本銜接能在幾十秒內完整跑過一次。
"""
import json
import os

import numpy as np
import yaml
from PIL import Image

from src import features
from src.config import DEFAULT_CONFIG
from src.data.sets import list_sets, load_set

N_CATS, VOCAB = 6, 30
DIM = 32


def _save_random_jpeg(path: str, rng, w: int, h: int) -> None:
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(path)


def make_data(work: str) -> str:
    """在 work 下建立假資料與 config，回傳 config 路徑。"""
    rng = np.random.default_rng(0)
    cats = [{"id": i * 2 + 1, "name": f"cat{i}"} for i in range(N_CATS)]   # 與 COCO 一樣，類別 id 不連續
    for split, n, offset in (("train2017", 300, 0), ("val2017", 60, 10000)):
        d = os.path.join(work, "data", "coco", split)
        os.makedirs(d, exist_ok=True)
        images, anns = [], []
        for i in range(n):
            iid = offset + i
            w, h = int(rng.integers(80, 160)), int(rng.integers(80, 160))
            _save_random_jpeg(os.path.join(d, f"{iid:012d}.jpg"), rng, w, h)
            images.append({"id": iid, "file_name": f"{iid:012d}.jpg", "width": w, "height": h, "coco_url": "x"})
            if i % 17 == 0:
                continue                                                # 沒有標註的圖
            for c in rng.choice(N_CATS, size=int(rng.integers(1, 4)), replace=False):
                bw, bh = w * rng.uniform(0.2, 0.6), h * rng.uniform(0.2, 0.6)
                anns.append({"id": len(anns), "image_id": iid, "category_id": cats[c]["id"], "bbox": [1, 1, bw, bh],
                             "area": bw * bh, "iscrowd": 0, "segmentation": [[1, 1, 2, 2, 3, 1]]})
        ann_dir = os.path.join(work, "data", "coco", "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        with open(os.path.join(ann_dir, f"instances_{split}.json"), "w", encoding="utf-8") as f:
            json.dump({"images": images, "annotations": anns, "categories": cats}, f)

    d = os.path.join(work, "data", "holidays", "jpg")
    os.makedirs(d)
    for g in range(10):
        for j in range(3):
            _save_random_jpeg(os.path.join(d, f"{100000 + g * 100 + j}.jpg"), rng, 80, 60)
    d = os.path.join(work, "data", "phone")
    os.makedirs(d)
    for i in range(5):
        _save_random_jpeg(os.path.join(d, f"IMG_{i}.jpg"), rng, 70, 90)

    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: os.path.join(work, k.replace("_root", "")) for k in ("data_root", "cache_root", "results_root")}
    cfg["paths"]["revisitop"] = os.path.join(work, "revisitop")
    cfg["data"]["coco"].update(gallery_size=200, n_semantic_queries=40, n_similar_queries=25)
    cfg["data"]["phone"]["dir"] = d
    vocab = os.path.join(work, "vocab.txt")
    with open(vocab, "w", encoding="utf-8") as f:
        f.write("\n".join(f"thing {i}" for i in range(VOCAB)))
    cfg["tags"]["vocab"] = vocab
    cfg["index"]["ivfpq"].update(nlist=8, m=8, nbits=4, nprobe=4)        # 200 張圖庫訓練不了預設的 256 個中心
    cfg["retrieval"].update(k=20, depth=50)
    path = os.path.join(work, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def make_features(cfg: dict) -> None:
    """取代 03（不需要 torch）：向量用隨機值，edit query = 來源圖向量 + 雜訊，similar 才有意義；pHash 用真的。"""
    cache = cfg["paths"]["cache_root"]
    rng = np.random.default_rng(0)
    gallery_vecs = {}
    sets = [load_set(cache, n) for n in list_sets(cache)]
    for s in sorted(sets, key=lambda s: s.meta["kind"] != "gallery"):    # 先做圖庫，query 才能參照來源圖
        os.makedirs(os.path.join(cache, "features", s.name), exist_ok=True)
        is_query = s.meta["kind"] == "queries"
        for rep in ("clip", "siglip", "dinov2"):
            x = rng.standard_normal((len(s), DIM)).astype(np.float32)
            if is_query and s.rows[0].get("source_idx") not in (None, ""):
                x = gallery_vecs[(s.meta["gallery"], rep)][[int(v) for v in s.column("source_idx")]] + 0.1 * x
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            if not is_query:
                gallery_vecs[(s.name, rep)] = x
            np.save(features.feature_path(cache, s.name, rep), x)
            if is_query:
                features.save_times(features.time_path(cache, s.name, rep), [1.0] * len(s), "cpu")
        images = (features.load_query_image(p, s.box(i)) for i, p in enumerate(s.paths))
        bits, times = features.extract_phash(images, cfg["models"]["phash"]["hash_size"], True)
        np.save(features.feature_path(cache, s.name, "phash"), bits)
        if is_query:
            features.save_times(features.time_path(cache, s.name, "phash"), times, "cpu")
        top_n = cfg["tags"]["top_n"]
        top = np.stack([rng.permutation(VOCAB)[:top_n] for _ in range(len(s))]).astype(np.int32)
        probs = np.sort(rng.dirichlet(np.ones(VOCAB), len(s))[:, :top_n], axis=1)[:, ::-1].astype(np.float32)
        features.TagSet.save(cache, s.name, top, probs, cfg["tags"])
