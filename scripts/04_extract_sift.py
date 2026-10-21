"""抽取圖庫所有圖片的 SIFT 特徵，存到 cache/sift/{dataset}/db_sift.h5。

格式見 src/rerank/geometric.py 的 DbSift。
"""
import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.data import revisited
from src.pipeline import sift_path
from src.rerank.geometric import extract_sift


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=revisited.DATASETS)
    p.add_argument("--config", default=None)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    return p.parse_args()


def _extract(path: str, max_side: int, max_keypoints: int):
    return extract_sift(revisited.load_image(path), max_side, max_keypoints)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    g = cfg["geometric"]
    gnd = revisited.load_gnd(args.dataset, cfg["paths"]["data_root"])
    paths = revisited.db_image_paths(gnd, args.dataset, cfg["paths"]["data_root"])
    out = sift_path(cfg["paths"]["cache_root"], args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    offsets = np.zeros(len(paths) + 1, np.int64)
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as f, Pool(args.workers) as pool:
        kp = f.create_dataset("kp", (0, 2), np.float32, maxshape=(None, 2), chunks=(4096, 2))
        desc = f.create_dataset("desc", (0, 128), np.uint8, maxshape=(None, 128), chunks=(4096, 128))
        work = partial(_extract, max_side=g["max_side"], max_keypoints=g["max_keypoints"])
        # imap 保持順序，第 i 筆結果對應 imlist 的第 i 張圖
        for i, (pts, d) in enumerate(tqdm(pool.imap(work, paths, chunksize=8), total=len(paths), desc="sift")):
            a = offsets[i]
            offsets[i + 1] = a + len(pts)
            kp.resize(offsets[i + 1], axis=0)
            desc.resize(offsets[i + 1], axis=0)
            kp[a:], desc[a:] = pts, d
        f["offsets"] = offsets
    os.replace(tmp, out)
    print(f">> {args.dataset}: {len(paths)} images, {offsets[-1]} keypoints, {os.path.getsize(out) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
