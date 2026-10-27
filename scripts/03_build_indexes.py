"""為每份圖庫 embedding 建立 Flat / HNSW / IVF-PQ 索引並存檔。"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import load_config
from src.data.revisited import DATASETS
from src.index.faiss_index import INDEX_KINDS, build_index, save_index
from src.models.encoders import MODEL_NAMES
from src.pipeline import SCALES, embedding_path, index_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cache_root = cfg["paths"]["cache_root"]
    os.makedirs(os.path.dirname(index_path(cache_root, args.dataset, "x", "ss", "flat")), exist_ok=True)
    for model in MODEL_NAMES:
        for scale in SCALES:
            path = embedding_path(cache_root, args.dataset, model, scale, "db")
            if not os.path.exists(path):
                print(f"!! skip {model}_{scale}: {path} not found")
                continue
            db = np.load(path)
            for kind in INDEX_KINDS:
                t = time.perf_counter()
                index = build_index(kind, db, cfg)
                out = index_path(cache_root, args.dataset, model, scale, kind)
                save_index(index, out)
                print(f">> {model}_{scale}_{kind}: {index.ntotal} vectors, "
                      f"{os.path.getsize(out) / 2**20:.1f} MiB, {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
