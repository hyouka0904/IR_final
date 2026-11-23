"""Step 4: build search indexes from cached gallery vectors.

    python scripts/04_build_index.py --index flat [--models clip dinov2 phash]
Only the exact Flat index exists this week (cluster pruning / IVF-PQ / HNSW: week 3).
"""
import argparse

import _bootstrap  # noqa: F401
import numpy as np

from src.config import load_config, resolve_path
from src.encoders import PHashEncoder
from src.index import FlatIndex
from src.pipeline import DENSE


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", nargs="+", default=["flat"], choices=["flat"])
    ap.add_argument("--models", nargs="+", default=list(DENSE), choices=DENSE)
    args = ap.parse_args()
    for rep in args.models:
        d = resolve_path(cfg, "cache") / rep
        vecs, ids = np.load(d / "vectors.npy"), np.load(d / "ids.npy")
        metric = PHashEncoder.metric if rep == "phash" else "ip"
        idx = FlatIndex.build(vecs, ids, metric)
        idx.save(d / "flat")
        size_mb = (d / "flat" / "index.faiss").stat().st_size / 2**20
        print(f"{rep}: flat/{metric}, {idx.ntotal} vectors, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
