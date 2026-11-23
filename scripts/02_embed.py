"""Step 2: embed every gallery image with each representation.

    python scripts/02_embed.py --models clip siglip dinov2 phash
Writes cache/<rep>/vectors.npy and cache/<rep>/ids.npy.
"""
import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np

from src.config import load_config, resolve_path
from src.data import gallery_path, read_manifest
from src.embed import embed_paths
from src.encoders import get_encoder
from src.pipeline import DENSE


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(DENSE), choices=DENSE)
    args = ap.parse_args()

    root = resolve_path(cfg, "data")
    gallery = read_manifest(root / "gallery.json")
    paths = [gallery_path(cfg, r) for r in gallery]  # COCO train2017 and merged Holidays
    ids = np.array([r["image_id"] for r in gallery], dtype=np.int64)
    for rep in args.models:
        t0 = time.perf_counter()
        enc = get_encoder(rep, cfg)
        vecs = embed_paths(enc, paths, cfg["embed"]["batch_size"], cfg["preprocess"]["max_side"],
                           progress=True)
        out = resolve_path(cfg, "cache") / rep
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", vecs)
        np.save(out / "ids.npy", ids)
        print(f"{rep}: {vecs.shape} {vecs.dtype} in {time.perf_counter() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
