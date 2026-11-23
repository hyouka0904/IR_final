"""Step 4: build search indexes from cached gallery vectors.

    python scripts/04_build_index.py --index flat cluster ivfpq hnsw [--models clip siglip dinov2 phash]
Writes cache/<rep>/<kind>/ and cache/<rep>/build.json (build time and size per kind).
pHash codes only get the flat index (binary Hamming); the other kinds need float vectors.
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import faiss
import numpy as np

from src.config import load_config, resolve_path
from src.encoders import PHashEncoder
from src.index import build_index, index_size_bytes
from src.pipeline import DENSE
from src.schema import INDEX_KINDS


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", nargs="+", default=["flat"], choices=INDEX_KINDS)
    ap.add_argument("--models", nargs="+", default=list(DENSE), choices=DENSE)
    args = ap.parse_args()
    faiss.omp_set_num_threads(0)  # building may use every core; evaluation times search on one
    for rep in args.models:
        d = resolve_path(cfg, "cache") / rep
        if not (d / "vectors.npy").exists():
            print(f"skip {rep}: no {d / 'vectors.npy'} (run 02_embed.py --models {rep})")
            continue
        vecs, ids = np.load(d / "vectors.npy"), np.load(d / "ids.npy")
        metric = PHashEncoder.metric if rep == "phash" else "ip"
        info_path = d / "build.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        for kind in args.index:
            if metric == "hamming" and kind != "flat":
                print(f"{rep}: skip {kind} (pHash codes only support flat)")
                continue
            t0 = time.perf_counter()
            idx = build_index(kind, vecs, ids, metric, cfg, seed=cfg["data"]["seed"])
            build_s = time.perf_counter() - t0
            idx.save(d / kind)
            size = index_size_bytes(d / kind)
            info[kind] = {"build_s": round(build_s, 3), "bytes": size, "ntotal": idx.ntotal}
            print(f"{rep}: {kind}/{metric}, {idx.ntotal} vectors, {size / 2**20:.2f} MB, "
                  f"{build_s:.1f}s")
        info_path.write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
