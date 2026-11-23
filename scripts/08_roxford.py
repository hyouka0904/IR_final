"""Step 8 (supplementary): the similar-image representations on ROxford5k / RParis6k.

    python scripts/08_roxford.py [--models clip siglip dinov2 phash]
Needs `python scripts/01_download.py --with-roxford` (images, gnd files, revisitop code).
Official protocol: queries cropped to the gnd box, gallery = gnd imlist, revisitop compute_map
for Easy / Medium / Hard mAP and mP@k. The galleries are small (5-6k), so every query is ranked
exactly against the whole gallery. Features are cached in cache/revisited/<dataset>/.
Writes results/roxford/summary.csv and results/roxford/per_query/<dataset>_<model>.csv.
"""
import argparse
import csv

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from src.config import load_config, resolve_path
from src.encoders import get_encoder
from src.preprocess import preprocess
from src.revisited import PROTOCOLS, evaluate, full_ranking, image_paths, load_gnd, query_box


def embed(encoder, paths, boxes, cfg) -> np.ndarray:
    bs, max_side = cfg["embed"]["batch_size"], cfg["preprocess"]["max_side"]
    out = []
    for start in range(0, len(paths), bs):
        batch = []
        for p, box in zip(paths[start:start + bs], boxes[start:start + bs]):
            with Image.open(p) as im:
                batch.append(preprocess(im, max_side, box))
        out.append(encoder.encode(batch))
        print(f"  {encoder.name}: {min(start + bs, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    return np.concatenate(out)


def main():
    cfg = load_config()
    rcfg = cfg["roxford"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=rcfg["models"],
                    choices=["clip", "siglip", "dinov2", "phash"])
    args = ap.parse_args()
    data, revisitop = resolve_path(cfg, "revisited"), resolve_path(cfg, "revisitop")
    if not (revisitop / "evaluate.py").exists():
        raise SystemExit("revisitop code missing: run python scripts/01_download.py --with-roxford")
    out_dir = resolve_path(cfg, "results") / "roxford"
    (out_dir / "per_query").mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in rcfg["datasets"]:
        if not (data / ds / f"gnd_{ds}.pkl").exists():
            print(f"skip {ds}: not downloaded (01_download.py --with-roxford)")
            continue
        gnd = load_gnd(data, ds)
        db_paths = image_paths(data, ds, gnd["imlist"])
        q_paths = image_paths(data, ds, gnd["qimlist"])
        q_boxes = [query_box(gnd, i) for i in range(len(q_paths))]
        feat_dir = resolve_path(cfg, "cache") / "revisited" / ds
        feat_dir.mkdir(parents=True, exist_ok=True)
        for rep in args.models:
            files = {s: feat_dir / f"{rep}_{s}.npy" for s in ("db", "q")}
            if not all(f.exists() for f in files.values()):
                enc = get_encoder(rep, cfg)
                np.save(files["db"], embed(enc, db_paths, [None] * len(db_paths), cfg))
                np.save(files["q"], embed(enc, q_paths, q_boxes, cfg))
            db, q = np.load(files["db"]), np.load(files["q"])
            ranks = full_ranking(db, q, "hamming" if rep == "phash" else "ip")
            summary, per_query = evaluate(ranks, gnd["gnd"], rcfg["mp_ks"], revisitop)
            rows.append({"dataset": ds, "model": rep, **{k: f"{v:.2f}" for k, v in summary.items()}})
            with open(out_dir / "per_query" / f"{ds}_{rep}.csv", "w", newline="",
                      encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["query", *(f"{p}_AP" for p in PROTOCOLS)])
                for i, name in enumerate(gnd["qimlist"]):
                    w.writerow([name, *(f"{per_query[p][i]:.2f}" for p in PROTOCOLS)])
            print(f"{ds} {rep}: " + ", ".join(f"{p} mAP {summary[f'{p}_mAP']:.2f}" for p in PROTOCOLS))
    if rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
