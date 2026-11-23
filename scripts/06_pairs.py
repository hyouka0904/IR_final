"""Step 6: photo-pair check. Each pair's gallery photo joins the gallery (in memory only),
its query photo is searched, and the partner's rank is recorded.

    python scripts/06_pairs.py [--synthetic N]
Pairs live in pairs.dir as <name>/gallery.<ext> + <name>/query.<ext>. --synthetic N (default
pairs.synthetic) first makes N pairs from COCO val2017 images that are not semantic queries.
Demo photos in data/user/ are never loaded. Writes results/pairs.csv.
"""
import argparse
import csv

import _bootstrap  # noqa: F401
import requests

from src.config import ROOT, load_config, resolve_path
from src.data import download_images, load_coco_instances, read_manifest, select_subset
from src.pairs import evaluate_pairs, load_pairs, make_synthetic_pairs, summarize_pairs
from src.pipeline import SearchEngine


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", type=int, default=cfg["pairs"]["synthetic"])
    args = ap.parse_args()

    root = resolve_path(cfg, "data")
    pairs_dir = ROOT / cfg["pairs"]["dir"]
    if args.synthetic > 0:
        query_ids = {r["image_id"] for r in read_manifest(root / "queries_semantic.json")}
        pool = [r for r in load_coco_instances(root / "annotations" / "instances_val2017.json")
                if r["image_id"] not in query_ids]
        records = select_subset(pool, args.synthetic, cfg["data"]["seed"])

        def fetch(url: str) -> bytes:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content

        stats = download_images(records, root / "val2017", fetch)
        failed = set(stats["failed"])
        made = make_synthetic_pairs([r for r in records if r["image_id"] not in failed],
                                    root / "val2017", pairs_dir, cfg)
        print(f"synthetic pairs: {len(made)} (download failures: {len(failed)})")

    pairs = load_pairs(pairs_dir)
    if not pairs:
        raise SystemExit(f"no pairs under {pairs_dir}")
    engine = SearchEngine.from_cache(cfg)  # COCO gallery only; data/user is not loaded
    rows = evaluate_pairs(engine, pairs, cfg)

    out = resolve_path(cfg, "results") / "pairs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "task", "representation", "rank"])
        w.writeheader()
        w.writerows({**r, "rank": "" if r["rank"] is None else r["rank"]} for r in rows)
    print(f"{len(pairs)} pairs; partner at rank 1 / in top 10:")
    for s in summarize_pairs(rows):
        print(f"  {s['task']:8s} {s['representation']:13s} {s['at1']:>3}/{s['n']}  "
              f"{s['top10']:>3}/{s['n']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
