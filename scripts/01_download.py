"""Step 1: pick the COCO gallery (train2017) and semantic queries (val2017),
download just those images, and write manifests.

    python scripts/01_download.py [--gallery-size N] [--queries N]
"""
import argparse
import logging
import zipfile

import _bootstrap  # noqa: F401
import requests

from src.config import load_config, resolve_path
from src.data import download_images, load_coco_instances, select_subset, write_manifest

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery-size", type=int, default=cfg["data"]["gallery_size"])
    ap.add_argument("--queries", type=int, default=cfg["data"]["semantic_queries"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = resolve_path(cfg, "data")
    ann_dir = root / "annotations"
    zip_path = root.parent / "annotations_trainval2017.zip"
    if not (ann_dir / "instances_train2017.json").exists():
        if not zip_path.exists():
            raise SystemExit(f"Download {ANN_URL} to {zip_path} first (241 MB).")
        with zipfile.ZipFile(zip_path) as z:
            for name in ("instances_train2017.json", "instances_val2017.json"):
                ann_dir.mkdir(parents=True, exist_ok=True)
                (ann_dir / name).write_bytes(z.read(f"annotations/{name}"))

    session = requests.Session()

    def fetch(url: str) -> bytes:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    seed = cfg["data"]["seed"]
    for split, n, manifest in (("train2017", args.gallery_size, "gallery.json"),
                               ("val2017", args.queries, "queries_semantic.json")):
        records = select_subset(load_coco_instances(ann_dir / f"instances_{split}.json"), n, seed)
        stats = download_images(records, root / split, fetch, workers=args.workers)
        failed = set(stats["failed"])
        kept = [r for r in records if r["image_id"] not in failed]
        write_manifest(root / manifest, kept)
        print(f"{split}: {len(kept)} in {manifest} (downloaded {stats['downloaded']}, "
              f"already present {stats['skipped']}, failed {len(failed)})")


if __name__ == "__main__":
    main()
