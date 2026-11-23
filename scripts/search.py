"""Search the gallery with one photo from the command line.

    python scripts/search.py PHOTO --task semantic|similar [--k 10] [--representation R]
                             [--crop LEFT TOP RIGHT BOTTOM]
"""
import argparse

import _bootstrap  # noqa: F401
from PIL import Image

from src.config import load_config, resolve_path
from src.data import read_manifest
from src.pipeline import SearchEngine
from src.schema import TASKS, SearchRequest


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo")
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--k", type=int, default=cfg["search"]["k"])
    ap.add_argument("--representation")
    ap.add_argument("--crop", type=int, nargs=4)
    args = ap.parse_args()

    root = resolve_path(cfg, "data")
    files = {r["image_id"]: r["file_name"] for r in read_manifest(root / "gallery.json")}
    engine = SearchEngine.from_cache(cfg)
    with Image.open(args.photo) as img:
        resp = engine.search(SearchRequest(img, args.task, args.k, args.representation,
                                           tuple(args.crop) if args.crop else None))
    print(f"task={args.task} representation={resp.representation}")
    if resp.query_tags:
        print("query tags: " + ", ".join(resp.query_tags))
    for r in resp.results:
        tags = f"  [{', '.join(r.shared_tags)}]" if r.shared_tags else ""
        print(f"{r.rank:>3}  {r.image_id:>7}  {r.score:+.4f}  "
              f"{root / 'train2017' / files[r.image_id]}{tags}")
    print("timings (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in resp.timings_ms.items()))


if __name__ == "__main__":
    main()
