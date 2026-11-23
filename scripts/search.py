"""Search the gallery with one photo from the command line.

    python scripts/search.py PHOTO --task semantic|similar [--k 10] [--representation R]
                             [--crop LEFT TOP RIGHT BOTTOM] [--index flat|cluster|ivfpq|hnsw]
                             [--rerank aqe mmr ...]
"""
import argparse

import _bootstrap  # noqa: F401
from PIL import Image

from src.config import load_config, resolve_path
from src.data import gallery_path, read_manifest
from src.pipeline import SearchEngine
from src.schema import INDEX_KINDS, RERANKERS, TASKS, SearchRequest


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo")
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--k", type=int, default=cfg["search"]["k"])
    ap.add_argument("--representation")
    ap.add_argument("--crop", type=int, nargs=4)
    ap.add_argument("--index", choices=INDEX_KINDS, default="flat")
    ap.add_argument("--rerank", nargs="+", choices=[r for r in RERANKERS if r != "rocchio"],
                    default=[], help="rocchio needs marked results; use the demo")
    args = ap.parse_args()

    root = resolve_path(cfg, "data")
    files = {r["image_id"]: gallery_path(cfg, r) for r in read_manifest(root / "gallery.json")}
    engine = SearchEngine.from_cache(cfg)
    with Image.open(args.photo) as img:
        resp = engine.search(SearchRequest(img, args.task, args.k, args.representation,
                                           tuple(args.crop) if args.crop else None,
                                           index=args.index, rerank=tuple(args.rerank)))
    print(f"task={args.task} representation={resp.representation} index={resp.index}"
          + (f" rerank={'+'.join(resp.rerank)}" if resp.rerank else ""))
    if resp.query_tags:
        print("query tags: " + ", ".join(resp.query_tags))
    for r in resp.results:
        tags = f"  [{', '.join(r.shared_tags)}]" if r.shared_tags else ""
        print(f"{r.rank:>3}  {r.image_id:>7}  {r.score:+.4f}  "
              f"{files[r.image_id]}{tags}")
    print("timings (ms): " + ", ".join(f"{k}={v:.1f}" for k, v in resp.timings_ms.items()))


if __name__ == "__main__":
    main()
