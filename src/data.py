"""COCO subset selection, resumable image download, JSON manifests."""
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)


def load_coco_instances(path: Path) -> list[dict]:
    """Records for images with at least one annotation, sorted by image_id."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cats = defaultdict(set)
    for ann in data["annotations"]:
        cats[ann["image_id"]].add(ann["category_id"])
    records = [
        {"image_id": img["id"], "file_name": img["file_name"], "url": img["coco_url"],
         "categories": sorted(cats[img["id"]])}
        for img in data["images"] if img["id"] in cats
    ]
    return sorted(records, key=lambda r: r["image_id"])


def select_subset(records: list[dict], n: int, seed: int) -> list[dict]:
    ordered = sorted(records, key=lambda r: r["image_id"])
    if n >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, n)


def _download_one(rec: dict, dest: Path, fetch: Callable[[str], bytes]) -> str:
    target = dest / rec["file_name"]
    if target.exists() and target.stat().st_size > 0:
        return "skipped"
    part = target.with_name(target.name + ".part")
    try:
        part.write_bytes(fetch(rec["url"]))
        part.replace(target)
        return "downloaded"
    except Exception as e:  # network errors must not stop the batch
        log.warning("download failed for image %s (%s): %s", rec["image_id"], rec["url"], e)
        part.unlink(missing_ok=True)
        return "failed"


def download_images(records: list[dict], dest: Path, fetch: Callable[[str], bytes],
                    workers: int = 1) -> dict:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(lambda r: _download_one(r, dest, fetch), records))
    return {
        "downloaded": outcomes.count("downloaded"),
        "skipped": outcomes.count("skipped"),
        "failed": sorted(r["image_id"] for r, o in zip(records, outcomes) if o == "failed"),
    }


def write_manifest(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1), encoding="utf-8")


def read_manifest(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
