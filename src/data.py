"""COCO subset selection, resumable image download, JSON manifests, Holidays and box queries.

Gallery records come from two sources: COCO train2017 (`source` absent) and, when holidays.merge
is on, INRIA Holidays (`source: holidays`, no COCO categories, so never relevant for semantic
queries). gallery_path() resolves either to its file.
"""
import json
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

HOLIDAYS_ID_BASE = 500_000_000  # Holidays 100000.jpg -> 500100000; below user.first_id (1e9)


def _drop_segmentation(obj: dict) -> dict:
    # called for every JSON object as it is parsed, so the polygons never pile up in memory:
    # instances_train2017.json (~450 MB) would otherwise take several GB of RAM
    obj.pop("segmentation", None)
    return obj


def load_coco_instances(path: Path, with_boxes: bool = False) -> list[dict]:
    """Records for images with at least one annotation, sorted by image_id. with_boxes adds the
    image size and the non-crowd boxes [category_id, x, y, w, h, area] (for box queries)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f, object_hook=_drop_segmentation)
    cats = defaultdict(set)
    boxes = defaultdict(list)
    for ann in data["annotations"]:
        cats[ann["image_id"]].add(ann["category_id"])
        if with_boxes and not ann.get("iscrowd", 0) and "bbox" in ann:
            boxes[ann["image_id"]].append([ann["category_id"], *ann["bbox"],
                                           ann.get("area", ann["bbox"][2] * ann["bbox"][3])])
    records = []
    for img in data["images"]:
        if img["id"] not in cats:
            continue
        rec = {"image_id": img["id"], "file_name": img["file_name"], "url": img["coco_url"],
               "categories": sorted(cats[img["id"]])}
        if with_boxes:
            rec.update(width=img["width"], height=img["height"], boxes=boxes[img["id"]])
        records.append(rec)
    return sorted(records, key=lambda r: r["image_id"])


def gallery_path(cfg: dict, rec: dict) -> Path:
    """The image file of a gallery record (COCO train2017 or Holidays)."""
    from src.config import resolve_path
    if rec.get("source") == "holidays":
        return resolve_path(cfg, "holidays") / "jpg" / rec["file_name"]
    return resolve_path(cfg, "data") / "train2017" / rec["file_name"]


def holidays_records(jpg_dir: Path) -> tuple[list[dict], list[dict]]:
    """(gallery records, query records) for INRIA Holidays: 1,491 images in 500 groups named
    GGGGNN.jpg; the first image of a group (NN = 00) is its query. A query's relevant images
    are the other members of its group; the query itself stays in the gallery and is excluded
    from its own ranking at evaluation time."""
    jpg_dir = Path(jpg_dir)
    names = sorted(p.name for p in jpg_dir.glob("*.jpg")) if jpg_dir.is_dir() else []
    gallery = [{"image_id": HOLIDAYS_ID_BASE + int(n[:-4]), "file_name": n, "source": "holidays",
                "group": n[:4], "categories": []} for n in names if n[:-4].isdigit()]
    queries = [r for r in gallery if r["file_name"][4:6] == "00"]
    return gallery, queries


def box_queries(records: list[dict], cfg: dict, seed: int) -> list[dict]:
    """Intent disambiguation ("the cat or the sofa?"): for every semantic query with at least
    box_queries.min_categories categories, pick one category at random and crop to its largest
    object box (padded). Only that category counts as relevant."""
    c = cfg["box_queries"]
    rng = random.Random(seed)
    out = []
    for rec in records:
        if len(rec["categories"]) < c["min_categories"]:
            continue
        w, h = rec["width"], rec["height"]
        largest: dict[int, list] = {}
        for cat, x, y, bw, bh, area in rec["boxes"]:
            if area >= c["min_area_frac"] * w * h and (cat not in largest or area > largest[cat][5]):
                largest[cat] = [cat, x, y, bw, bh, area]
        if not largest:
            continue
        cat = sorted(largest)[rng.randrange(len(largest))]
        _, x, y, bw, bh, _ = largest[cat]
        px, py = bw * c["pad_frac"], bh * c["pad_frac"]
        crop = [max(0, round(x - px)), max(0, round(y - py)),
                min(w, round(x + bw + px)), min(h, round(y + bh + py))]
        out.append({"image_id": rec["image_id"], "file_name": rec["file_name"],
                    "category": cat, "crop": crop})
    return out


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
