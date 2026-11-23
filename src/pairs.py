"""Same-subject photo pairs: the gallery photo joins the gallery, the query photo must find it."""
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.edits import apply_edit_chain
from src.schema import TASKS, SearchRequest

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Pair:
    name: str
    gallery: Path
    query: Path


def load_pairs(root: Path) -> list[Pair]:
    """Every subfolder of root holding exactly one gallery.* and one query.* image."""
    root = Path(root)
    if not root.is_dir():
        return []
    pairs = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        found = {}
        for stem in ("gallery", "query"):
            found[stem] = [f for f in d.iterdir() if f.stem == stem and f.suffix.lower() in IMAGE_EXTS]
        if all(len(v) == 1 for v in found.values()):
            pairs.append(Pair(d.name, found["gallery"][0], found["query"][0]))
        else:
            log.warning("skipping pair folder %s: needs exactly one gallery.* and one query.* "
                        "image, found %s", d, {k: len(v) for k, v in found.items()})
    return pairs


def make_synthetic_pairs(records: list[dict], image_dir: Path, out_root: Path,
                         cfg: dict) -> list[Pair]:
    """synth_<id>/gallery.jpg = the original, query.jpg = the combined strong edit."""
    edit = cfg["pairs"]["synthetic_edit"]
    pairs = []
    for rec in records:
        d = Path(out_root) / f"synth_{rec['image_id']}"
        pair = Pair(d.name, d / "gallery.jpg", d / "query.jpg")
        if not (pair.gallery.exists() and pair.query.exists()):
            d.mkdir(parents=True, exist_ok=True)
            with Image.open(Path(image_dir) / rec["file_name"]) as img:
                img = img.convert("RGB")
                img.save(pair.gallery, "JPEG", quality=95)
                apply_edit_chain(img, edit["edits"], edit).save(pair.query, "JPEG", quality=95)
        pairs.append(pair)
    return pairs


def evaluate_pairs(engine, pairs: list[Pair], cfg: dict) -> list[dict]:
    """Add every gallery photo (in memory only), then rank each partner for every task and
    allowed, loaded representation. rank is None when the partner is beyond search.eval_depth."""
    partner = {}
    for p in pairs:
        with Image.open(p.gallery) as img:
            partner[p.name] = engine.add_image(img)
    depth = cfg["search"]["eval_depth"]
    available = engine.available()
    rows = []
    for task in TASKS:
        for rep in (r for r in cfg["tasks"][task]["allowed"] if r in available):
            for p in pairs:
                with Image.open(p.query) as img:
                    resp = engine.search(SearchRequest(img, task, depth, rep))
                rank = next((r.rank for r in resp.results if r.image_id == partner[p.name]), None)
                rows.append({"pair": p.name, "task": task, "representation": rep, "rank": rank})
    return rows


def summarize_pairs(rows: list[dict]) -> list[dict]:
    """Per (task, representation): pairs, partners at rank 1, partners in the top 10."""
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        s = out.setdefault((r["task"], r["representation"]),
                           {"task": r["task"], "representation": r["representation"],
                            "n": 0, "at1": 0, "top10": 0})
        s["n"] += 1
        s["at1"] += r["rank"] == 1
        s["top10"] += r["rank"] is not None and r["rank"] <= 10
    return list(out.values())
