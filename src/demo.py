"""Demo logic without UI code: user gallery, search/add handlers, hard-queries panel.
app/app.py wires these into Gradio; every search goes through SearchEngine.search."""
import csv
import json
import logging
import math
import os
import threading
from pathlib import Path

from PIL import Image

from src.config import ROOT, resolve_path
from src.data import read_manifest
from src.pipeline import SearchEngine
from src.preprocess import preprocess
from src.schema import SearchRequest

log = logging.getLogger(__name__)


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


class UserGallery:
    """Photos added from the demo: <root>/images/<id>.jpg listed in <root>/gallery.json."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = self.root / "gallery.json"

    def records(self) -> list[dict]:
        if not self.manifest.exists():
            return []
        return json.loads(self.manifest.read_text(encoding="utf-8"))

    def path(self, record: dict) -> Path:
        return self.root / record["file_name"]

    def write_image(self, img: Image.Image, image_id: int) -> tuple[dict, Path]:
        """Write images/<id>.jpg (not yet listed in the manifest)."""
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        rec = {"image_id": int(image_id), "file_name": f"images/{int(image_id)}.jpg"}
        path = self.path(rec)
        img.convert("RGB").save(path, "JPEG", quality=95)
        return rec, path

    def save(self, img: Image.Image, image_id: int) -> Path:
        rec, path = self.write_image(img, image_id)
        self.append(rec)
        return path

    def append(self, rec: dict) -> None:
        records = self.records() + [rec]
        tmp = self.manifest.with_suffix(".json.tmp")
        tmp.write_bytes(json.dumps(records, indent=1).encode("utf-8"))
        os.replace(tmp, self.manifest)


class DemoState:
    def __init__(self, engine: SearchEngine, cfg: dict, coco_files: dict[int, Path],
                 user: UserGallery):
        self.engine, self.cfg, self.coco_files, self.user = engine, cfg, coco_files, user
        self._user_files: dict[int, Path] = {r["image_id"]: user.path(r) for r in user.records()}
        # Gradio runs Search and Add in parallel threads; both touch the same in-memory indexes
        self.lock = threading.Lock()
        self.skipped: list[int] = []  # user photos that could not be re-added at startup

    @classmethod
    def create(cls, cfg: dict, engine: SearchEngine | None = None) -> "DemoState":
        """Load the cached gallery, then re-add every user photo under its saved id."""
        engine = engine or SearchEngine.from_cache(cfg)
        root = resolve_path(cfg, "data")
        manifest = root / "gallery.json"
        coco_files = ({r["image_id"]: root / "train2017" / r["file_name"]
                       for r in read_manifest(manifest)} if manifest.exists() else {})
        user = UserGallery(_abs(cfg["user"]["dir"]))
        skipped = []
        for rec in user.records():
            try:
                with Image.open(user.path(rec)) as img:
                    engine.add_image(img, rec["image_id"])
            except (OSError, ValueError) as e:  # missing or unreadable file
                log.warning("skipping user photo %s (%s): %s", rec["image_id"],
                            rec["file_name"], e)
                skipped.append(rec["image_id"])
        state = cls(engine, cfg, coco_files, user)
        state.skipped = skipped
        for i in skipped:
            state._user_files.pop(i, None)
        return state

    def path_of(self, image_id: int) -> Path | None:
        """The image file for an id, or None if the demo doesn't know one."""
        image_id = int(image_id)
        return self._user_files.get(image_id) or self.coco_files.get(image_id)

    def _next_id(self) -> int:
        """Above every user id in the engine and the manifest, so a skipped photo's id is
        never handed out again."""
        first = int(self.cfg["user"]["first_id"])
        used = [i for i in self.engine.all_ids() if i >= first]
        used += [r["image_id"] for r in self.user.records()]
        return max(used) + 1 if used else first

    def add(self, img: Image.Image) -> int:
        """Write the file, add it to the engine, then list it in the manifest: a failure at
        any step leaves the engine and the manifest as they were."""
        # save exactly what the engine embeds (EXIF-upright, RGB on white), so a restart
        # re-adds the same pixels
        img = preprocess(img, self.cfg["preprocess"]["max_side"])
        with self.lock:
            image_id = self._next_id()
            rec, path = self.user.write_image(img, image_id)
            try:
                self.engine.add_image(img, image_id)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            self._user_files[image_id] = path
            self.user.append(rec)
        return image_id


def search_handler(state: DemoState, image, task: str, rep: str,
                   k: int) -> tuple[list[tuple[str, str]], str, str]:
    """(gallery items (path, caption), query tags, timings or a ⚠ message)."""
    if image is None:
        return [], "", "Upload a photo first."
    try:
        with state.lock:
            resp = state.engine.search(SearchRequest(image, task, int(k), rep))
    except ValueError as e:
        return [], "", f"⚠ {e}"
    except Exception as e:  # keep the UI alive; show what went wrong
        return [], "", f"⚠ search failed ({type(e).__name__}): {e}"
    items, missing = [], 0
    for r in resp.results:
        path = state.path_of(r.image_id)
        if path is None:
            missing += 1
            continue
        caption = f"#{r.rank} · {r.score:.3f}"
        if r.shared_tags:
            caption += " · " + ", ".join(r.shared_tags)
        items.append((str(path), caption))
    t = resp.timings_ms
    timings = (f"{resp.representation}: embed {t['embed']:.0f} ms · search {t['search']:.1f} ms"
               f" · rerank {t['rerank']:.1f} ms")
    if missing:
        timings += (f" · ⚠ {missing} result{'s' if missing > 1 else ''} without an image file"
                    f" (is data/coco/gallery.json missing?)")
    return items, ", ".join(resp.query_tags), timings


def add_handler(state: DemoState, image) -> str:
    if image is None:
        return "Upload a photo first."
    try:
        return f"Added as #{state.add(image)}"
    except Exception as e:  # nothing was added; say why
        return f"⚠ could not add the photo ({type(e).__name__}): {e}"


def reps_for_task(cfg: dict, available: list[str], task: str) -> list[str]:
    return [r for r in cfg["tasks"][task]["allowed"] if r in available]


def hard_queries(cfg: dict, rep: str) -> list[list]:
    """[qid, AP] for the lowest-AP semantic queries of `rep` (empty if not evaluated yet)."""
    path = resolve_path(cfg, "results") / "per_query" / f"semantic_{rep}.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(int(r["qid"]), float(r["ap"])) for r in csv.DictReader(f)]
    rows = sorted((r for r in rows if not math.isnan(r[1])), key=lambda r: (r[1], r[0]))
    return [[qid, round(ap, 4)] for qid, ap in rows[:cfg["demo"]["hard_queries"]]]


def hard_queries_label(task: str, rep: str) -> str:
    label = f"Hard semantic queries for {rep} (lowest AP); click a row to load it"
    if task != "semantic":
        label += ". AP comes from the semantic evaluation, not the similar task"
    return label


def query_image_path(cfg: dict, qid: int) -> Path:
    root = resolve_path(cfg, "data")
    files = {r["image_id"]: r["file_name"] for r in read_manifest(root / "queries_semantic.json")}
    return root / "val2017" / files[int(qid)]
