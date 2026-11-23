"""Data integrity checks for scripts/check_data.py (and the smoke test / CI).

Each check returns a Section of (level, message) items:
  ok    passed
  skip  an optional dataset that is not downloaded or built yet (Holidays, ROxford, a cache)
  fail  a missing or broken file, a count that disagrees with the config, ids that do not match
Expected dataset sizes come from cfg["integrity"].
"""
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import resolve_path
from src.data import gallery_path, read_manifest
from src.download import image_problems



@dataclass
class Section:
    name: str
    items: list[tuple[str, str]] = field(default_factory=list)

    def add(self, level: str, msg: str) -> None:
        self.items.append((level, msg))

    def expect(self, cond: bool, msg: str) -> bool:
        self.add("ok" if cond else "fail", msg)
        return cond

    @property
    def failed(self) -> bool:
        return any(level == "fail" for level, _ in self.items)


def check_images(sec: Section, label: str, paths: list, full: bool = False) -> None:
    problems = image_problems(paths, full)
    bad = [(Path(p), m) for p, m in zip(paths, problems) if m]
    if not bad:
        sec.add("ok", f"{label}: all {len(paths)} images readable")
        return
    examples = "; ".join(f"{p.name} {m}" for p, m in bad[:5])
    sec.add("fail", f"{label}: {len(bad)} / {len(paths)} images missing or broken, e.g. {examples}"
                    " (run 01_download.py to repair)")


def check_partials(cfg: dict) -> Section:
    sec = Section("interrupted downloads")
    roots = {resolve_path(cfg, k) for k in ("data", "holidays", "revisited")}
    roots.add(resolve_path(cfg, "data").parent)  # the annotations zip lives next to data/coco
    parts = sorted({p for r in roots if r.is_dir() for p in r.rglob("*.part")})
    if parts:
        sec.add("fail", f"{len(parts)} interrupted download(s), e.g. {parts[0].name}: "
                        "run 01_download.py to resume")
    else:
        sec.add("ok", "no interrupted downloads (.part)")
    return sec


def check_annotations(cfg: dict) -> Section:
    sec = Section("COCO annotations")
    counts = cfg["integrity"]["coco_images"]
    n_cats = cfg["integrity"]["coco_categories"]
    ann = resolve_path(cfg, "data") / "annotations"
    for split, n in counts.items():
        path = ann / f"instances_{split}.json"
        if not path.exists():
            sec.add("fail", f"{split}: {path} missing (run 01_download.py)")
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f, object_hook=lambda o: (o.pop("segmentation", None), o)[1])
        except ValueError as e:
            sec.add("fail", f"{split}: unreadable JSON ({e}); run 01_download.py to repair")
            continue
        sec.expect(len(data["images"]) == n, f"{split}: {len(data['images'])} images (expected {n})")
        sec.expect(len(data["categories"]) == n_cats,
                   f"{split}: {len(data['categories'])} categories (expected {n_cats})")
    return sec


def _manifest(sec: Section, path: Path) -> list[dict] | None:
    if not path.exists():
        sec.add("fail", f"{path.name} missing (run 01_download.py)")
        return None
    try:
        return read_manifest(path)
    except ValueError as e:
        sec.add("fail", f"{path.name} unreadable ({e})")
        return None


def check_manifests(cfg: dict, full: bool = False) -> list[Section]:
    root = resolve_path(cfg, "data")
    n_hol, n_hol_q = cfg["integrity"]["holidays"]["images"], cfg["integrity"]["holidays"]["queries"]
    out = []

    sec = Section("gallery")
    out.append(sec)
    gallery = _manifest(sec, root / "gallery.json")
    if gallery is not None:
        ids = [r["image_id"] for r in gallery]
        sec.expect(len(set(ids)) == len(ids), "image ids are unique")
        coco = [r for r in gallery if r.get("source") != "holidays"]
        hol = [r for r in gallery if r.get("source") == "holidays"]
        n = cfg["data"]["gallery_size"]
        sec.expect(len(coco) == n, f"{len(coco)} COCO images (data.gallery_size {n})")
        if hol:
            sec.expect(len(hol) == n_hol, f"{len(hol)} Holidays images (expected {n_hol})")
        elif cfg["holidays"]["merge"]:
            sec.add("skip", "no Holidays images merged (not downloaded, or --skip-holidays)")
        check_images(sec, "gallery images", [gallery_path(cfg, r) for r in gallery], full)

    sec = Section("semantic queries")
    out.append(sec)
    queries = _manifest(sec, root / "queries_semantic.json")
    if queries is not None:
        n = cfg["data"]["semantic_queries"]
        sec.expect(len(queries) == n, f"{len(queries)} queries (data.semantic_queries {n})")
        sec.expect(all(q["categories"] for q in queries), "every query has COCO categories")
        if gallery is not None:
            overlap = {q["image_id"] for q in queries} & {r["image_id"] for r in gallery}
            sec.expect(not overlap, f"no query is in the gallery (overlap: {len(overlap)})")
        check_images(sec, "query images", [root / "val2017" / q["file_name"] for q in queries], full)

    sec = Section("box queries")
    out.append(sec)
    boxes = _manifest(sec, root / "queries_box.json")
    if boxes is not None and queries is not None:
        cats = {q["image_id"]: set(q["categories"]) for q in queries}
        sec.expect(all(b["image_id"] in cats for b in boxes), "every box query is a semantic query")
        sec.expect(all(b["category"] in cats.get(b["image_id"], ()) for b in boxes),
                   "every box category belongs to its query")
        bad = [b for b in boxes if not (b["crop"][0] < b["crop"][2] and b["crop"][1] < b["crop"][3])]
        sec.expect(not bad, f"crop boxes have area (bad: {len(bad)})")

    sec = Section("Holidays queries")
    out.append(sec)
    hq_path = root / "queries_holidays.json"
    hq = read_manifest(hq_path) if hq_path.exists() else []
    if not hq:
        sec.add("skip", "no Holidays queries (not downloaded or not merged)")
    elif gallery is not None:
        in_gallery = {r["image_id"] for r in gallery}
        sec.expect(all(q["image_id"] in in_gallery for q in hq), "every query is in the gallery")
        sec.expect(len(hq) == n_hol_q, f"{len(hq)} queries (expected {n_hol_q})")
        groups: dict[str, int] = {}
        for r in gallery:
            if r.get("source") == "holidays":
                groups[r["group"]] = groups.get(r["group"], 0) + 1
        lonely = [q for q in hq if groups.get(q["group"], 0) < 2]
        sec.expect(not lonely, f"every query has a relevant image (without: {len(lonely)})")
    return out


def check_revisited(cfg: dict, full: bool = False) -> list[Section]:
    out = []
    for ds, (n_db, n_q) in cfg["integrity"]["revisited"].items():
        sec = Section(ds)
        out.append(sec)
        d = resolve_path(cfg, "revisited") / ds
        gnd_path = d / f"gnd_{ds}.pkl"
        if not gnd_path.exists():
            sec.add("skip", "not downloaded (optional; 01_download.py --with-roxford)")
            continue
        try:
            with open(gnd_path, "rb") as f:
                gnd = pickle.load(f)
        except Exception as e:
            sec.add("fail", f"{gnd_path.name} unreadable ({type(e).__name__}); run 01_download.py")
            continue
        sec.expect(len(gnd["imlist"]) == n_db, f"{len(gnd['imlist'])} gallery images (expected {n_db})")
        sec.expect(len(gnd["qimlist"]) == n_q, f"{len(gnd['qimlist'])} queries (expected {n_q})")
        # only the images the gnd lists: RParis's archives hold 20 corrupt ones outside it
        names = gnd["imlist"] + gnd["qimlist"]
        check_images(sec, f"{ds} images", [d / "jpg" / f"{n}.jpg" for n in names], full)
    return out


def check_cache(cfg: dict) -> list[Section]:
    """Embeddings, indexes and tag indexes must cover exactly the gallery manifest."""
    from src.index import load_index
    from src.pipeline import DENSE
    from src.schema import INDEX_KINDS
    from src.sparse.tag_index import TagIndex

    cache = resolve_path(cfg, "cache")
    manifest = resolve_path(cfg, "data") / "gallery.json"
    if not manifest.exists():
        return []
    ids = sorted(r["image_id"] for r in read_manifest(manifest))
    out = []
    for rep in DENSE:
        sec = Section(f"cache {rep}")
        out.append(sec)
        d = cache / rep
        if not (d / "vectors.npy").exists():
            sec.add("skip", f"not embedded yet (02_embed.py --models {rep})")
            continue
        vec_ids = np.load(d / "ids.npy")
        rows = np.load(d / "vectors.npy", mmap_mode="r").shape[0]
        same = sorted(vec_ids.tolist()) == ids
        sec.expect(same and rows == len(vec_ids),
                   f"{rows} vectors match the {len(ids)}-image gallery" if same else
                   f"vectors cover {len(vec_ids)} images but the gallery has {len(ids)}; "
                   f"rerun 02_embed.py --models {rep} and 04_build_index.py")
        for kind in INDEX_KINDS:
            if (d / kind / "meta.json").exists():
                index = load_index(d / kind, cfg)
                sec.expect(sorted(index.ids.tolist()) == ids,
                           f"{kind} index: {index.ntotal} vectors"
                           + ("" if sorted(index.ids.tolist()) == ids
                              else f"; rebuild with 04_build_index.py --index {kind}"))
    for v in cfg["tags"]["variants"]:
        sec = Section(f"cache {v}")
        out.append(sec)
        if not (cache / v / "postings.txt").exists():
            sec.add("skip", "not tagged yet (03_tag.py)")
            continue
        index = TagIndex.load(cache / v, cfg["bm25"]["k1"], cfg["bm25"]["b"])
        sec.expect(index.ids == ids, f"{index.N} tagged images"
                   + ("" if index.ids == ids else f", gallery has {len(ids)}; rerun 03_tag.py"))
    return out


def check_phone(cfg: dict, full: bool = False) -> Section:
    sec = Section("phone photos")
    d = resolve_path(cfg, "phone")
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    photos = sorted(p for p in d.iterdir() if p.suffix.lower() in exts) if d.is_dir() else []
    if not photos:
        sec.add("skip", f"no photos in {d} yet (user study; HEIC must be converted first)")
    else:
        check_images(sec, "phone photos", photos, full)
    return sec


def summarize(sections: list[Section]) -> dict[str, int]:
    counts = {"ok": 0, "skip": 0, "fail": 0}
    for sec in sections:
        for level, _ in sec.items:
            counts[level] += 1
    return counts
