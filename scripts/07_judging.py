"""Step 7: pooled human judgments of the phone photos (README: Evaluation, user study).

    python scripts/07_judging.py pool [--force]   # pool -> results/phone/judgments.csv + .html
    python scripts/07_judging.py kappa            # agreement of the two judges -> kappa.json
    python scripts/07_judging.py qrels            # combined grades -> results/phone/qrels.txt
    python scripts/05_evaluate.py --tasks phone   # then evaluate the phone set

Put the ~100 phone photos (JPEG / PNG / WebP; convert HEIC first) in paths.phone.
pool: for every photo, the top judging.pool_depth results of every judging.systems configuration,
merged and shuffled, so a judge cannot tell which system found an image or at what rank. The two
judges fill the judge_a / judge_b columns with 0 (not relevant), 1 (partly) or 2 (relevant),
using judgments.html to see the images. Which systems found each image is in pool_systems.csv;
don't open it while judging. pool never overwrites an existing judgments.csv (use --force).
Images missing from the pool count as not relevant, as in trec_eval.
"""
import argparse
import csv
import html
import json
import os
import random
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from PIL import Image

from src.config import ROOT, load_config, resolve_path
from src.data import gallery_path, read_manifest
from src.evaluate import parse_config
from src.metrics import cohen_kappa
from src.pipeline import SearchEngine
from src.schema import SearchRequest

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def out_dir(cfg) -> Path:
    d = resolve_path(cfg, "results") / "phone"
    d.mkdir(parents=True, exist_ok=True)
    return d


def phone_photos(cfg) -> list[Path]:
    d = resolve_path(cfg, "phone")
    photos = sorted(p for p in d.iterdir() if p.suffix.lower() in PHOTO_EXTS) if d.is_dir() else []
    if not photos:
        sys.exit(f"!! no photos in {d} (JPEG / PNG / WebP; convert HEIC first)")
    return photos


def _rel(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def pool(cfg, force: bool) -> None:
    j = cfg["judging"]
    out = out_dir(cfg)
    csv_path = out / "judgments.csv"
    if csv_path.exists() and not force:
        sys.exit(f"!! {csv_path} exists (it may hold judgments already); add --force to redo it")
    photos = phone_photos(cfg)
    systems = [parse_config(s) for s in j["systems"]]
    engine = SearchEngine.from_cache(cfg, sorted({r for r, _, _ in systems}))
    available = engine.available()
    systems = [s for s in systems if s[0] in available and "rocchio" not in s[2]]
    if not systems:
        sys.exit("!! none of judging.systems is built")
    paths = {r["image_id"]: gallery_path(cfg, r)
             for r in read_manifest(resolve_path(cfg, "data") / "gallery.json")}
    rng = random.Random(cfg["data"]["seed"])
    rows, sources = [], []
    for photo in photos:
        found: dict[int, list[str]] = {}
        with Image.open(photo) as img:
            img.load()
            for rep, index, rerank in systems:
                label = rep + (f"@{index}" if index != "flat" else "") + "".join(f"+{r}" for r in rerank)
                resp = engine.search(SearchRequest(img, "semantic", j["pool_depth"], rep,
                                                   index=index, rerank=rerank))
                for r in resp.results:
                    found.setdefault(r.image_id, []).append(label)
        docs = list(found)
        rng.shuffle(docs)  # order carries no hint of system or rank
        for d in docs:
            rows.append({"query_id": photo.stem, "doc_id": d,
                         "doc_path": _rel(paths[d], ROOT), **{name: "" for name in j["judges"]}})
            sources.append({"query_id": photo.stem, "doc_id": d, "systems": ";".join(found[d])})
    for path, data in ((csv_path, rows), (out / "pool_systems.csv", sources)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    write_sheet(out / "judgments.html", photos, rows)
    print(f"{csv_path}: {len(photos)} photos, {len(rows)} rows "
          f"({len(rows) / len(photos):.1f} per photo); see judgments.html")


def write_sheet(path: Path, photos: list[Path], rows: list[dict]) -> None:
    """Static page: each photo next to its pooled images and doc ids, in the CSV's order."""
    parts = ["<!doctype html><meta charset=utf-8><title>Judging sheet</title>",
             "<style>body{font-family:sans-serif}td{vertical-align:top;border-top:1px solid #ccc}"
             "img{max-height:160px;margin:2px}figure{display:inline-block;margin:4px;"
             "text-align:center;font-size:12px}</style><table>"]
    for photo in photos:
        figs = "".join(
            f"<figure><img src='{html.escape(_rel(ROOT / r['doc_path'], path.parent))}' loading=lazy>"
            f"<figcaption>{r['doc_id']}</figcaption></figure>"
            for r in rows if r["query_id"] == photo.stem)
        parts.append(f"<tr><td><b>{html.escape(photo.stem)}</b><br>"
                     f"<img src='{html.escape(_rel(photo, path.parent))}'></td><td>{figs}</td></tr>")
    parts.append("</table>")
    path.write_text("\n".join(parts), encoding="utf-8")


def load_judgments(cfg) -> list[dict]:
    """Rows both judges have graded. Excel's "CSV UTF-8" adds a BOM, hence utf-8-sig."""
    a, b = cfg["judging"]["judges"]
    grades = {str(g) for g in cfg["judging"]["grades"]}
    with open(out_dir(cfg) / "judgments.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if r[a].strip() and r[b].strip()]
    if len(done) < len(rows):
        print(f"-- {len(rows) - len(done)} rows are not graded by both judges yet; "
              f"using the {len(done)} that are")
    for n, r in enumerate(done):
        for judge in (a, b):
            v = r[judge].strip().removesuffix(".0")  # Excel may write 2 as 2.0
            if v not in grades:
                sys.exit(f"!! {judge} = {r[judge]!r} for {r['query_id']} / {r['doc_id']}; "
                         f"grades are {sorted(grades)}")
            r[judge] = int(v)
    return done


def kappa(cfg) -> None:
    a, b = cfg["judging"]["judges"]
    rows = load_judgments(cfg)
    if not rows:
        sys.exit("!! nothing graded yet")
    ga, gb = [r[a] for r in rows], [r[b] for r in rows]
    binary = lambda g: [int(x >= 1) for x in g]
    confusion: dict[str, dict[str, int]] = {}
    for x, y in zip(ga, gb):
        confusion.setdefault(str(x), {}).setdefault(str(y), 0)
        confusion[str(x)][str(y)] += 1
    out = {"n_pairs": len(rows), "agreement": sum(x == y for x, y in zip(ga, gb)) / len(rows),
           "kappa": cohen_kappa(ga, gb), "kappa_linear_weighted": cohen_kappa(ga, gb, "linear"),
           "kappa_binary": cohen_kappa(binary(ga), binary(gb)), "confusion": confusion}
    (out_dir(cfg) / "kappa.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=2))


def qrels(cfg) -> None:
    a, b = cfg["judging"]["judges"]
    how = cfg["judging"]["aggregate"]
    combine = {"min": min, "max": max, "mean": lambda x, y: int((x + y) / 2 + 0.5)}[how]
    rows = load_judgments(cfg)
    path = out_dir(cfg) / "qrels.txt"
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            grade = combine(r[a], r[b])
            if grade > 0:
                f.write(f"{r['query_id']} 0 {r['doc_id']} {grade}\n")
                n += 1
    print(f"{path}: {n} relevant judgments ({how} of the two judges); "
          f"next: python scripts/05_evaluate.py --tasks phone")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["pool", "kappa", "qrels"])
    ap.add_argument("--force", action="store_true", help="pool: overwrite judgments.csv")
    args = ap.parse_args()
    cfg = load_config()
    if args.command == "pool":
        pool(cfg, args.force)
    elif args.command == "kappa":
        kappa(cfg)
    else:
        qrels(cfg)


if __name__ == "__main__":
    main()
