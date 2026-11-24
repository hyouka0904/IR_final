"""Step 1: download and check the data, then write the manifests.

    python scripts/01_download.py [--gallery-size N] [--queries N] [--skip-holidays] [--with-roxford]
    python scripts/01_download.py --check-only [--with-roxford]    # report only, change nothing

Every run first checks what is already in data/ and repairs it (src/download.py): complete files
are kept, interrupted downloads resume, short or corrupt archives are fetched again, and broken
images (empty, unreadable, truncated) are deleted and downloaded or extracted again.

  1. COCO annotations (annotations_trainval2017.zip, 241 MB)
  2. the COCO subset: gallery from train2017, semantic queries from val2017 (only those images)
  3. INRIA Holidays (FTP, ~2.7 GB; optional) merged into the gallery when holidays.merge is on
  4. --with-roxford: ROxford5k / RParis6k and the revisitop evaluation code (supplementary)
Writes data/coco/gallery.json, queries_semantic.json, queries_box.json and queries_holidays.json.
"""
import argparse
import logging
import os
import pickle
import shutil
import sys
import zipfile

import _bootstrap  # noqa: F401
import requests

from src.config import load_config, resolve_path
from src.data import (box_queries, download_images, gallery_path, holidays_records,
                      load_coco_instances, select_subset, write_manifest)
from src.download import ensure_archive, extract_with_retry, fetch, split_problems

ANN_FILES = ("instances_train2017.json", "instances_val2017.json")  # URLs: config `sources`


def report_broken(broken, check_only: bool) -> None:
    if broken:
        names = ", ".join(p.name for p in broken[:5])
        print(f"   {len(broken)} broken image(s), e.g. {names}"
              + ("" if check_only else "; deleted, fetching again"))
        if not check_only:
            for p in broken:
                p.unlink()


# ---------------- 1. annotations ----------------

def annotations(cfg, root, check_only: bool) -> bool:
    print(">> COCO annotations")
    ann_dir = root / "annotations"
    zip_path = root.parent / "annotations_trainval2017.zip"
    legacy = root / "annotations_trainval2017.zip"  # where the previous version put it
    if not zip_path.exists() and legacy.exists() and not check_only:
        os.replace(legacy, zip_path)
    have_json = all((ann_dir / n).exists() for n in ANN_FILES)
    has_zip = any(p.exists() for p in (zip_path, zip_path.with_name(zip_path.name + ".part"),
                                       legacy))
    if has_zip or not have_json:
        z = ensure_archive(cfg["sources"]["coco_annotations"], zip_path if zip_path.exists() or not legacy.exists() else legacy,
                           check_only)
        if z is None:
            return False
        try:
            with zipfile.ZipFile(z) as zf:
                for n in ANN_FILES:
                    size = zf.getinfo(f"annotations/{n}").file_size
                    if not (ann_dir / n).exists() or (ann_dir / n).stat().st_size != size:
                        print(f"   {n}: missing or wrong size"
                              + ("" if check_only else "; extracting"))
                        if check_only:
                            return False
                        ann_dir.mkdir(parents=True, exist_ok=True)
                        (ann_dir / n).write_bytes(zf.read(f"annotations/{n}"))
        except zipfile.BadZipFile as e:
            print(f"   {z.name} is corrupt ({e})" + ("" if check_only else "; downloading again"))
            if check_only:
                return False
            z.unlink()
            return annotations(cfg, root, check_only)
    else:  # no zip to compare with: at least make sure the JSON files are not cut short
        for n in ANN_FILES:
            with open(ann_dir / n, "rb") as f:
                head = f.read(1)
                f.seek(-1, os.SEEK_END)
                tail = f.read(1)
            if (head, tail) != (b"{", b"}"):
                print(f"   {n}: truncated" + ("" if check_only else "; downloading again"))
                if check_only:
                    return False
                (ann_dir / n).unlink()
                return annotations(cfg, root, check_only)
    print("   complete")
    return True


# ---------------- 2. COCO subset ----------------

def coco_subset(cfg, root, gallery_size, n_queries, workers, check_only) -> list[dict] | None:
    """Download the subset; returns the COCO gallery records (None in check-only mode)."""
    session = requests.Session()

    def fetch_bytes(url: str) -> bytes:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    seed = cfg["data"]["seed"]
    gallery = None
    ok = True
    for split, n, manifest in (("train2017", gallery_size, "gallery.json"),
                               ("val2017", n_queries, "queries_semantic.json")):
        print(f">> COCO {split}: {n} images")
        records = select_subset(load_coco_instances(root / "annotations" / f"instances_{split}.json",
                                                    with_boxes=split == "val2017"), n, seed)
        paths = [root / split / r["file_name"] for r in records]
        missing, broken = split_problems(paths)
        report_broken(broken, check_only)
        if check_only:
            todo = len(missing) + len(broken)
            print(f"   {len(records) - todo} present, {todo} to download")
            ok &= todo == 0
            continue
        stats = download_images(records, root / split, fetch_bytes, workers=workers)
        failed = set(stats["failed"])
        kept = [r for r in records if r["image_id"] not in failed]
        if split == "val2017":
            write_manifest(root / "queries_box.json", box_queries(kept, cfg, seed))
            kept = [{k: v for k, v in r.items() if k not in ("width", "height", "boxes")}
                    for r in kept]
            write_manifest(root / manifest, kept)
        else:
            gallery = kept
        print(f"   {len(kept)} in {manifest} (downloaded {stats['downloaded']}, already present "
              f"{stats['skipped']}, failed {len(failed)})")
    return gallery if not check_only else (None if ok else False)


# ---------------- 3. Holidays ----------------

def holidays(cfg, check_only: bool) -> bool:
    print(">> INRIA Holidays")
    d = resolve_path(cfg, "holidays")
    n_expected = cfg["integrity"]["holidays"]["images"]
    jpgs = sorted((d / "jpg").glob("*.jpg")) if (d / "jpg").is_dir() else []
    _, broken = split_problems(jpgs)
    report_broken(broken, check_only)
    n_ok = len(jpgs) - len(broken)
    if n_ok >= n_expected:
        print(f"   complete ({n_ok} images)")
        return True
    print(f"   {n_ok} / {n_expected} images usable")
    urls = cfg["sources"]["holidays"]
    for url in urls:
        try:
            archive = ensure_archive(url, d / url.rsplit("/", 1)[1], check_only)
        except Exception as e:  # FTP blocked etc.: Holidays is optional, carry on without it
            print(f"!! Holidays download failed ({type(e).__name__}: {e}); skipped.\n"
                  f"   Some networks block FTP. Try another network (or a mirror in config\n"
                  f"   sources.holidays), or fetch {', '.join(urls)}\n"
                  f"   elsewhere, put the files in {d} and run this script again.", file=sys.stderr)
            return False
        if archive is not None and not check_only:
            extract_with_retry(url, archive, d / "jpg")
    if check_only:
        return False
    n = len(list((d / "jpg").glob("*.jpg")))
    print(f"   {'complete' if n >= n_expected else 'still incomplete'} ({n} images)")
    return n >= n_expected


# ---------------- 4. ROxford / RParis ----------------

def revisitop_code(cfg) -> None:
    code = resolve_path(cfg, "revisitop")
    if (code / "evaluate.py").exists():
        print(">> revisitop code: present")
        return
    repo = code.parent
    z = fetch(cfg["sources"]["revisitop_code"], repo.parent / "revisitop-master.zip")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(repo.parent)
    if repo.exists():
        shutil.rmtree(repo)
    os.replace(repo.parent / "revisitop-master", repo)
    z.unlink()
    print(f">> revisitop code -> {repo}")


def _load_gnd(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError):
        return None


def revisited(cfg, check_only: bool) -> bool:
    """Complete = every image the gnd file lists (gallery + queries) is readable. RParis's
    archives contain 20 corrupt images that the gnd file leaves out."""
    all_ok = True
    for ds, urls in cfg["sources"]["revisited"].items():
        print(f">> {ds}")
        d = resolve_path(cfg, "revisited") / ds
        gnd_url, gnd_path = cfg["sources"]["revisited_gnd"].format(ds), d / f"gnd_{ds}.pkl"
        gnd = _load_gnd(gnd_path) if ensure_archive(gnd_url, gnd_path, check_only) else None
        if gnd is None and gnd_path.exists():
            print(f"   {gnd_path.name} is unreadable" + ("" if check_only else "; downloading again"))
            if not check_only:
                gnd_path.unlink()
                gnd = _load_gnd(fetch(gnd_url, gnd_path))
        if gnd is None:
            all_ok = False
            continue
        paths = [d / "jpg" / f"{n}.jpg" for n in gnd["imlist"] + gnd["qimlist"]]
        missing, broken = split_problems(paths)
        report_broken(broken, check_only)
        need = len(missing) + len(broken)
        if need == 0:
            print(f"   complete ({len(paths)} images)")
            continue
        if len(missing) == len(paths):
            print(f"   images not extracted yet ({len(paths)})"
                  + ("" if check_only else "; downloading the archives"))
        else:
            print(f"   {need} / {len(paths)} images missing or broken"
                  + ("" if check_only else "; extracting again"))
        all_ok = False
        for url in urls:
            archive = ensure_archive(url, d / url.rsplit("/", 1)[1], check_only)
            if archive is not None and not check_only:
                extract_with_retry(url, archive, d / "jpg")
                archive.unlink()  # the images are out; the archives are large
        if not check_only:
            missing, broken = split_problems(paths)
            if missing or broken:
                sys.exit(f"!! {ds}: {len(missing) + len(broken)} images still missing or broken "
                         "after extraction; if the source moved, see the revisitop issues")
            print(f"   complete ({len(paths)} images)")
            all_ok = True
    return all_ok


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--gallery-size", type=int, default=cfg["data"]["gallery_size"])
    ap.add_argument("--queries", type=int, default=cfg["data"]["semantic_queries"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-holidays", action="store_true", help="leave INRIA Holidays out")
    ap.add_argument("--with-roxford", action="store_true",
                    help="also ROxford5k / RParis6k (supplementary run, ~4.5 GB)")
    ap.add_argument("--check-only", action="store_true",
                    help="report what is complete, partial or broken; download nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = resolve_path(cfg, "data")
    status = {"annotations": annotations(cfg, root, args.check_only)}
    if not status["annotations"] and not args.check_only:
        sys.exit("!! the COCO annotations are not usable")
    if status["annotations"]:  # the subset is picked from the annotations
        gallery = coco_subset(cfg, root, args.gallery_size, args.queries, args.workers,
                              args.check_only)
        status["COCO subset"] = gallery is not False
    use_holidays = cfg["holidays"]["merge"] and not args.skip_holidays
    if use_holidays:
        status["Holidays"] = holidays(cfg, args.check_only)
    if not args.check_only:
        h_gallery, h_queries = (holidays_records(resolve_path(cfg, "holidays") / "jpg")
                                if use_holidays and status["Holidays"] else ([], []))
        write_manifest(root / "gallery.json", gallery + h_gallery)
        write_manifest(root / "queries_holidays.json", h_queries)
        print(f">> gallery.json: {len(gallery)} COCO + {len(h_gallery)} Holidays images; "
              f"{len(h_queries)} Holidays queries")
        missing = [r for r in gallery + h_gallery if not gallery_path(cfg, r).exists()]
        if missing:
            sys.exit(f"!! {len(missing)} gallery images are missing, e.g. {missing[0]['file_name']}")
    if args.with_roxford:
        if not args.check_only:
            revisitop_code(cfg)
        status["ROxford / RParis"] = revisited(cfg, args.check_only)
    print("\n" + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in status.items()))
    if args.check_only and not all(status.values()):
        print("run without --check-only to resume or repair")


if __name__ == "__main__":
    main()
