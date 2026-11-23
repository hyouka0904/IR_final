"""Pack the project into one zip a teammate can unzip and run on their own PC.

    python scripts/makezip.py              # code + data + caches: runs offline, no re-embedding
    python scripts/makezip.py --code-only  # just the git files (they rerun 01-03 + 05 themselves)
    python scripts/makezip.py --out D:/share/project.zip

Full pack = git files (tracked + new, not ignored) plus what the pipeline would otherwise
download or recompute: the COCO manifests, the gallery images listed in gallery.json, the val
images, instances_val2017.json, the label lists, the photo pairs, cache/ (embeddings, FAISS
indexes, tag indexes) and results/per_query/ (hard-queries panel).
Left out: .venv (rebuild with pip), demo photos in data/user/, instances_train2017.json
(449 MB, only 01_download needs it), results/runs/ (05_evaluate rewrites it), the homework zips.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = "ir_image_search"
STORED = {".jpg", ".jpeg", ".png", ".webp", ".zip", ".npy", ".faiss"}  # already dense
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git", ".superpowers", "dist"}


def git_files(root: Path) -> list[str]:
    """Tracked files plus new files that .gitignore does not exclude."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [f for f in out.split("\0") if f]


def _files_under(root: Path, rel: str) -> list[str]:
    base = root / rel
    if not base.is_dir():
        return []
    return [p.relative_to(root).as_posix() for p in base.rglob("*") if p.is_file()]


def collect_files(root: Path, cfg: dict, tracked: list[str],
                  code_only: bool) -> list[tuple[Path, str]]:
    """(source path, name inside the zip) for every file to pack, sorted, no duplicates."""
    root = Path(root)
    names = set(tracked)
    if not code_only:
        data = cfg["paths"]["data"]
        names |= {f"{data}/gallery.json", f"{data}/queries_semantic.json",
                  f"{data}/annotations/instances_val2017.json"}
        gallery = root / data / "gallery.json"
        if gallery.exists():
            records = json.loads(gallery.read_text(encoding="utf-8"))
            names |= {f"{data}/train2017/{r['file_name']}" for r in records}
        for rel in (f"{data}/val2017", cfg["paths"]["vocab"], cfg["pairs"]["dir"],
                    cfg["paths"]["cache"], f"{cfg['paths']['results']}/per_query"):
            names |= set(_files_under(root, rel))
    user = Path(cfg["user"]["dir"]).as_posix().rstrip("/") + "/"
    keep = []
    for name in sorted(names):
        parts = name.split("/")
        if SKIP_PARTS & set(parts) or name.startswith(user):
            continue
        if "/" not in name and name.endswith(".zip"):  # homework archives at the repo root
            continue
        path = root / name
        if path.is_file():
            keep.append((path, name))
    return keep


def write_zip(files: list[tuple[Path, str]], out: Path, top: str = TOP) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for path, name in files:
            method = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            z.write(path, f"{top}/{name}", compress_type=method)
    os.replace(tmp, out)


def main():
    sys.path.insert(0, str(ROOT))
    from src.config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--code-only", action="store_true", help="only the git files")
    ap.add_argument("--out", type=Path, help="zip path (default dist/<name>-<date>.zip)")
    args = ap.parse_args()

    cfg = load_config()
    stamp = datetime.date.today().strftime("%Y%m%d")
    out = args.out or ROOT / "dist" / f"{TOP}{'-code' if args.code_only else ''}-{stamp}.zip"
    files = collect_files(ROOT, cfg, git_files(ROOT), args.code_only)
    missing = [] if args.code_only else [
        p for p in ("cache/clip/flat/meta.json", "cache/tags/postings.txt",
                    f"{cfg['paths']['data']}/gallery.json") if not (ROOT / p).exists()]
    if missing:
        print("warning: not built yet, the zip will lack: " + ", ".join(missing))
    write_zip(files, out)
    size = sum(p.stat().st_size for p, _ in files)
    print(f"wrote {out}: {len(files)} files, {size / 2**20:.0f} MB before compression, "
          f"{out.stat().st_size / 2**20:.0f} MB zipped")
    print(f"teammate: unzip, cd {TOP}, then follow README 'Running from the zip'")


if __name__ == "__main__":
    main()
