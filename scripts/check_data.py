"""Check that the data and caches are complete and consistent.

    python scripts/check_data.py                 # everything
    python scripts/check_data.py --full          # decode every image (slower)
    python scripts/check_data.py --only manifests cache

Optional datasets that are not there yet (Holidays, ROxford, phone photos, a cache not built)
are reported as skipped; only missing or broken files, wrong counts and id mismatches fail
(exit code 1). 01_download.py repairs the data; 02-04 rebuild the caches.
"""
import argparse
import sys

import _bootstrap  # noqa: F401

from src import integrity
from src.config import load_config

GROUPS = ("partials", "annotations", "manifests", "revisited", "phone", "cache")
MARK = {"ok": "✓", "skip": "-", "fail": "✗"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--full", action="store_true", help="decode every image completely")
    ap.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = ap.parse_args()
    cfg = load_config()
    sections = []
    if "partials" in args.only:
        sections.append(integrity.check_partials(cfg))
    if "annotations" in args.only:
        sections.append(integrity.check_annotations(cfg))
    if "manifests" in args.only:
        sections += integrity.check_manifests(cfg, args.full)
    if "revisited" in args.only:
        sections += integrity.check_revisited(cfg, args.full)
    if "phone" in args.only:
        sections.append(integrity.check_phone(cfg, args.full))
    if "cache" in args.only:
        sections += integrity.check_cache(cfg)
    for sec in sections:
        print(f"\n[{sec.name}]")
        for level, msg in sec.items:
            print(f"  {MARK[level]} {msg}")
    c = integrity.summarize(sections)
    print(f"\npassed {c['ok']}, skipped {c['skip']}, failed {c['fail']}")
    sys.exit(1 if c["fail"] else 0)


if __name__ == "__main__":
    main()
