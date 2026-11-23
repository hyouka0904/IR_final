"""檢查資料完整性：01 下載的原始資料集，以及 02 產生的 cache/sets。

可以在任何階段執行：還沒下載 / 還沒準備的選用資料集（Holidays、ROxford / RParis、phone 的 qrels）會標成 skip，
只有缺檔、數量不符、圖片損毀、qrels 與圖庫對不上才算失敗（結束碼 1）。

用法：
  python scripts/check_data.py                 # 全部檢查（JPEG 檢查結尾標記，可抓出截斷的檔案）
  python scripts/check_data.py --full          # 每張圖完整解碼（較慢，4 萬多張約數分鐘）
  python scripts/check_data.py --only coco sets
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from src.config import load_config
from src.data import integrity, revisited

GROUPS = ("partials", "coco", "holidays", "roxford", "sets")
MARK = {"ok": "✓", "skip": "-", "fail": "✗"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--full", action="store_true", help="每張圖都完整解碼")
    p.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = p.parse_args()
    cfg = load_config(args.config)
    data = cfg["paths"]["data_root"]

    sections = []
    if "partials" in args.only:
        sections.append(integrity.check_partials(data))
    if "coco" in args.only:
        sections.append(integrity.check_coco_annotations(data))
    if "holidays" in args.only:
        sections.append(integrity.check_holidays(data, args.full, tqdm))
    if "roxford" in args.only:
        sections += [integrity.check_revisited(data, ds, args.full, tqdm) for ds in revisited.DATASETS]
    if "sets" in args.only:
        sections += integrity.check_sets(cfg, args.full, tqdm)

    for sec in sections:
        print(f"\n[{sec.name}]")
        for level, msg in sec.items:
            print(f"  {MARK[level]} {msg}")
    n_ok, n_skip, n_fail = integrity.summarize(sections)
    print(f"\n通過 {n_ok}、略過 {n_skip}、失敗 {n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
