"""Supplementary ROxford / RParis run：同一批 zero-shot 表示法在地標 instance retrieval 上的表現。

依官方協定：query 以 gnd 的 bbx 裁切、圖庫只含 imlist、以 revisitop compute_map 計算 Easy / Medium / Hard 的 mAP 與 mP@k。
圖庫只有五、六千張，直接以精確分數排出完整排序（compute_map 需要每個 query 對整個圖庫的排序）。
需要先執行 bash scripts/01_download_data.sh --with-roxford。

輸出：cache/features/{dataset}/{rep}_{db|q}.npy、results/roxford/summary.csv、results/roxford/per_query/{dataset}_{rep}.csv
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.data import revisited
from src.eval.revisited_eval import PROTOCOLS, evaluate
from src.features import extract_phash, extract_vectors
from src.models import phash
from src.models.encoders import load_encoder
from src.preprocess.crop import crop_bbx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def full_ranking(rep: str, db: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(n_db, n_q)。pHash 以 Hamming 距離排序。"""
    if rep == "phash":
        scores = np.stack([phash.similarity(phash.hamming(db, qq), db.shape[1] * 8) for qq in q], axis=1)
    else:
        scores = db @ q.T
    return np.argsort(-scores, axis=0, kind="stable")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    rcfg, data_root, cache = cfg["roxford"], cfg["paths"]["data_root"], cfg["paths"]["cache_root"]
    out_dir = os.path.join(cfg["paths"]["results_root"], "roxford")
    os.makedirs(os.path.join(out_dir, "per_query"), exist_ok=True)
    rows = []
    for ds in rcfg["datasets"]:
        gnd = revisited.load_gnd(ds, data_root)
        db_paths = revisited.db_image_paths(gnd, ds, data_root)
        q_paths = revisited.query_image_paths(gnd, ds, data_root)
        load_q = lambda: (crop_bbx(revisited.load_image(p), revisited.query_bbx(gnd, i)) for i, p in enumerate(q_paths))
        load_db = lambda: (revisited.load_image(p) for p in db_paths)
        feat_dir = os.path.join(cache, "features", ds)
        os.makedirs(feat_dir, exist_ok=True)
        for rep in rcfg["models"]:
            paths = {s: os.path.join(feat_dir, f"{rep}_{s}.npy") for s in ("db", "q")}
            if not all(os.path.exists(p) for p in paths.values()):
                if rep == "phash":
                    hs = cfg["models"]["phash"]["hash_size"]
                    db, _ = extract_phash(tqdm(load_db(), total=len(db_paths), desc=f"{ds} phash"), hs, False)
                    q, _ = extract_phash(load_q(), hs, False)
                else:
                    enc = load_encoder(rep, cfg, args.device)
                    db, _ = extract_vectors(enc, tqdm(load_db(), total=len(db_paths), desc=f"{ds} {rep}"), len(db_paths), args.batch_size, False)
                    q, _ = extract_vectors(enc, load_q(), len(q_paths), args.batch_size, False)
                    del enc
                np.save(paths["db"], db)
                np.save(paths["q"], q)
            db, q = np.load(paths["db"]), np.load(paths["q"])
            summary, per_query = evaluate(full_ranking(rep, db, q), gnd["gnd"], rcfg["mp_ks"], cfg["paths"]["revisitop"])
            rows.append({"dataset": ds, "model": rep, **summary})
            pd.DataFrame({"query": gnd["qimlist"], **{f"{p}_AP": per_query[p] for p in PROTOCOLS}}).to_csv(
                os.path.join(out_dir, "per_query", f"{ds}_{rep}.csv"), index=False, float_format="%.4f")
            print(f">> {ds} {rep}: " + ", ".join(f"{p} mAP {summary[f'{p}_mAP']:.2f}" for p in PROTOCOLS))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
