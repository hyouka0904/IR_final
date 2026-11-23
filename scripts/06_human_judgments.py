"""Pooled human judgments（proposal：100 phone photos, kappa）。標註本身由組員人工完成，本腳本只處理前後步驟。

  pool    對 phone 的每張照片，取 judging.systems 各系統的前 pool_depth 名，聯集後打亂順序，寫成
          results/phone/judgments.csv（judge_a / judge_b 欄位留空給組員填 0 / 1 / 2），
          另外產生 judgments.html（每個 query 一列，方便對照 doc_id 看圖），
          以及 pool_systems.csv（每張圖由哪些系統找到；評審標註時不要看，以免影響判斷）
  kappa   讀填好的 CSV，計算兩位評審的 Cohen's kappa（一般與線性加權），寫 results/phone/kappa.json
  qrels   依 judging.aggregate 合併兩位評審的分數，寫 cache/sets/phone/qrels.txt；之後 05_run_eval.py 就能評估 phone

pool 不會覆寫已存在的 judgments.csv（避免蓋掉人工標註），要重做請加 --force。
"""
import argparse
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import features
from src.config import ROOT, load_config
from src.data.sets import SparseQrels, load_set, set_dir
from src.eval.metrics import cohen_kappa
from src.pipeline import Pipeline, parse_config_id
from src.schema import SearchRequest

SET = "phone"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pool", "kappa", "qrels"])
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def out_dir(cfg: dict) -> str:
    d = os.path.join(cfg["paths"]["results_root"], SET)
    os.makedirs(d, exist_ok=True)
    return d


def pool(cfg: dict, force: bool) -> None:
    jcfg, cache = cfg["judging"], cfg["paths"]["cache_root"]
    csv_path = os.path.join(out_dir(cfg), "judgments.csv")
    if os.path.exists(csv_path) and not force:
        sys.exit(f"!! {csv_path} 已存在（可能已有人工標註）。確定要重做請加 --force")
    queries = load_set(cache, SET)
    pipe = Pipeline(queries.meta["gallery"], cfg)
    reps = set()
    for cid in jcfg["systems"]:
        c = parse_config_id(cid)
        reps |= ({c["vec"]} if c["vec"] else set()) | ({"tags"} if c["text"] else set())
    qfeats = features.load_query_features(cache, SET, len(queries), reps)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for qi, qid in enumerate(queries.ids):
        found: dict[int, list[str]] = {}
        for cid in jcfg["systems"]:
            res = pipe.search(SearchRequest(cid, k=jcfg["pool_depth"]), qfeats[qi])
            for d in res.ids:
                found.setdefault(int(d), []).append(cid)
        docs = list(found)
        rng.shuffle(docs)                                  # 打亂順序，評審看不出是哪個系統、第幾名
        for d in docs:
            rows.append({"query_id": qid, "doc_id": pipe.gallery.ids[d], "doc_path": pipe.gallery.rows[d]["path"],
                         "systems": ";".join(found[d]), **{j: "" for j in jcfg["judges"]}})
    df = pd.DataFrame(rows)
    df.drop(columns="systems").to_csv(csv_path, index=False)
    df[["query_id", "doc_id", "systems"]].to_csv(os.path.join(out_dir(cfg), "pool_systems.csv"), index=False)
    write_sheet(os.path.join(out_dir(cfg), "judgments.html"), queries, df)
    print(f">> {csv_path}: {len(queries)} queries, {len(df)} 列（平均每個 query {len(df) / len(queries):.1f} 張）")


def write_sheet(path: str, queries, df: pd.DataFrame) -> None:
    """靜態 HTML 對照表：左邊 query，右邊 pool 中的圖與 doc_id（順序與 CSV 相同，不顯示系統來源）。"""
    rel = lambda p: Path(os.path.relpath(ROOT / p, os.path.dirname(path))).as_posix()
    parts = ["<!doctype html><meta charset=utf-8><title>Judging sheet</title>",
             "<style>body{font-family:sans-serif}td{vertical-align:top}img{max-height:160px;margin:2px}"
             "figure{display:inline-block;margin:4px;text-align:center;font-size:12px}</style><table>"]
    for qid, row in zip(queries.ids, queries.rows):
        figs = "".join(f"<figure><img src='{html.escape(rel(p))}' loading=lazy><figcaption>{html.escape(d)}</figcaption></figure>"
                       for d, p in df.loc[df.query_id == qid, ["doc_id", "doc_path"]].itertuples(index=False))
        parts.append(f"<tr><td><b>{html.escape(qid)}</b><br><img src='{html.escape(rel(row['path']))}'></td><td>{figs}</td></tr>")
    parts.append("</table>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def load_judgments(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(out_dir(cfg), "judgments.csv"), dtype={"query_id": str, "doc_id": str},
                     encoding="utf-8-sig")             # Excel 另存「CSV UTF-8」會加 BOM
    a, b = cfg["judging"]["judges"]
    done = df[a].notna() & df[b].notna()
    if not done.all():
        print(f"-- 還有 {(~done).sum()} 列尚未由兩位評審都標完，只用已完成的 {done.sum()} 列")
    df = df[done].copy()
    bad = ~df[a].isin(cfg["judging"]["grades"]) | ~df[b].isin(cfg["judging"]["grades"])
    if bad.any():
        sys.exit(f"!! 有 {bad.sum()} 列的分數不在 {cfg['judging']['grades']} 中，例如第 {df.index[bad][0] + 2} 行")
    df[a], df[b] = df[a].astype(int), df[b].astype(int)
    return df


def kappa(cfg: dict) -> None:
    df = load_judgments(cfg)
    a, b = cfg["judging"]["judges"]
    binary = lambda s: (s >= 1).astype(int)
    out = {
        "n_pairs": int(len(df)),
        "agreement": float((df[a] == df[b]).mean()),
        "kappa": cohen_kappa(df[a], df[b]),
        "kappa_linear_weighted": cohen_kappa(df[a], df[b], "linear"),
        "kappa_binary": cohen_kappa(binary(df[a]), binary(df[b])),     # 相關（>= 1）與否
        "confusion": pd.crosstab(df[a], df[b]).to_dict(),
    }
    path = os.path.join(out_dir(cfg), "kappa.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=int)
    print(json.dumps({k: v for k, v in out.items() if k != "confusion"}, indent=2))


def qrels(cfg: dict) -> None:
    df = load_judgments(cfg)
    a, b = cfg["judging"]["judges"]
    how = cfg["judging"]["aggregate"]
    if how == "min":
        grade = np.minimum(df[a], df[b])
    elif how == "max":
        grade = np.maximum(df[a], df[b])
    elif how == "mean":
        grade = np.floor((df[a] + df[b]) / 2 + 0.5).astype(int)       # 0.5 進位
    else:
        raise ValueError(how)
    cache = cfg["paths"]["cache_root"]
    queries = load_set(cache, SET)
    gallery = load_set(cache, queries.meta["gallery"])
    q_pos = {q: i for i, q in enumerate(queries.ids)}
    d_pos = {d: i for i, d in enumerate(gallery.ids)}
    rels = [dict() for _ in queries.ids]
    for qid, did, g in zip(df.query_id, df.doc_id, grade):
        if g > 0:
            rels[q_pos[qid]][d_pos[did]] = int(g)
    path = os.path.join(set_dir(cache, SET), "qrels.txt")
    SparseQrels(rels, len(gallery)).save(path, queries.ids, gallery.ids)
    print(f">> {path}: {sum(map(len, rels))} 個 relevant（aggregate = {how}）")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    {"pool": lambda: pool(cfg, args.force), "kappa": lambda: kappa(cfg), "qrels": lambda: qrels(cfg)}[args.command]()


if __name__ == "__main__":
    main()
