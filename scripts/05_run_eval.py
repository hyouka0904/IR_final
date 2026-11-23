"""對各 query set 跑 eval.experiments 中的設定，輸出整體指標、逐 query 指標與延遲。

輸出（results/{query_set}/）：
  summary.csv            每個設定的 nDCG@10 / mAP / Recall@1/10 / MRR（semantic 另有 cat_coverage_10）、
                         分階段延遲、索引大小、相對 Flat 的 recall 損失
  summary_by_edit.csv    manifest 有 edit 欄位時（coco_similar），逐 edit 類型的指標（robustness 分析）
  per_query/{config}.csv 每個 query 的指標（failure analysis 與 demo 的困難 query 清單）
  runs/{config}.trec     --write-runs 時輸出 TREC run 檔，可用官方 trec_eval 交叉驗證

--speed：改跑 eval.experiments.speed（索引比較），輸出 summary_speed.csv。

- query 特徵讀快取，embed 延遲取 03 腳本逐張量測的值；search / rerank 在本腳本現場量測（CPU、單執行緒）
- recall 損失：同 method / post 下，ANN 索引的前 K 名與 Flat 的前 K 名比較（K = retrieval.k）
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import features
from src.config import load_config
from src.data.sets import list_sets, load_qrels, load_set
from src.eval.metrics import METRICS, recall_loss
from src.eval.protocol import run_queries
from src.eval.trec import write_run
from src.index import faiss_index
from src.pipeline import Pipeline, parse_config_id, text_index_path

METRIC_COLS = (*METRICS, "cat_coverage_10")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", nargs="*", default=None, help="預設為 eval.experiments 中有列、且已準備好的 query set")
    p.add_argument("--configs", nargs="*", default=None, help="只跑指定的 config_id")
    p.add_argument("--speed", action="store_true", help="跑索引比較（eval.experiments.speed）")
    p.add_argument("--write-runs", action="store_true")
    p.add_argument("--config", default=None)
    return p.parse_args()


def needed_reps(c: dict) -> set[str]:
    reps = set()
    if c["vec"] is not None:
        reps.add(c["vec"])
    if c["text"]:
        reps.add("tags")
    return reps


def embed_ms(cache: str, set_name: str, c: dict, cfg: dict) -> float:
    """embed 階段：向量（或 pHash）抽取 + 有 BM25 時 tagger 的 CLIP 抽取（與向量模型相同時不重複計）。"""
    reps = {c["vec"]} if c["vec"] else set()
    if c["text"]:
        reps.add(cfg["tags"]["model"])
    vals = [features.load_ms_per_image(cache, set_name, r) for r in reps]
    return float(sum(v for v in vals if v is not None)) if vals and all(v is not None for v in vals) else np.nan


def index_size(cache: str, gallery: str, c: dict) -> int:
    total = 0
    if c["vec"] is not None:
        total += os.path.getsize(faiss_index.index_file(cache, gallery, c["vec"], c["index"]))
    if c["text"]:
        total += os.path.getsize(text_index_path(cache, gallery))
    return total


def evaluate_set(cfg: dict, set_name: str, config_ids: list[str], out_name: str, write_runs: bool) -> None:
    cache = cfg["paths"]["cache_root"]
    queries = load_set(cache, set_name)
    pipe = Pipeline(queries.meta["gallery"], cfg)
    qrels = load_qrels(cache, queries, pipe.gallery, cfg)
    if qrels is None:
        print(f"-- {set_name}: 沒有 qrels（phone 需要先完成人工標註），略過")
        return
    out_dir = os.path.join(cfg["paths"]["results_root"], set_name)
    os.makedirs(os.path.join(out_dir, "per_query"), exist_ok=True)
    depth, k = cfg["retrieval"]["depth"], cfg["retrieval"]["k"]
    edits = queries.column("edit") if "edit" in queries.rows[0] else None

    feats_cache: dict[str, list[dict]] = {}
    rows, by_edit, topk = [], [], {}
    for cid in tqdm(config_ids, desc=set_name):
        c = parse_config_id(cid)
        reps = needed_reps(c)
        key = ",".join(sorted(reps))
        if key not in feats_cache:
            feats_cache[key] = features.load_query_features(cache, set_name, len(queries), reps)
        rankings, per_query, lat = run_queries(pipe, queries, cid, feats_cache[key], qrels, depth)
        topk[cid] = [r[:k] for r in rankings]

        pq = pd.DataFrame(per_query)
        pq.insert(0, "query", queries.ids)
        if edits is not None:
            pq.insert(1, "edit", edits)
        pq.to_csv(os.path.join(out_dir, "per_query", f"{cid}.csv"), index=False, float_format="%.4f")
        if write_runs:
            os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
            write_run(os.path.join(out_dir, "runs", f"{cid}.trec"), queries.ids, rankings, pipe.gallery.ids, cid)

        n_lat = min(cfg["eval"]["latency_queries"], len(lat))
        rows.append({
            "config_id": cid, "method": c["method"], "index": c["index"], "post": c["post"],
            **{m: pq[m].mean() for m in METRIC_COLS if m in pq},
            "lat_embed_ms": embed_ms(cache, set_name, c, cfg),
            "lat_search_ms": float(np.mean([l.get("search", 0.0) for l in lat[:n_lat]])),
            "lat_rerank_ms": float(np.mean([l.get("rerank", 0.0) for l in lat[:n_lat]])),
            "index_bytes": index_size(cache, pipe.name, c),
            "n_queries": int(pq["map"].notna().sum()),
        })
        if edits is not None:
            for edit, g in pq.groupby("edit"):
                by_edit.append({"config_id": cid, "edit": edit, **{m: g[m].mean() for m in METRICS}, "n_queries": len(g)})

    for row in rows:
        c = parse_config_id(row["config_id"])
        flat_id = f"{c['method']}-flat-{c['post']}"
        if c["vec"] is None or c["index"] == "flat":
            row["recall_loss"] = 0.0 if c["vec"] is not None else np.nan
        else:
            row["recall_loss"] = recall_loss(topk[row["config_id"]], topk[flat_id]) if flat_id in topk else np.nan

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, out_name), index=False, float_format="%.4f")
    if by_edit:
        pd.DataFrame(by_edit).to_csv(os.path.join(out_dir, out_name.replace("summary", "summary_by_edit")),
                                     index=False, float_format="%.4f")
    print(f">> wrote {os.path.join(out_dir, out_name)} ({len(rows)} configs)")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    faiss.omp_set_num_threads(cfg["eval"]["num_threads"])
    exps = cfg["eval"]["experiments"]
    prepared = set(list_sets(cfg["paths"]["cache_root"], "queries"))
    if args.speed:
        names = args.queries or [n for n in ("coco_semantic", "coco_similar") if n in prepared]
    else:
        names = args.queries or [n for n in exps if n in prepared]
    if not names:
        sys.exit("!! cache/sets 沒有可評估的 query set，請先執行 02_prepare_data.py、03_extract_features.py、04_build_indexes.py")
    for name in names:
        config_ids = args.configs or (exps["speed"] if args.speed else exps[name])
        evaluate_set(cfg, name, config_ids, "summary_speed.csv" if args.speed else "summary.csv", args.write_runs)


if __name__ == "__main__":
    main()
