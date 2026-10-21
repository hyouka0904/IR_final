"""跑完整 96 組設定網格，產生 summary.csv 與 per_query/*.csv。

- query embedding 讀快取（cache/embeddings），embed 延遲取 02 腳本逐張量測的值。
- search / rerank 延遲在本腳本現場量測（CPU）；幾何驗證命中 inlier 快取時，取快取中記錄的計算耗時。
- recall 損失：同一組 model / scale / rerank 下，ANN 索引的 top-K 與 Flat 的 top-K 比較。
- --no-crop：不裁切 query 的補充實驗（僅 baseline：三個模型、ss、flat、none），輸出 summary_nocrop.csv。
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.data.revisited import DATASETS
from src.eval.metrics import PROTOCOLS, evaluate, recall_loss
from src.models.encoders import MODEL_NAMES
from src.pipeline import Pipeline, all_config_ids, embedding_path, index_path, parse_config_id
from src.schema import SearchRequest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--config", default=None)
    p.add_argument("--configs", nargs="*", default=None, help="只跑指定的 config_id（預設全部 96 組）")
    p.add_argument("--no-crop", action="store_true")
    return p.parse_args()


def embed_ms(cache_root: str, dataset: str, models: list[str], scale: str) -> float:
    total = 0.0
    for m in models:
        with open(embedding_path(cache_root, dataset, m, scale, "q_time").replace(".npy", ".json")) as f:
            total += json.load(f)["ms_per_query"]
    return total


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cache_root = cfg["paths"]["cache_root"]
    out_dir = os.path.join(cfg["paths"]["results_root"], args.dataset)
    os.makedirs(os.path.join(out_dir, "per_query"), exist_ok=True)
    pipe = Pipeline(args.dataset, cfg)
    gnd, nq = pipe.gnd["gnd"], len(pipe.q_paths)
    ks = cfg["eval"]["mp_ks"]

    if args.no_crop:
        config_ids, q_split = [f"{m}-ss-flat-none" for m in MODEL_NAMES], "q_nocrop"
    else:
        config_ids, q_split = args.configs or all_config_ids(), "q"
    # 不含幾何驗證的先跑，最後跑幾何驗證（inlier 快取逐步建立）
    config_ids = sorted(config_ids, key=lambda cid: parse_config_id(cid)["gv"])

    q_embs_cache, topk, rows = {}, {}, []
    for cid in tqdm(config_ids, desc=args.dataset):
        c = parse_config_id(cid)
        for m in c["models"]:
            key = (m, c["scale"])
            if key not in q_embs_cache:
                q_embs_cache[key] = np.load(embedding_path(cache_root, args.dataset, m, c["scale"], q_split))

        ranks, ids_k, lat = [], [], {"search": [], "rerank": []}
        for qidx in range(nq):
            req = SearchRequest(args.dataset, cid, qidx=qidx, crop=not args.no_crop, full_ranking=True)
            res = pipe.search(req, query_embs={m: q_embs_cache[(m, c["scale"])][qidx] for m in c["models"]})
            ranks.append(res.full_ranking)
            ids_k.append(res.ids)          # 索引實際回傳的結果（IVF-PQ 可能少於 K 個）
            for stage in lat:
                lat[stage].append(res.latency_ms.get(stage, 0.0))
        if c["gv"]:
            pipe.inlier_cache.save()

        summary, per_query = evaluate(np.stack(ranks, axis=1), gnd, ks)
        topk[cid] = ids_k
        rows.append({
            "config_id": cid, **{k: c[k] for k in ("model", "scale", "index", "rerank")}, **summary,
            "lat_embed_ms": embed_ms(cache_root, args.dataset, c["models"], c["scale"]),
            "lat_search_ms": float(np.mean(lat["search"])),
            "lat_rerank_ms": float(np.mean(lat["rerank"])),
            "index_bytes": sum(os.path.getsize(index_path(cache_root, args.dataset, m, c["scale"], c["index"])) for m in c["models"]),
        })
        if not args.no_crop:
            pd.DataFrame({"qidx": range(nq), "query": pipe.gnd["qimlist"],
                          **{f"{p}_AP": per_query[p] for p in PROTOCOLS}}).to_csv(
                os.path.join(out_dir, "per_query", f"{cid}.csv"), index=False, float_format="%.4f")

    for row in rows:
        c = parse_config_id(row["config_id"])
        flat_id = f"{c['model']}-{c['scale']}-flat-{c['rerank']}"
        row["recall_loss"] = 0.0 if c["index"] == "flat" else (
            recall_loss(topk[row["config_id"]], topk[flat_id]) if flat_id in topk else np.nan)

    name = "summary_nocrop.csv" if args.no_crop else "summary.csv"
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, name), index=False, float_format="%.4f")
    print(f">> wrote {os.path.join(out_dir, name)} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
