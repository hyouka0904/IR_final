"""抽取圖庫與官方 query（bbx 裁切）的 embedding，單一尺度與 multi-scale。

輸出（順序與 imlist / qimlist 一致）：
  cache/embeddings/{dataset}/{model}_{ss|ms}_db.npy
  cache/embeddings/{dataset}/{model}_{ss|ms}_q.npy
  cache/embeddings/{dataset}/{model}_{ss|ms}_q_nocrop.npy   # 補充實驗用
  cache/embeddings/{dataset}/{model}_{ss|ms}_q_time.json    # query 抽 embedding 的延遲（逐張量測）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.data import revisited
from src.models.encoders import MODEL_NAMES, load_encoder
from src.pipeline import SCALES, embedding_path
from src.preprocess.crop import crop_bbx
from src.preprocess.multiscale import combine, extract_multiscale, scales_for


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=revisited.DATASETS)
    p.add_argument("--model", nargs="+", default=["all"], choices=[*MODEL_NAMES, "all"])
    p.add_argument("--scale", nargs="+", default=list(SCALES), choices=SCALES)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def encode_db(encoder, paths: list[str], scale_modes: list[str], cfg: dict, batch_size: int) -> dict[str, np.ndarray]:
    """每個尺度只算一次；ss 直接取 scale 1.0，ms 為各尺度合併。"""
    needed = sorted({s for mode in scale_modes for s in scales_for(mode, cfg)})
    out = {mode: [] for mode in scale_modes}
    for i in tqdm(range(0, len(paths), batch_size), desc=f"{encoder.name} db"):
        images = [revisited.load_image(p) for p in paths[i:i + batch_size]]
        per_scale = {s: encoder.encode(images, scale=s) for s in needed}
        for mode in scale_modes:
            out[mode].append(combine([per_scale[s] for s in scales_for(mode, cfg)]))
    return {mode: np.concatenate(v) for mode, v in out.items()}


def encode_queries(encoder, images, scales: list[float]) -> tuple[np.ndarray, list[float]]:
    """逐張抽取並計時，作為 embed 階段的延遲。"""
    embs, times = [], []
    for im in images:
        t = time.perf_counter()
        embs.append(extract_multiscale(encoder, [im], scales)[0])
        times.append((time.perf_counter() - t) * 1000)
    return np.stack(embs), times


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_root, cache_root = cfg["paths"]["data_root"], cfg["paths"]["cache_root"]
    gnd = revisited.load_gnd(args.dataset, data_root)
    db_paths = revisited.db_image_paths(gnd, args.dataset, data_root)
    q_raw = [revisited.load_image(p) for p in revisited.query_image_paths(gnd, args.dataset, data_root)]
    q_crop = [crop_bbx(im, revisited.query_bbx(gnd, i)) for i, im in enumerate(q_raw)]
    os.makedirs(os.path.dirname(embedding_path(cache_root, args.dataset, "x", "ss", "db")), exist_ok=True)

    models = list(MODEL_NAMES) if "all" in args.model else args.model
    for model in models:
        encoder = load_encoder(model, cfg, args.device)
        for mode, embs in encode_db(encoder, db_paths, args.scale, cfg, args.batch_size).items():
            np.save(embedding_path(cache_root, args.dataset, model, mode, "db"), embs)
        for mode in args.scale:
            scales = scales_for(mode, cfg)
            q, times = encode_queries(encoder, q_crop, scales)
            np.save(embedding_path(cache_root, args.dataset, model, mode, "q"), q)
            q_nocrop, _ = encode_queries(encoder, q_raw, scales)
            np.save(embedding_path(cache_root, args.dataset, model, mode, "q_nocrop"), q_nocrop)
            with open(embedding_path(cache_root, args.dataset, model, mode, "q_time").replace(".npy", ".json"), "w") as f:
                json.dump({"device": args.device, "ms_per_query": float(np.mean(times)), "ms": times}, f)
            print(f">> {args.dataset} {model} {mode}: db {len(db_paths)}, q {len(q)}, {np.mean(times):.1f} ms/query on {args.device}")
        del encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
