"""抽取所有 image set 的特徵：CLIP / SigLIP / DINOv2 向量、pHash、CLIP tag。

- 圖庫以 batch 抽取；query set 逐張抽取並計時（評估時 embed 階段的延遲）
- manifest 有 box 欄位的 query（coco_semantic_box）先裁切再抽
- tag 由 CLIP 向量與詞彙表的 text embedding 算出，所以 clip 必須先抽（同一次執行會依序處理）

輸出：cache/features/{set}/{rep}.npy、{rep}_time.json、tags.npz；cache/tags/{vocab}_clip.npy
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from src import features
from src.config import load_config
from src.data.sets import list_sets, load_set
from src.models import tagger
from src.models.encoders import load_encoder
from src.models.names import VECTOR_MODELS

STEPS = (*VECTOR_MODELS, "phash", "tags")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="*", default=None, help="預設為 cache/sets 下的全部")
    p.add_argument("--reps", nargs="+", default=list(STEPS), choices=STEPS)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force", action="store_true", help="已存在的特徵也重算")
    return p.parse_args()


def images_of(s):
    for i, path in enumerate(s.paths):
        yield features.load_query_image(path, s.box(i))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cache = cfg["paths"]["cache_root"]
    sets = [load_set(cache, n) for n in (args.sets or list_sets(cache))]
    if not sets:
        sys.exit("!! cache/sets 是空的，請先執行 02_prepare_data.py")
    for s in sets:
        os.makedirs(os.path.dirname(features.feature_path(cache, s.name, "x")), exist_ok=True)
    todo = lambda s, rep: args.force or not os.path.exists(features.feature_path(cache, s.name, rep))

    for rep in [r for r in args.reps if r in VECTOR_MODELS]:
        pending = [s for s in sets if todo(s, rep)]
        if not pending:
            continue
        encoder = load_encoder(rep, cfg, args.device)
        for s in pending:
            timed = s.meta["kind"] == "queries"
            embs, times = features.extract_vectors(encoder, tqdm(images_of(s), total=len(s), desc=f"{rep} {s.name}"),
                                                   len(s), args.batch_size, timed)
            np.save(features.feature_path(cache, s.name, rep), embs)
            if timed:
                features.save_times(features.time_path(cache, s.name, rep), times, args.device)
        del encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "phash" in args.reps:
        hs = cfg["models"]["phash"]["hash_size"]
        for s in [s for s in sets if todo(s, "phash")]:
            timed = s.meta["kind"] == "queries"
            bits, times = features.extract_phash(tqdm(images_of(s), total=len(s), desc=f"phash {s.name}"), hs, timed)
            np.save(features.feature_path(cache, s.name, "phash"), bits)
            if timed:
                features.save_times(features.time_path(cache, s.name, "phash"), times, "cpu")

    pending = [s for s in sets if args.force or not os.path.exists(features.tags_path(cache, s.name))]
    if "tags" in args.reps and pending:
        tcfg = cfg["tags"]
        vocab = tagger.load_vocab(tcfg["vocab"])
        encoder = load_encoder(tcfg["model"], cfg, args.device)
        text_embs = tagger.text_embeddings(encoder, vocab, tcfg["templates"])
        os.makedirs(os.path.dirname(features.text_emb_path(cache, tcfg)), exist_ok=True)
        np.save(features.text_emb_path(cache, tcfg), text_embs)
        scale = tagger.logit_scale(encoder)
        for s in pending:
            embs = np.load(features.feature_path(cache, s.name, tcfg["model"]))
            top, probs = tagger.tag_images(embs, text_embs, tcfg["top_n"], scale)
            features.TagSet.save(cache, s.name, top, probs, tcfg)
            print(f">> tags {s.name}: 例如 {[vocab[c] for c in top[0][:5]]}")


if __name__ == "__main__":
    main()
