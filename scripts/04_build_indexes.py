"""為每個圖庫建立索引：向量模型的 Flat / HNSW / IVF-PQ / cluster pruning、pHash 的二元 Flat、tag 的 inverted index。

輸出：cache/indexes/{gallery}/{rep}_{kind}.index|.npz、cache/text/{gallery}/tags_index.pkl、
      cache/indexes/{gallery}/build_time.json（每個索引的建立時間）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from src import features
from src.config import load_config
from src.data.sets import list_sets, load_set
from src.index import faiss_index
from src.models.names import VECTOR_MODELS
from src.models.tagger import load_vocab
from src.pipeline import text_index_path
from src.text.inverted_index import InvertedIndex


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--galleries", nargs="*", default=None, help="預設為全部圖庫")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cache = cfg["paths"]["cache_root"]
    faiss.omp_set_num_threads(os.cpu_count() or 1)          # 建索引可以用滿 CPU；量延遲時才限制執行緒
    for name in args.galleries or list_sets(cache, "gallery"):
        build_time = {}
        for rep in (*VECTOR_MODELS, "phash"):
            path = features.feature_path(cache, name, rep)
            if not os.path.exists(path):
                print(f"-- {name}: 沒有 {rep} 特徵，略過")
                continue
            db = np.load(path)
            for kind in (faiss_index.INDEX_KINDS if rep != "phash" else ("flat",)):
                t = time.perf_counter()
                index = faiss_index.build_index(kind, db, cfg, seed=cfg["seed"])
                build_time[f"{rep}_{kind}"] = time.perf_counter() - t
                out = faiss_index.index_file(cache, name, rep, kind)
                faiss_index.save_index(index, out)
                print(f">> {name} {rep} {kind}: {os.path.getsize(out) / 2**20:.1f} MB, {build_time[f'{rep}_{kind}']:.1f} s")

        if os.path.exists(features.tags_path(cache, name)):
            tags = features.TagSet.load(cache, name, load_vocab(cfg["tags"]["vocab"]), cfg["tags"])
            t = time.perf_counter()
            idx = InvertedIndex.build([tags.terms(i) for i in range(len(tags.top))])
            build_time["bm25_inv"] = time.perf_counter() - t
            os.makedirs(os.path.dirname(text_index_path(cache, name)), exist_ok=True)
            idx.save(text_index_path(cache, name))
            print(f">> {name} tags: {len(idx.vocabulary())} terms, 平均文件長度 {idx.avg_doc_len:.1f}")
        os.makedirs(os.path.join(cache, "indexes", name), exist_ok=True)
        with open(os.path.join(cache, "indexes", name, "build_time.json"), "w") as f:
            json.dump(build_time, f, indent=2)


if __name__ == "__main__":
    main()
