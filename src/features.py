"""各 image set 的離線特徵：檔案位置、抽取與載入。離線腳本與線上 pipeline 共用。

cache/features/{set}/{rep}.npy          rep ∈ clip | siglip | dinov2 | phash，順序與 manifest 一致
cache/features/{set}/{rep}_time.json    逐張量測的抽取延遲（只有 query set 會量）
cache/features/{set}/tags.npz           CLIP tag：top（類別索引）與 probs
cache/tags/{vocab}_{model}.npy          詞彙表的 text embedding
"""
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image

from src.config import ROOT
from src.models import phash, tagger
from src.preprocess.crop import crop_user_box

REPS = ("clip", "siglip", "dinov2", "phash")


def feature_path(cache_root: str, set_name: str, rep: str) -> str:
    return os.path.join(cache_root, "features", set_name, f"{rep}.npy")


def time_path(cache_root: str, set_name: str, rep: str) -> str:
    return os.path.join(cache_root, "features", set_name, f"{rep}_time.json")


def tags_path(cache_root: str, set_name: str) -> str:
    return os.path.join(cache_root, "features", set_name, "tags.npz")


def vocab_name(spec: str) -> str:
    """詞彙表的檔名代號；自訂檔案以內容雜湊區分，換詞彙表就不會誤用舊的快取。"""
    if spec == "imagenet1k":
        return spec
    with open(spec if os.path.isabs(spec) else ROOT / spec, "rb") as f:
        return "custom_" + hashlib.sha1(f.read()).hexdigest()[:8]


def text_emb_path(cache_root: str, tcfg: dict) -> str:
    return os.path.join(cache_root, "tags", f"{vocab_name(tcfg['vocab'])}_{tcfg['model']}.npy")


def load_query_image(path: str, box=None) -> Image.Image:
    """manifest 有 box 欄位時先裁切，走與 demo 框選相同的程式。"""
    with Image.open(path) as im:
        image = im.convert("RGB")
    return crop_user_box(image, box) if box is not None else image


def extract_vectors(encoder, image_iter, n: int, batch_size: int, timed: bool) -> tuple[np.ndarray, list[float]]:
    """timed=True 時逐張抽取並計時（query 的 embed 延遲）；否則以 batch 抽取。"""
    out, times, batch = [], [], []
    for im in image_iter:
        if timed:
            t = time.perf_counter()
            out.append(encoder.encode([im]))
            times.append((time.perf_counter() - t) * 1000)
            continue
        batch.append(im)
        if len(batch) == batch_size:
            out.append(encoder.encode(batch))
            batch = []
    if batch:
        out.append(encoder.encode(batch))
    return (np.concatenate(out) if out else np.zeros((0, encoder.dim), np.float32)), times


def extract_phash(image_iter, hash_size: int, timed: bool) -> tuple[np.ndarray, list[float]]:
    out, times = [], []
    for im in image_iter:
        t = time.perf_counter()
        out.append(phash.phash_bits(im, hash_size))
        times.append((time.perf_counter() - t) * 1000)
    return np.stack(out), (times if timed else [])


def save_times(path: str, times: list[float], device: str) -> None:
    with open(path, "w") as f:
        json.dump({"device": device, "ms_per_image": float(np.mean(times)), "ms": times}, f)


def load_ms_per_image(cache_root: str, set_name: str, rep: str) -> float | None:
    p = time_path(cache_root, set_name, rep)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["ms_per_image"]


class TagSet:
    """一個 image set 的 tag（top 類別與機率），可轉成 BM25 的 {term: tf}。"""

    def __init__(self, top: np.ndarray, probs: np.ndarray, vocab: list[str], tcfg: dict):
        self.top, self.probs, self.vocab, self.tcfg = top, probs, vocab, tcfg

    def terms(self, i: int) -> dict[str, int]:
        return tagger.tag_terms(self.top[i], self.probs[i], self.vocab, self.tcfg["tf_mode"], self.tcfg["tf_scale"])

    def labels(self, i: int) -> list[tuple[str, float]]:
        return [(self.vocab[c], float(p)) for c, p in zip(self.top[i], self.probs[i])]

    @classmethod
    def load(cls, cache_root: str, set_name: str, vocab: list[str], tcfg: dict) -> "TagSet":
        z = np.load(tags_path(cache_root, set_name))
        if str(z["vocab"]) != vocab_name(tcfg["vocab"]):
            raise ValueError(f"{set_name} 的 tag 是用 {z['vocab']} 詞彙表算的，請重跑 03_extract_features.py")
        return cls(z["top"], z["probs"], vocab, tcfg)

    @staticmethod
    def save(cache_root: str, set_name: str, top: np.ndarray, probs: np.ndarray, tcfg: dict) -> None:
        np.savez(tags_path(cache_root, set_name), top=top, probs=probs, vocab=vocab_name(tcfg["vocab"]))
