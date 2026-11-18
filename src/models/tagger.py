"""CLIP zero-shot 概念標籤：每張圖取 top-15 tag，作為 BM25 的「文件」。

詞彙表來自固定的外部類別清單（configs tags.vocab）。不能用 COCO 80 類，
因為 semantic 的 qrels 正是依 COCO 類別分級，用同一份清單會把答案洩漏給 tag 檢索。
每個類別名稱套多個 prompt template 後取平均（CLIP 論文的 prompt ensembling）。
"""
import os

import numpy as np

from src.config import ROOT


def load_vocab(spec: str) -> list[str]:
    """imagenet1k，或一行一個類別名稱的文字檔路徑。"""
    if spec == "imagenet1k":
        from open_clip.zero_shot_metadata import IMAGENET_CLASSNAMES
        return list(IMAGENET_CLASSNAMES)
    path = spec if os.path.isabs(spec) else str(ROOT / spec)
    with open(path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class names in {path}")
    return names


def text_embeddings(encoder, vocab: list[str], templates: list[str], batch: int = 256) -> np.ndarray:
    """(V, D)，每個類別 = 各 template 的 text embedding 平均後再 normalize。"""
    out = []
    for i in range(0, len(vocab), batch):
        names = vocab[i:i + batch]
        per_t = np.stack([encoder.encode_text([t.format(n) for n in names]) for t in templates])  # (T, b, D)
        e = per_t.mean(0)
        out.append(e / np.linalg.norm(e, axis=-1, keepdims=True))
    return np.concatenate(out).astype(np.float32)


def logit_scale(encoder) -> float:
    s = getattr(encoder.model, "logit_scale", None)
    return float(s.exp().item()) if s is not None else 100.0


def tag_images(image_embs: np.ndarray, text_embs: np.ndarray, top_n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (top-n 類別索引 (N, top_n), 對整個詞彙表 softmax 後的機率 (N, top_n))。"""
    logits = scale * image_embs @ text_embs.T
    logits -= logits.max(1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    top = np.argsort(-probs, axis=1, kind="stable")[:, :top_n]
    return top.astype(np.int32), np.take_along_axis(probs, top, 1).astype(np.float32)


def tag_terms(top: np.ndarray, probs: np.ndarray, vocab: list[str], tf_mode: str, tf_scale: float) -> dict[str, int]:
    """單張圖的 tag → {term: tf}。term 為類別名稱（空白換成底線），一個 tag 就是一個 term。

    binary：每個 tag tf = 1（15 個 tag 的文件長度都一樣，BM25 退化成 IDF 加權的交集）。
    weighted：tf = max(1, round(tf_scale × 機率))，讓高信心的 tag 權重較高、文件長度也有差異。
    """
    terms = {}
    for c, p in zip(top, probs):
        term = vocab[c].strip().lower().replace(" ", "_")
        tf = 1 if tf_mode == "binary" else max(1, int(round(tf_scale * float(p))))
        terms[term] = terms.get(term, 0) + tf
    return terms
