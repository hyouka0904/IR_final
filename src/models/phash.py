"""pHash：similar search 的傳統 baseline。

以 imagehash.phash 計算（灰階 → 32x32 → DCT → 取左上 8x8 低頻與中位數比較），64 bits。
存成 packed uint8（每張 8 bytes），以 FAISS 的 IndexBinaryFlat 用 Hamming 距離檢索。
相似度分數定義為 1 − hamming / bits，與向量模型的 cosine 一樣是越大越相似。
"""
import imagehash
import numpy as np
from PIL import Image


def phash_bits(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    h = imagehash.phash(image.convert("RGB"), hash_size=hash_size)
    return np.packbits(h.hash.flatten()).astype(np.uint8)


def compute(images: list[Image.Image], hash_size: int = 8) -> np.ndarray:
    """(N, hash_size² / 8) uint8。"""
    return np.stack([phash_bits(im, hash_size) for im in images])


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (B,) 或 (N, B)，b: (B,) → 每列的 Hamming 距離。"""
    return np.unpackbits(np.bitwise_xor(a, b), axis=-1).sum(-1)


def similarity(distance: np.ndarray, n_bits: int) -> np.ndarray:
    return (1.0 - np.asarray(distance, dtype=np.float32) / n_bits).astype(np.float32)
