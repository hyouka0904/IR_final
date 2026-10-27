"""SIFT 比對 + RANSAC 幾何驗證；先查 inlier 快取，沒有才計算。

- keypoint 座標一律存成原圖像素座標（抽取時縮圖，存檔前換算回去），方便 demo 畫連線。
- descriptor 以 uint8 存檔省空間，比對前轉成 RootSIFT。
- inlier 數只取決於（query, 圖庫圖片）這一對，與模型、索引無關，所以所有設定共用同一份快取。
  快取只用於以 bbx 裁切的官方 query；裁切方式不同，inlier 數就不同。
"""
import os
import time

import cv2
import h5py
import numpy as np
from PIL import Image


def extract_sift(image: Image.Image, max_side: int, max_keypoints: int) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (keypoints (N, 2) float32 原圖座標, descriptors (N, 128) uint8)。"""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    r = min(1.0, max_side / max(gray.shape))
    if r < 1.0:
        gray = cv2.resize(gray, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.array([kp.pt for kp in kps], np.float32) / r
    return pts, np.clip(np.rint(desc), 0, 255).astype(np.uint8)


def _root_sift(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    d /= d.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(d)


def match_pair(q_kp, q_desc, db_kp, db_desc, cfg: dict) -> np.ndarray:
    """回傳通過 RANSAC 的匹配點 (M, 4)：每列為 (qx, qy, dbx, dby)。"""
    g = cfg["geometric"]
    if len(q_desc) < 4 or len(db_desc) < 4:
        return np.zeros((0, 4), np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(_root_sift(q_desc), _root_sift(db_desc), k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < g["ratio_test"] * m[1].distance]
    if len(good) < 4:
        return np.zeros((0, 4), np.float32)
    src = q_kp[[m.queryIdx for m in good]]
    dst = db_kp[[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, g["ransac_threshold"])
    if mask is None:
        return np.zeros((0, 4), np.float32)
    keep = mask.ravel().astype(bool)
    return np.hstack([src[keep], dst[keep]]).astype(np.float32)


class DbSift:
    """圖庫 SIFT 特徵（04_extract_sift.py 的輸出）。

    檔案格式：kp (total, 2) float32、desc (total, 128) uint8、offsets (N + 1,) int64，
    第 i 張圖的特徵為 [offsets[i], offsets[i + 1])。
    """

    def __init__(self, path: str):
        self.f = h5py.File(path, "r")
        self.offsets = self.f["offsets"][:]

    def get(self, db_idx: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.offsets[db_idx], self.offsets[db_idx + 1]
        return self.f["kp"][a:b], self.f["desc"][a:b]


class InlierCache:
    """(query, 圖庫圖片) → inlier 數；同時記下計算耗時，讓快取命中時仍能回報真實的 re-ranking 成本。

    counts (nq, ndb) int32，-1 表示尚未計算；pair_ms (nq, ndb) float32；q_ms (nq,) 為 query 抽 SIFT 的耗時。
    """

    def __init__(self, path: str, n_queries: int, n_db: int):
        self.path = path
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                self.counts, self.pair_ms, self.q_ms = f["counts"][:], f["pair_ms"][:], f["q_ms"][:]
        else:
            self.counts = np.full((n_queries, n_db), -1, np.int32)
            self.pair_ms = np.zeros((n_queries, n_db), np.float32)
            self.q_ms = np.full(n_queries, np.nan, np.float32)
        self.dirty = False

    def get(self, qidx: int, db_idx: int) -> int | None:
        c = int(self.counts[qidx, db_idx])
        return None if c < 0 else c

    def put(self, qidx: int, db_idx: int, n_inliers: int, ms: float) -> None:
        self.counts[qidx, db_idx] = n_inliers
        self.pair_ms[qidx, db_idx] = ms
        self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with h5py.File(tmp, "w") as f:
            f["counts"], f["pair_ms"], f["q_ms"] = self.counts, self.pair_ms, self.q_ms
        os.replace(tmp, self.path)
        self.dirty = False


def rerank(query_image: Image.Image, candidates: np.ndarray, db_sift: DbSift, cfg: dict,
           cache: InlierCache | None = None, qidx: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """依 inlier 數重新排序 top-K。

    達到 min_inliers 的候選依 inlier 數由多到少排在前面，其餘維持原本順序。
    回傳 (重新排序的 ids, 對應的 inlier 數, 成本 ms)。快取命中時，成本取快取中記錄的計算耗時。
    """
    g = cfg["geometric"]
    use_cache = cache is not None and qidx is not None
    q_feat, cost = None, 0.0
    inliers = np.zeros(len(candidates), np.int32)
    for i, db_idx in enumerate(candidates):
        if use_cache and (hit := cache.get(qidx, db_idx)) is not None:
            inliers[i], cost = hit, cost + float(cache.pair_ms[qidx, db_idx])
            continue
        if q_feat is None:
            t = time.perf_counter()
            q_feat = extract_sift(query_image, g["max_side"], g["max_keypoints"])
            q_ms = (time.perf_counter() - t) * 1000
            if use_cache:
                cache.q_ms[qidx] = q_ms
        t = time.perf_counter()
        inliers[i] = len(match_pair(*q_feat, *db_sift.get(int(db_idx)), cfg))
        ms = (time.perf_counter() - t) * 1000
        cost += ms
        if use_cache:
            cache.put(qidx, int(db_idx), int(inliers[i]), ms)
    # query 抽 SIFT 的成本：有現場算就用現場值，否則用快取記錄的值
    if q_feat is not None:
        cost += q_ms
    elif use_cache and not np.isnan(cache.q_ms[qidx]):
        cost += float(cache.q_ms[qidx])

    verified = inliers >= g["min_inliers"]
    # 先依「是否通過驗證」，再依 inlier 數，最後依原排序（stable）
    order = np.lexsort((np.arange(len(candidates)), -inliers * verified, ~verified))
    return candidates[order], inliers[order], cost
