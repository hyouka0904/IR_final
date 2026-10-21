"""畫出 query 與結果圖之間的 SIFT inlier 匹配連線。"""
import cv2
import numpy as np
from PIL import Image


def draw_inliers(query_image: Image.Image, db_image: Image.Image, matches: np.ndarray, height: int = 480) -> np.ndarray:
    """左右並排兩張圖（等高），每個 inlier 畫一條線。matches: (M, 4) 為 (qx, qy, dbx, dby) 原圖座標。"""
    q, d = np.asarray(query_image.convert("RGB")), np.asarray(db_image.convert("RGB"))
    rq, rd = height / q.shape[0], height / d.shape[0]
    q = cv2.resize(q, (max(1, round(q.shape[1] * rq)), height))
    d = cv2.resize(d, (max(1, round(d.shape[1] * rd)), height))
    canvas = np.hstack([q, d]).copy()
    rng = np.random.default_rng(0)
    for qx, qy, dx, dy in matches:
        color = tuple(int(c) for c in rng.integers(64, 256, 3))
        p1 = (round(qx * rq), round(qy * rq))
        p2 = (round(dx * rd) + q.shape[1], round(dy * rd))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1)
        cv2.circle(canvas, p2, 3, color, -1)
    return canvas
