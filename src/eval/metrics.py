"""包裝 revisitop compute_map，並計算 recall 損失與分階段延遲。"""
import sys
import time
from contextlib import contextmanager

import numpy as np

PROTOCOLS = ("E", "M", "H")


def _compute_map():
    from src.config import load_config
    path = load_config()["paths"]["revisitop"]
    if path not in sys.path:
        sys.path.insert(0, path)
    # revisitop 使用 np.in1d，新版 NumPy 已移除，以 np.isin 代替（行為相同）
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    from evaluate import compute_map
    return compute_map


def protocol_gnd(gnd: list[dict], protocol: str) -> list[dict]:
    """與 revisitop example_evaluate.py 相同的 Easy / Medium / Hard 定義。"""
    out = []
    for g in gnd:
        easy, hard, junk = (np.asarray(g[k], dtype=np.int64) for k in ("easy", "hard", "junk"))
        if protocol == "E":
            out.append({"ok": easy, "junk": np.concatenate([junk, hard])})
        elif protocol == "M":
            out.append({"ok": np.concatenate([easy, hard]), "junk": junk})
        elif protocol == "H":
            out.append({"ok": hard, "junk": np.concatenate([junk, easy])})
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
    return out


def evaluate(ranks: np.ndarray, gnd: list[dict], ks: list[int]) -> tuple[dict, dict]:
    """ranks: (n_db, n_q)，每一欄是該 query 對整個圖庫的完整排序（compute_map 的要求）。

    回傳 (summary, per_query)：summary 為 {"E_mAP": .., "E_mP@1": .., ...}（百分比），
    per_query 為 {"E": (n_q,) AP, ...}；沒有 positive 的 query 為 NaN。
    """
    compute_map = _compute_map()
    summary, per_query = {}, {}
    for p in PROTOCOLS:
        m, aps, pr, _ = compute_map(ranks, protocol_gnd(gnd, p), ks)
        summary[f"{p}_mAP"] = m * 100
        for k, v in zip(ks, pr):
            summary[f"{p}_mP@{k}"] = v * 100
        per_query[p] = aps * 100
    return summary, per_query


def query_ap(full_ranking: np.ndarray, gnd_q: dict, protocol: str = "M") -> float:
    """單一 query 的 AP（demo 顯示用）。"""
    _, aps, _, _ = _compute_map()(full_ranking[:, None], protocol_gnd([gnd_q], protocol), [])
    return float(aps[0] * 100)


def recall_loss(ann_ids: list[np.ndarray], flat_ids: list[np.ndarray]) -> float:
    """1 − mean_q |ANN top-K ∩ Flat top-K| / K。每個 query 一個 id 陣列（ANN 可能少於 K 個）。"""
    overlaps = [len(np.intersect1d(a[a >= 0], f)) / len(f) for a, f in zip(ann_ids, flat_ids)]
    return 1.0 - float(np.mean(overlaps))


class StageTimer:
    """分階段累計耗時（ms）；同一階段可進出多次。"""

    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t) * 1000)

    def add(self, name: str, ms: float) -> None:
        self.ms[name] = self.ms.get(name, 0.0) + ms

    def as_dict(self) -> dict[str, float]:
        return dict(self.ms)
