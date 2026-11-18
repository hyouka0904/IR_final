"""Supplementary ROxford / RParis：包裝 revisitop 的 compute_map（Easy / Medium / Hard 的 mAP 與 mP@k）。"""
import sys

import numpy as np

PROTOCOLS = ("E", "M", "H")


def _compute_map(revisitop_path: str):
    if revisitop_path not in sys.path:
        sys.path.insert(0, revisitop_path)
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


def evaluate(ranks: np.ndarray, gnd: list[dict], ks: list[int], revisitop_path: str) -> tuple[dict, dict]:
    """ranks: (n_db, n_q)，每一欄是該 query 對整個圖庫的完整排序（compute_map 的要求）。

    回傳 (summary, per_query)：summary 為 {"E_mAP": .., "E_mP@1": .., ...}（百分比），per_query 為 {"E": (n_q,) AP, ...}。
    """
    compute_map = _compute_map(revisitop_path)
    summary, per_query = {}, {}
    for p in PROTOCOLS:
        m, aps, pr, _ = compute_map(ranks, protocol_gnd(gnd, p), ks)
        summary[f"{p}_mAP"] = m * 100
        for k, v in zip(ks, pr):
            summary[f"{p}_mP@{k}"] = v * 100
        per_query[p] = aps * 100
    return summary, per_query
