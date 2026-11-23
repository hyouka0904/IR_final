"""Supplementary ROxford5k / RParis6k run under the official revisitop protocol (README: Datasets).

Queries are cropped to their gnd bounding box, the gallery is gnd['imlist'] only, and
revisitop's compute_map scores Easy / Medium / Hard mAP and mP@k from a full ranking.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

PROTOCOLS = ("E", "M", "H")


def load_gnd(directory: Path, dataset: str) -> dict:
    with open(Path(directory) / dataset / f"gnd_{dataset}.pkl", "rb") as f:
        return pickle.load(f)


def image_paths(directory: Path, dataset: str, names: list[str]) -> list[Path]:
    return [Path(directory) / dataset / "jpg" / f"{n}.jpg" for n in names]


def query_box(gnd: dict, i: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = gnd["gnd"][i]["bbx"]
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def protocol_gnd(gnd: list[dict], protocol: str) -> list[dict]:
    """Easy / Medium / Hard exactly as revisitop's example_evaluate.py defines them."""
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
            raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    return out


def _compute_map(revisitop_path: Path):
    path = str(revisitop_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    if not hasattr(np, "in1d"):  # revisitop uses np.in1d, removed in NumPy 2; isin is the same
        np.in1d = np.isin
    from evaluate import compute_map
    return compute_map


def evaluate(ranks: np.ndarray, gnd: list[dict], ks: list[int],
             revisitop_path: Path) -> tuple[dict, dict]:
    """ranks: (n_db, n_q), each column a query's full ranking of the gallery.
    Returns ({"E_mAP": .., "E_mP@1": .., ...} in percent, {"E": per-query AP (percent), ...})."""
    compute_map = _compute_map(revisitop_path)
    summary, per_query = {}, {}
    for p in PROTOCOLS:
        m, aps, pr, _ = compute_map(ranks, protocol_gnd(gnd, p), ks)
        summary[f"{p}_mAP"] = float(m * 100)
        for k, v in zip(ks, pr):
            summary[f"{p}_mP@{k}"] = float(v * 100)
        per_query[p] = np.asarray(aps) * 100
    return summary, per_query


def full_ranking(db: np.ndarray, q: np.ndarray, metric: str) -> np.ndarray:
    """(n_db, n_q) gallery positions, best first; pHash codes rank by Hamming distance."""
    if metric == "hamming":
        bits = np.unpackbits(db, axis=1).astype(np.int16)
        qbits = np.unpackbits(q, axis=1).astype(np.int16)
        scores = -(bits[:, None, :] != qbits[None, :, :]).sum(-1)
    else:
        scores = db @ q.T
    return np.argsort(-scores, axis=0, kind="stable")
