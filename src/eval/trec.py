"""TREC 格式的 run / qrels 輸出，供官方 trec_eval 或 pytrec_eval 交叉驗證。

trec_eval 依分數排序，同分時以 docno 字串排序，會打亂我們的名次。
因此 run 檔的分數一律改寫成依名次遞減（depth − rank），保證 trec_eval 看到的順序與我們相同。
"""
import numpy as np


def write_run(path: str, q_ids: list[str], rankings: list[np.ndarray], doc_ids: list[str], tag: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, ranking in zip(q_ids, rankings):
            n = len(ranking)
            for r, d in enumerate(ranking):
                f.write(f"{qid} Q0 {doc_ids[int(d)]} {r + 1} {n - r} {tag}\n")


def write_qrels(path: str, q_ids: list[str], qrels, doc_ids: list[str]) -> None:
    """展開任何 Qrels（CategoryQrels 會很大，只在需要交叉驗證時用）。"""
    with open(path, "w", encoding="utf-8") as f:
        for qi, qid in enumerate(q_ids):
            g = qrels.grades(qi)
            for d in np.flatnonzero(g):
                f.write(f"{qid} 0 {doc_ids[d]} {int(g[d])}\n")


def to_pytrec(q_ids: list[str], rankings: list[np.ndarray], qrels, doc_ids: list[str]) -> tuple[dict, dict]:
    """轉成 pytrec_eval 的 (qrel, run) dict。"""
    qrel, run = {}, {}
    for qi, qid in enumerate(q_ids):
        g = qrels.grades(qi)
        rel = {doc_ids[d]: int(g[d]) for d in np.flatnonzero(g)}
        if rel:
            qrel[qid] = rel
        n = len(rankings[qi])
        run[qid] = {doc_ids[int(d)]: float(n - r) for r, d in enumerate(rankings[qi])}
    return qrel, run
