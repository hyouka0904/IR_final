"""TREC run / qrels writers, so results can be re-scored with trec_eval."""
from collections.abc import Iterable
from pathlib import Path

# Semantic qrels grades are 0-3 (round(3 * gain)). Grades >= TREC_REL_LEVEL are exactly the
# images with gain >= eval.semantic_binary_threshold, so `trec_eval -l 2` reproduces our mAP.
# trec_eval's nDCG uses these integer grades, so it differs slightly from our float-gain nDCG.
TREC_REL_LEVEL = 2


def semantic_grade(gain: float, threshold: float) -> int:
    if gain <= 0:
        return 0
    grade = round(3 * gain)
    return max(grade, TREC_REL_LEVEL) if gain >= threshold else min(grade, TREC_REL_LEVEL - 1)


def write_run(path: Path, rows: Iterable[tuple[str, int, int, float]], tag: str) -> None:
    """Rows must come in rank order per query. trec_eval ignores the rank column and re-sorts
    by score (ties by docno), so tied scores are nudged down by 1e-6 to keep our order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_qid, prev_micro = None, None
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docid, rank, score in rows:
            micro = round(score * 1_000_000)
            if qid == prev_qid and micro >= prev_micro:
                micro = prev_micro - 1
            prev_qid, prev_micro = qid, micro
            f.write(f"{qid} Q0 {docid} {rank} {micro / 1_000_000:.6f} {tag}\n")


def write_qrels(path: Path, rows: Iterable[tuple[str, int, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docid, grade in rows:
            f.write(f"{qid} 0 {docid} {grade}\n")
