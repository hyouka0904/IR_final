"""Evaluation of both tasks through the shared SearchEngine."""
from pathlib import Path

import numpy as np
from PIL import Image

from src.edits import EDIT_TYPES, apply_edit, assign_edits
from src.data import select_subset
from src.metrics import (average_precision, mean_ignore_nan, ndcg_at_k, reciprocal_rank,
                         recall_at_k, semantic_gains)
from src.schema import SearchRequest
from src.trec import semantic_grade, write_qrels, write_run

STAGES = ("embed", "search", "rerank")


def _check_index_matches(engine, representation: str, gallery: list[dict]) -> None:
    ids = engine.gallery_ids(representation)
    if ids != {r["image_id"] for r in gallery}:
        raise ValueError(f"{representation} index has {len(ids)} images but the gallery manifest "
                         f"has {len(gallery)} different ones; rerun 02_embed and 04_build_index")


def write_semantic_qrels(path: Path, queries: list[dict], gallery: list[dict],
                         threshold: float) -> None:
    """Graded qrels for every query, streamed to disk (millions of lines at 40k scale)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gallery_cats = [set(r["categories"]) for r in gallery]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            for rec, g in zip(gallery, semantic_gains(set(q["categories"]), gallery_cats)):
                grade = semantic_grade(g, threshold)
                if grade > 0:
                    f.write(f"{q['image_id']} 0 {rec['image_id']} {grade}\n")


def _latency(timings: list[dict]) -> dict:
    return {f"{s}_ms": float(np.median([t[s] for t in timings])) for s in STAGES}


def evaluate_semantic(engine, queries: list[dict], gallery: list[dict], image_root: Path,
                      representation: str, cfg: dict,
                      runs_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Graded relevance = share of the query's COCO categories the result also has.
    Qrels are written separately, once, by write_semantic_qrels."""
    _check_index_matches(engine, representation, gallery)
    depth = cfg["search"]["eval_depth"]
    threshold = cfg["eval"]["semantic_binary_threshold"]
    gallery_cats = [set(r["categories"]) for r in gallery]
    pos = {r["image_id"]: i for i, r in enumerate(gallery)}
    rows, timings, run_rows = [], [], []
    for q in queries:
        gains = semantic_gains(set(q["categories"]), gallery_cats)
        with Image.open(Path(image_root) / q["file_name"]) as img:
            resp = engine.search(SearchRequest(img, "semantic", depth, representation))
        timings.append(resp.timings_ms)
        ranked = [gains[pos[r.image_id]] for r in resp.results]
        n_rel = int((gains >= threshold).sum())
        rows.append({"qid": q["image_id"],
                     "ap": average_precision([g >= threshold for g in ranked], n_rel),
                     "ndcg@10": ndcg_at_k(ranked, gains, 10)})
        qid = str(q["image_id"])
        run_rows += [(qid, r.image_id, r.rank, r.score) for r in resp.results]
    mean_ap, skipped = mean_ignore_nan([r["ap"] for r in rows])
    mean_ndcg, _ = mean_ignore_nan([r["ndcg@10"] for r in rows])
    if runs_dir is not None:
        write_run(Path(runs_dir) / f"semantic_{representation}.run", run_rows, representation)
    summary = {"task": "semantic", "representation": representation,
               "n_queries": len(rows), "n_skipped": skipped,
               "ndcg@10": mean_ndcg, "map": mean_ap, **_latency(timings)}
    return summary, rows


def evaluate_similar(engine, gallery: list[dict], image_root: Path, representation: str,
                     cfg: dict, n: int, runs_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Query = synthetic edit of a gallery image; the only relevant result is the original."""
    _check_index_matches(engine, representation, gallery)
    depth = cfg["search"]["eval_depth"]
    originals = select_subset(gallery, n, cfg["data"]["seed"])
    rows, timings, run_rows = [], [], []
    for rec, edit in zip(originals, assign_edits(len(originals))):
        with Image.open(Path(image_root) / rec["file_name"]) as img:
            query = apply_edit(img, edit, cfg["edits"])
        resp = engine.search(SearchRequest(query, "similar", depth, representation))
        timings.append(resp.timings_ms)
        ids = [r.image_id for r in resp.results]
        target = rec["image_id"]
        qid = f"{target}_{edit}"
        rows.append({"qid": qid, "edit": edit, "rr": reciprocal_rank(ids, target),
                     "recall@1": recall_at_k(ids, target, 1),
                     "recall@10": recall_at_k(ids, target, 10)})
        run_rows += [(qid, r.image_id, r.rank, r.score) for r in resp.results]
    if runs_dir is not None:
        write_run(Path(runs_dir) / f"similar_{representation}.run", run_rows, representation)
        write_qrels(Path(runs_dir) / "similar.qrels",
                    [(r["qid"], int(r["qid"].split("_")[0]), 1) for r in rows])

    def means(subset):
        return {"recall@1": float(np.mean([r["recall@1"] for r in subset])),
                "recall@10": float(np.mean([r["recall@10"] for r in subset])),
                "mrr": float(np.mean([r["rr"] for r in subset]))}

    summary = {"task": "similar", "representation": representation, "n_queries": len(rows),
               "n_skipped": 0, **means(rows)}
    for e in EDIT_TYPES:
        subset = [r for r in rows if r["edit"] == e]
        if subset:
            summary.update({f"{k}_{e}": v for k, v in means(subset).items()})
    summary.update(_latency(timings))
    return summary, rows
