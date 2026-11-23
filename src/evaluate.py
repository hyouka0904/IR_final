"""Evaluation of every query set through the shared SearchEngine.

A configuration is a representation, an index kind and re-rankers, labelled like `fusion`,
`clip@hnsw` or `fusion+aqe+mmr` (config_label / parse_config). Query sets:

- semantic  COCO val2017 queries; graded gain = share of the query's categories a result has
- box       semantic queries cropped to one object; relevant = has that category (binary)
- box_full  the same queries and relevance without the crop (control for the crop's effect)
- similar   synthetic edits of COCO gallery images; the only relevant result is the original
- holidays  INRIA Holidays queries; relevant = the rest of the group, the query itself excluded
- phone     phone photos judged by pooling (scripts/07_judging.py); graded 0-2

Rocchio is evaluated with a simulated user (README: Feedback): the initial ranking's top
rerank.rocchio.feedback_depth results are judged with the set's own relevance, the query is
re-run with that feedback, and the judged results stay frozen at the top (the user already saw
them), so only what Rocchio brings below them can change the metrics. Its latency is both rounds.
An approximate index also reports recall_loss@10 = 1 - |top-10 ∩ flat top-10| / 10.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.data import select_subset
from src.edits import EDIT_TYPES, apply_edit, assign_edits
from src.metrics import (average_precision, mean_ignore_nan, ndcg_at_k, reciprocal_rank,
                         recall_at_k, semantic_gains)
from src.schema import INDEX_KINDS, RERANKERS, SearchRequest
from src.trec import semantic_grade, write_qrels, write_run

STAGES = ("embed", "search", "rerank")


# ---------------- configurations ----------------

def config_label(rep: str, index: str = "flat", rerank=()) -> str:
    return rep + (f"@{index}" if index != "flat" else "") + "".join(f"+{r}" for r in rerank)


def parse_config(label: str) -> tuple[str, str, tuple[str, ...]]:
    """'fusion@hnsw+aqe+mmr' -> ('fusion', 'hnsw', ('aqe', 'mmr'))."""
    head, *rerank = label.split("+")
    rep, _, index = head.partition("@")
    index = index or "flat"
    if index not in INDEX_KINDS:
        raise ValueError(f"{label}: index must be one of {INDEX_KINDS}")
    bad = [r for r in rerank if r not in RERANKERS]
    if bad:
        raise ValueError(f"{label}: unknown re-ranker(s) {bad}; expected {RERANKERS}")
    return rep, index, tuple(r for r in RERANKERS if r in rerank)


@dataclass
class EvalQuery:
    qid: str
    image: Callable[[], Image.Image]  # loads the query image
    relevant: Callable[[int], bool]   # for the simulated Rocchio user
    crop: tuple[int, int, int, int] | None = None
    exclude: frozenset = field(default_factory=frozenset)  # ids dropped from its ranking


def _opener(path: Path) -> Callable[[], Image.Image]:
    def load():
        with Image.open(path) as im:
            im.load()
            return im.copy()
    return load


def _search(engine, task, q: EvalQuery, img, rep, index, rerank, depth, feedback=None):
    resp = engine.search(SearchRequest(img, task, depth + len(q.exclude), rep, q.crop,
                                       index=index, rerank=rerank, feedback=feedback))
    hits = [(r.image_id, r.score) for r in resp.results if r.image_id not in q.exclude]
    return hits[:depth], resp.timings_ms


def run_query(engine, task: str, q: EvalQuery, rep: str, index: str, rerank: tuple, depth: int,
              cfg: dict) -> tuple[list[tuple[int, float]], dict]:
    """Ranked (image id, score) for one query and its stage timings."""
    img = q.image()
    if "rocchio" not in rerank:
        return _search(engine, task, q, img, rep, index, rerank, depth)
    first, t1 = _search(engine, task, q, img, rep, index,
                        tuple(r for r in rerank if r != "rocchio"), depth)
    judged = first[:cfg["rerank"]["rocchio"]["feedback_depth"]]
    rel = [d for d, _ in judged if q.relevant(d)]
    nonrel = [d for d, _ in judged if not q.relevant(d)]
    second, t2 = _search(engine, task, q, img, rep, index, rerank, depth, feedback=(rel, nonrel))
    seen = {d for d, _ in judged}
    hits = judged + [h for h in second if h[0] not in seen]  # frozen ranking
    return hits[:depth], {s: t1[s] + t2[s] for s in STAGES}


def _latency(timings: list[dict]) -> dict:
    return {f"{s}_ms": float(np.median([t[s] for t in timings])) for s in STAGES}


def _index_mb(engine, rep: str, index: str) -> float:
    from src.pipeline import vector_line
    line = vector_line(rep) or rep
    path = engine.index_dir(line, index)
    if path is None:
        return float("nan")
    from src.index import index_size_bytes
    return index_size_bytes(path) / 2**20


def run_config(engine, task: str, queries: list[EvalQuery], rep: str, index: str, rerank: tuple,
               cfg: dict, depth: int | None = None) -> tuple[list[list[tuple[int, float]]], dict]:
    """Rankings for every query plus the configuration's shared summary fields (latency, index
    size, recall loss against flat for approximate indexes)."""
    depth = depth or cfg["search"]["eval_depth"]
    rankings, timings = [], []
    for q in queries:
        hits, t = run_query(engine, task, q, rep, index, rerank, depth, cfg)
        rankings.append(hits)
        timings.append(t)
    extra = {"config": config_label(rep, index, rerank), "index": index,
             "rerank": "+".join(rerank), "index_mb": _index_mb(engine, rep, index),
             **_latency(timings)}
    if index != "flat":
        losses = []
        for q, hits in zip(queries, rankings):
            exact, _ = run_query(engine, task, q, rep, "flat", rerank, 10, cfg)
            top, ref = {d for d, _ in hits[:10]}, {d for d, _ in exact[:10]}
            losses.append(1 - len(top & ref) / max(len(ref), 1))
        extra["recall_loss@10"] = float(np.mean(losses))
    else:
        extra["recall_loss@10"] = 0.0
    return rankings, extra


def _check_index_matches(engine, representation: str, gallery: list[dict]) -> None:
    ids = engine.gallery_ids(representation)
    if ids != {r["image_id"] for r in gallery}:
        raise ValueError(f"{representation} index has {len(ids)} images but the gallery manifest "
                         f"has {len(gallery)} different ones; rerun 02_embed and 04_build_index")


def _run_rows(qids, rankings):
    return [(qid, d, rank, s) for qid, hits in zip(qids, rankings)
            for rank, (d, s) in enumerate(hits, start=1)]


# ---------------- semantic, box, phone (graded / binary relevance) ----------------

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


def _graded(engine, task, set_name, items, gains_list, threshold, gallery, rep, cfg, runs_dir,
            index, rerank, cats_of=None, query_cats=None):
    """nDCG@10 and AP for queries with a gain per gallery image (gains_list[i] aligned with
    gallery). AP counts gain >= threshold as relevant; a query without any is skipped (nan)."""
    pos = {r["image_id"]: i for i, r in enumerate(gallery)}
    rankings, extra = run_config(engine, task, items, rep, index, rerank, cfg)
    rows = []
    for i, (q, hits) in enumerate(zip(items, rankings)):
        gains = gains_list[i]
        ranked = [gains[pos[d]] for d, _ in hits]
        n_rel = int((gains >= threshold).sum())
        row = {"qid": q.qid, "ap": average_precision([g >= threshold for g in ranked], n_rel),
               "ndcg@10": ndcg_at_k(ranked, gains, 10)}
        if cats_of is not None:  # how many of the query's categories the top 10 cover (MMR)
            covered = set().union(*(cats_of[d] for d, _ in hits[:10])) if hits else set()
            row["coverage@10"] = len(query_cats[i] & covered) / len(query_cats[i])
        rows.append(row)
    mean_ap, skipped = mean_ignore_nan([r["ap"] for r in rows])
    mean_ndcg, _ = mean_ignore_nan([r["ndcg@10"] for r in rows])
    label = extra["config"]
    if runs_dir is not None:
        write_run(Path(runs_dir) / f"{set_name}_{label}.run", _run_rows([q.qid for q in items],
                                                                        rankings), label)
    summary = {"task": set_name, "representation": rep, "n_queries": len(rows),
               "n_skipped": skipped, "ndcg@10": mean_ndcg, "map": mean_ap}
    if cats_of is not None:
        summary["coverage@10"] = float(np.mean([r["coverage@10"] for r in rows]))
    return {**summary, **extra}, rows


def evaluate_semantic(engine, queries: list[dict], gallery: list[dict], image_root: Path,
                      representation: str, cfg: dict, runs_dir: Path | None = None,
                      index: str = "flat", rerank=()) -> tuple[dict, list[dict]]:
    """Graded relevance = share of the query's COCO categories the result also has.
    Qrels are written separately, once, by write_semantic_qrels."""
    _check_index_matches(engine, representation, gallery)
    threshold = cfg["eval"]["semantic_binary_threshold"]
    gallery_cats = [set(r["categories"]) for r in gallery]
    items, gains_list = [], []
    for q in queries:
        gains = semantic_gains(set(q["categories"]), gallery_cats)
        ok = {r["image_id"] for r, g in zip(gallery, gains) if g >= threshold}
        items.append(EvalQuery(str(q["image_id"]), _opener(Path(image_root) / q["file_name"]),
                               ok.__contains__))
        gains_list.append(gains)
    cats_of = {r["image_id"]: set(r["categories"]) for r in gallery}
    return _graded(engine, "semantic", "semantic", items, gains_list, threshold, gallery,
                   representation, cfg, runs_dir, index, tuple(rerank), cats_of,
                   [set(q["categories"]) for q in queries])


def evaluate_box(engine, boxes: list[dict], gallery: list[dict], image_root: Path,
                 representation: str, cfg: dict, crop: bool = True, runs_dir: Path | None = None,
                 index: str = "flat", rerank=()) -> tuple[dict, list[dict]]:
    """Box queries (crop=True) or the same queries uncropped (box_full): binary relevance =
    the gallery image has the boxed object's category."""
    _check_index_matches(engine, representation, gallery)
    items, gains_list = [], []
    for b in boxes:
        gains = np.array([1.0 if b["category"] in r["categories"] else 0.0 for r in gallery])
        ok = {r["image_id"] for r, g in zip(gallery, gains) if g}
        items.append(EvalQuery(f"{b['image_id']}_c{b['category']}",
                               _opener(Path(image_root) / b["file_name"]), ok.__contains__,
                               tuple(b["crop"]) if crop else None))
        gains_list.append(gains)
    name = "box" if crop else "box_full"
    if runs_dir is not None:
        write_qrels(Path(runs_dir) / "box.qrels",
                    [(q.qid, r["image_id"], 1) for q, g in zip(items, gains_list)
                     for r, x in zip(gallery, g) if x])
    return _graded(engine, "semantic", name, items, gains_list, 1.0, gallery, representation,
                   cfg, runs_dir, index, tuple(rerank))


def evaluate_phone(engine, photos: list[Path], qrels: dict[str, dict[int, int]],
                   gallery: list[dict], representation: str, cfg: dict,
                   runs_dir: Path | None = None, index: str = "flat",
                   rerank=()) -> tuple[dict, list[dict]]:
    """Phone photos against pooled human judgments (grades 0-2; unjudged = 0, as in trec_eval).
    AP counts grade >= 1 as relevant; nDCG uses the grades."""
    _check_index_matches(engine, representation, gallery)
    pos = {r["image_id"]: i for i, r in enumerate(gallery)}
    items, gains_list = [], []
    for p in photos:
        judged = qrels.get(p.stem, {})
        gains = np.zeros(len(gallery))
        for d, g in judged.items():
            if d in pos:
                gains[pos[d]] = g
        items.append(EvalQuery(p.stem, _opener(p), lambda d, j=judged: j.get(d, 0) >= 1))
        gains_list.append(gains)
    return _graded(engine, "semantic", "phone", items, gains_list, 1.0, gallery, representation,
                   cfg, runs_dir, index, tuple(rerank))


# ---------------- similar: synthetic edits and Holidays ----------------

def _similar_means(rows):
    return {"recall@1": float(np.mean([r["recall@1"] for r in rows])),
            "recall@10": float(np.mean([r["recall@10"] for r in rows])),
            "mrr": float(np.mean([r["rr"] for r in rows]))}


def evaluate_similar(engine, gallery: list[dict], image_root: Path, representation: str,
                     cfg: dict, n: int, runs_dir: Path | None = None, index: str = "flat",
                     rerank=()) -> tuple[dict, list[dict]]:
    """Query = synthetic edit of a COCO gallery image; the only relevant result is the original."""
    _check_index_matches(engine, representation, gallery)
    coco = [r for r in gallery if r.get("source") != "holidays"]
    originals = select_subset(coco, n, cfg["data"]["seed"])
    items, meta = [], []
    for rec, edit in zip(originals, assign_edits(len(originals))):
        def load(rec=rec, edit=edit):
            with Image.open(Path(image_root) / rec["file_name"]) as img:
                return apply_edit(img, edit, cfg["edits"])
        target = rec["image_id"]
        items.append(EvalQuery(f"{target}_{edit}", load, lambda d, t=target: d == t))
        meta.append((target, edit))
    rankings, extra = run_config(engine, "similar", items, representation, index, tuple(rerank),
                                 cfg)
    rows = []
    for q, (target, edit), hits in zip(items, meta, rankings):
        ids = [d for d, _ in hits]
        rows.append({"qid": q.qid, "edit": edit, "rr": reciprocal_rank(ids, target),
                     "recall@1": recall_at_k(ids, target, 1),
                     "recall@10": recall_at_k(ids, target, 10)})
    label = extra["config"]
    if runs_dir is not None:
        write_run(Path(runs_dir) / f"similar_{label}.run",
                  _run_rows([q.qid for q in items], rankings), label)
        write_qrels(Path(runs_dir) / "similar.qrels", [(q.qid, t, 1) for q, (t, _) in zip(items, meta)])
    summary = {"task": "similar", "representation": representation, "n_queries": len(rows),
               "n_skipped": 0, **_similar_means(rows)}
    for e in EDIT_TYPES:
        subset = [r for r in rows if r["edit"] == e]
        if subset:
            summary.update({f"{k}_{e}": v for k, v in _similar_means(subset).items()})
    return {**summary, **extra}, rows


def evaluate_holidays(engine, queries: list[dict], gallery: list[dict], representation: str,
                      cfg: dict, image_path: Callable[[dict], Path],
                      runs_dir: Path | None = None, index: str = "flat",
                      rerank=()) -> tuple[dict, list[dict]]:
    """INRIA Holidays: relevant = the other images of the query's group; the query itself is in
    the gallery and excluded from its own ranking (the standard protocol). mAP is the standard
    Holidays metric; recall@k / MRR use the first relevant result."""
    _check_index_matches(engine, representation, gallery)
    groups: dict[str, set[int]] = {}
    for r in gallery:
        if r.get("source") == "holidays":
            groups.setdefault(r["group"], set()).add(r["image_id"])
    items, rel_sets = [], []
    for q in queries:
        rel = groups.get(q["group"], set()) - {q["image_id"]}
        items.append(EvalQuery(str(q["image_id"]), _opener(image_path(q)), rel.__contains__,
                               exclude=frozenset({q["image_id"]})))
        rel_sets.append(rel)
    rankings, extra = run_config(engine, "similar", items, representation, index, tuple(rerank),
                                 cfg)
    rows = []
    for q, rel, hits in zip(items, rel_sets, rankings):
        flags = [d in rel for d, _ in hits]
        first = next((i for i, f in enumerate(flags) if f), None)
        rows.append({"qid": q.qid, "ap": average_precision(flags, len(rel)),
                     "rr": 0.0 if first is None else 1.0 / (first + 1),
                     "recall@1": float(any(flags[:1])), "recall@10": float(any(flags[:10]))})
    label = extra["config"]
    if runs_dir is not None:
        write_run(Path(runs_dir) / f"holidays_{label}.run",
                  _run_rows([q.qid for q in items], rankings), label)
        write_qrels(Path(runs_dir) / "holidays.qrels",
                    [(q.qid, d, 1) for q, rel in zip(items, rel_sets) for d in sorted(rel)])
    mean_ap, skipped = mean_ignore_nan([r["ap"] for r in rows])
    summary = {"task": "holidays", "representation": representation, "n_queries": len(rows),
               "n_skipped": skipped, "map": mean_ap, **_similar_means(rows)}
    return {**summary, **extra}, rows


def per_query_sort_key(row: dict):
    """Hardest first: lowest AP, else lowest reciprocal rank; nan last."""
    v = row.get("ap", row.get("rr", 0.0))
    return (isinstance(v, float) and math.isnan(v), v)
