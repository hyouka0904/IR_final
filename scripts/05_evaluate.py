"""Step 5: evaluate configurations on every query set.

    python scripts/05_evaluate.py                         # cross table: every representation, flat
    python scripts/05_evaluate.py --experiments           # eval.experiments in the config
    python scripts/05_evaluate.py --configs fusion+aqe clip@hnsw dinov2+rocchio
    python scripts/05_evaluate.py --tasks semantic box --representations clip fusion

A configuration is representation[@index][+reranker...] (src/evaluate.py). Query sets (--tasks):
semantic, box, box_full (COCO val), similar (synthetic edits), holidays (INRIA Holidays) and
phone (needs results/phone/qrels.txt from 07_judging.py). By default every set whose manifest
exists runs. A configuration runs on a set only if its representation is allowed for the set's
task (tasks.<task>.allowed) and its parts are built.
Writes results/summary.csv (one row per set and configuration), results/per_query/<set>_<config>.csv
(hardest first) and results/runs/*.run|*.qrels (TREC format). To re-score with trec_eval use
the relevance level that matches eval.semantic_binary_threshold:
    trec_eval -l 2 -m map -m ndcg_cut.10 results/runs/semantic.qrels results/runs/semantic_clip.run
(trec_eval nDCG uses integer grades 0-3, so it differs slightly from our float-gain nDCG.)
Search is timed on one CPU thread.
"""
import argparse
import csv
import math

import _bootstrap  # noqa: F401
import faiss

from src.config import load_config, resolve_path
from src.data import gallery_path, read_manifest
from src.evaluate import (config_label, evaluate_box, evaluate_holidays, evaluate_phone,
                          evaluate_semantic, evaluate_similar, parse_config, per_query_sort_key,
                          write_semantic_qrels)
from src.pipeline import REPRESENTATIONS, SearchEngine

SETS = ("semantic", "box", "box_full", "similar", "holidays", "phone")
SET_TASK = {"semantic": "semantic", "box": "semantic", "box_full": "semantic", "phone": "semantic",
            "similar": "similar", "holidays": "similar"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _update_summary(path, new_rows):
    rows = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    for r in rows:  # rows written before configurations existed
        r.setdefault("config", r["representation"])
        r["config"] = r["config"] or r["representation"]
    keys = {(r["task"], r["config"]) for r in new_rows}
    rows = [r for r in rows if (r["task"], r["config"]) not in keys] + new_rows
    rows.sort(key=lambda r: (SETS.index(r["task"]) if r["task"] in SETS else 99, r["config"]))
    _write_csv(path, rows)


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)


def load_phone_qrels(path) -> dict[str, dict[int, int]]:
    qrels: dict[str, dict[int, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc, grade = line.split()
            qrels.setdefault(qid, {})[int(doc)] = int(grade)
    return qrels


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS),
                    choices=REPRESENTATIONS, help="cross-table representations (flat, no re-rank)")
    ap.add_argument("--configs", nargs="+", help="explicit configurations, e.g. fusion+aqe clip@hnsw")
    ap.add_argument("--experiments", action="store_true", help="run eval.experiments from the config")
    ap.add_argument("--tasks", nargs="+", choices=SETS, help="query sets (default: all available)")
    ap.add_argument("--similar-queries", type=int, default=cfg["data"]["similar_queries"])
    args = ap.parse_args()
    faiss.omp_set_num_threads(1)  # latency is measured on one CPU thread (README: Efficiency)

    root, results = resolve_path(cfg, "data"), resolve_path(cfg, "results")
    runs = results / "runs"
    gallery = read_manifest(root / "gallery.json")
    data = {"semantic": read_manifest(root / "queries_semantic.json")}
    for name, file in (("box", "queries_box.json"), ("holidays", "queries_holidays.json")):
        data[name] = read_manifest(root / file) if (root / file).exists() else []
    data["box_full"] = data["box"]
    phone_qrels_path = results / "phone" / "qrels.txt"
    phone_dir = resolve_path(cfg, "phone")
    data["phone"] = (sorted(p for p in phone_dir.iterdir() if p.suffix.lower() in PHOTO_EXTS)
                     if phone_qrels_path.exists() and phone_dir.is_dir() else [])
    phone_qrels = load_phone_qrels(phone_qrels_path) if data["phone"] else {}
    sets = args.tasks or [s for s in SETS if s in ("semantic", "similar") or data.get(s)]
    for s in sets:
        if s != "similar" and not data.get(s):
            print(f"skip {s}: no queries (see 01_download.py / 07_judging.py)")
    sets = [s for s in sets if s == "similar" or data.get(s)]

    # which configurations run on which set
    plan: dict[str, list[str]] = {s: [] for s in sets}
    if args.configs:
        for s in sets:
            plan[s] = list(args.configs)
    elif args.experiments:
        for s in sets:
            plan[s] = list(cfg["eval"].get("experiments", {}).get(s, []))
    else:
        for s in sets:
            plan[s] = [config_label(r) for r in args.representations]
    reps = {parse_config(c)[0] for cs in plan.values() for c in cs}
    engine = SearchEngine.from_cache(cfg, sorted(reps))
    available = engine.available()

    if "semantic" in sets:
        write_semantic_qrels(runs / "semantic.qrels", data["semantic"], gallery,
                             cfg["eval"]["semantic_binary_threshold"])
    summaries = []
    for s in sets:
        task = SET_TASK[s]
        for label in plan[s]:
            rep, index, rerank = parse_config(label)
            if rep not in cfg["tasks"][task]["allowed"]:
                continue
            if rep not in available:
                print(f"skip {s} {label}: {rep} is not built")
                continue
            if index not in engine.index_kinds(rep):
                print(f"skip {s} {label}: no {index} index (04_build_index.py --index {index})")
                continue
            if rerank and not engine.can_rerank(rep):
                print(f"skip {s} {label}: {rep} cannot be re-ranked")
                continue
            kw = {"runs_dir": runs, "index": index, "rerank": rerank}
            if s == "semantic":
                summ, rows = evaluate_semantic(engine, data[s], gallery, root / "val2017", rep, cfg, **kw)
            elif s in ("box", "box_full"):
                summ, rows = evaluate_box(engine, data[s], gallery, root / "val2017", rep, cfg,
                                          crop=s == "box", **kw)
            elif s == "similar":
                summ, rows = evaluate_similar(engine, gallery, root / "train2017", rep, cfg,
                                              args.similar_queries, **kw)
            elif s == "holidays":
                summ, rows = evaluate_holidays(engine, data[s], gallery, rep, cfg,
                                               lambda r: gallery_path(cfg, r), **kw)
            else:
                summ, rows = evaluate_phone(engine, data[s], phone_qrels, gallery, rep, cfg, **kw)
            rows.sort(key=per_query_sort_key)
            _write_csv(results / "per_query" / f"{s}_{summ['config']}.csv", rows)
            summaries.append({k: _fmt(v) for k, v in summ.items()})
            keys = ("ndcg@10", "map") if task == "semantic" else ("recall@1", "recall@10", "mrr")
            if s == "holidays":
                keys = ("map",) + keys
            print(f"{s:9s} {summ['config']:22s} " + "  ".join(f"{k}={_fmt(summ[k])}" for k in keys)
                  + f"  n={summ['n_queries']}  embed={summ['embed_ms']:.1f}ms "
                    f"search={summ['search_ms']:.2f}ms rerank={summ['rerank_ms']:.2f}ms"
                  + (f"  recall_loss={summ['recall_loss@10']:.3f}" if index != "flat" else ""))
    _update_summary(results / "summary.csv", summaries)
    print(f"wrote {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
