"""Step 5: evaluate every representation on both tasks (the cross table).

    python scripts/05_evaluate.py [--representations clip fusion ...]
                                  [--tasks semantic similar] [--similar-queries N]
Each representation runs only on the tasks whose tasks.<task>.allowed lists it, and only if
its cache is loaded (default: every available one).
Writes results/summary.csv, results/per_query/<task>_<rep>.csv (hardest first)
and results/runs/*.run|*.qrels (TREC format). To re-score with trec_eval use the
relevance level that matches eval.semantic_binary_threshold:
    trec_eval -l 2 -m map -m ndcg_cut.10 results/runs/semantic.qrels results/runs/semantic_clip.run
(trec_eval nDCG uses integer grades 0-3, so it differs slightly from our float-gain nDCG.)
"""
import argparse
import csv
import math

import _bootstrap  # noqa: F401

from src.config import load_config, resolve_path
from src.data import read_manifest
from src.evaluate import evaluate_semantic, evaluate_similar, write_semantic_qrels
from src.pipeline import REPRESENTATIONS, SearchEngine
from src.schema import TASKS


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
    keys = {(r["task"], r["representation"]) for r in new_rows}
    rows = [r for r in rows if (r["task"], r["representation"]) not in keys] + new_rows
    rows.sort(key=lambda r: (r["task"], r["representation"]))
    _write_csv(path, rows)


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else str(v)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS),
                    choices=REPRESENTATIONS)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--similar-queries", type=int, default=cfg["data"]["similar_queries"])
    args = ap.parse_args()

    root, results = resolve_path(cfg, "data"), resolve_path(cfg, "results")
    gallery = read_manifest(root / "gallery.json")
    queries = read_manifest(root / "queries_semantic.json")
    engine = SearchEngine.from_cache(cfg, args.representations)
    if "semantic" in args.tasks:
        write_semantic_qrels(results / "runs" / "semantic.qrels", queries, gallery,
                             cfg["eval"]["semantic_binary_threshold"])
    summaries = []
    available = engine.available()
    for rep in args.representations:
        if rep not in available:
            print(f"skip {rep}: no cache loaded")
            continue
        for task in args.tasks:
            if rep not in cfg["tasks"][task]["allowed"]:
                continue
            if task == "semantic":
                s, rows = evaluate_semantic(engine, queries, gallery, root / "val2017", rep, cfg,
                                            runs_dir=results / "runs")
                rows.sort(key=lambda r: (math.isnan(r["ap"]), r["ap"]))
            else:
                s, rows = evaluate_similar(engine, gallery, root / "train2017", rep, cfg,
                                           args.similar_queries, runs_dir=results / "runs")
                rows.sort(key=lambda r: r["rr"])
            _write_csv(results / "per_query" / f"{task}_{rep}.csv", rows)
            summaries.append({k: _fmt(v) for k, v in s.items()})
            main_keys = ("ndcg@10", "map") if task == "semantic" else ("recall@1", "recall@10", "mrr")
            print(f"{task:8s} {rep:13s} " + "  ".join(f"{k}={_fmt(s[k])}" for k in main_keys)
                  + f"  n={s['n_queries']} skipped={s['n_skipped']}"
                  + f"  embed={s['embed_ms']:.1f}ms search={s['search_ms']:.2f}ms")
    _update_summary(results / "summary.csv", summaries)
    print(f"wrote {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
