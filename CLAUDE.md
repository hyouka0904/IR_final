# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Zero-shot image-to-image search engine (IR course final project, team 27). One gallery (COCO train2017 subset + INRIA Holidays), two tasks: **semantic** (same kind of thing: CLIP / SigLIP kNN, CLIP zero-shot tags + BM25, RRF fusion) and **similar** (same picture or scene: DINOv2 kNN, pHash Hamming). Nothing is trained; every model is pre-trained. `README.md` and `README.zh-TW.md` are kept in sync: change both. `HANDOFF.md` is the running log of decisions, server-run fixes and next steps (in Traditional Chinese).

## Commands

```bash
pip install -r requirements-ci.txt                 # enough for every test except -m models (no torch)
python -m pytest -q -m "not models and not smoke"  # unit tests, seconds
python -m pytest -q -m smoke                       # scripts 01-07 + check_data + demo end to end on fake data
python -m pytest -q tests/test_download.py::test_fetch_resumes_a_part   # a single test
python -m pytest -q -m models                      # real weights (downloads; needs torch/open_clip/timm)
python -m compileall -q app scripts src tests      # the CI syntax check
```

Pipeline (each step reads `configs/default.yaml`; only 01 resumes, 02-04 recompute every cache they write):

```bash
python scripts/01_download.py [--check-only] [--with-roxford]
python scripts/check_data.py
python scripts/02_embed.py --models clip siglip dinov2 phash
python scripts/03_tag.py
python scripts/04_build_index.py --index flat cluster ivfpq hnsw
python scripts/05_evaluate.py [--experiments | --configs fusion+aqe clip@hnsw | --tasks semantic box]
python scripts/06_pairs.py
python scripts/07_judging.py pool|kappa|qrels
python scripts/08_roxford.py
python app/app.py [--host 0.0.0.0 --auth user:pass]
```

CI (`.github/workflows/ci.yml`) runs the syntax check, unit tests and smoke test on **ubuntu and windows**, installing only `requirements-ci.txt`.

## Architecture

- **One engine for everything.** `src/pipeline.py` `SearchEngine.search(SearchRequest) -> SearchResponse` (types in `src/schema.py`) is called by `05_evaluate.py`, `search.py` and the demo, so the demo shows exactly what was evaluated. `app/app.py` is Gradio wiring only; the logic is in `src/demo.py`.
- **Representations** (`pipeline.REPRESENTATIONS`): `clip`, `siglip`, `dinov2`, `phash`, `tags`, `fusion`, `tags_nococo`, `fusion_nococo`. `fusion` = RRF of the clip kNN line and the `tags` BM25 line; `*_nococo` use a tag vocabulary without COCO-only labels (leakage ablation). `tasks.<task>.allowed` in the config limits which representations a task may use.
- **Configuration strings** in evaluation: `representation[@index][+reranker...]`, e.g. `clip@hnsw`, `fusion+aqe+mmr` (`src/evaluate.py` `parse_config`). Index kinds: flat, cluster (HW2 cluster pruning), ivfpq, hnsw (`src/index.py`); pHash and tags are flat only. Re-rankers: aqe, rocchio, mmr (`src/rerank.py`); Rocchio needs feedback (simulated in evaluation, user-marked in the demo).
- **Lazy loading.** Encoders (`src/encoders.py`) load weights on first use; ANN indexes load on first use and replay demo-added photos (`SearchEngine._added`) so they match flat. Model load stays out of the latency timings (embed / search / rerank, search always on one CPU thread).
- **Config.** `src/config.py` `load_config()` reads `IR_CONFIG` if set, else `configs/default.yaml`; the smoke test points `IR_CONFIG` at a temp config whose paths live under a temp dir. All parameters belong in the config, not in code. `data.gallery_size` / `semantic_queries` / `similar_queries` set the scale (2,000 / 200 / 200 by default; 40,000 / 1,000 / 1,000 is the target), and `check_data.py` checks the manifests against these values.
- **Data layer.** `src/download.py` does check-and-repair downloads (resume `.part`, retry dropped connections, re-download corrupt archives). `01_download.py` writes the manifests `data/coco/gallery.json`, `queries_semantic.json`, `queries_box.json`, `queries_holidays.json`; everything downstream reads those. `src/integrity.py` backs `check_data.py`. ROxford / RParis are optional: not downloaded, or an interrupted `.part`, is "skip"; once the gnd file is there, missing images are "fail".
- **Outputs.** `cache/<rep>/` holds embeddings and indexes, `cache/tags*/` tags in HW1 postings format (`src/sparse/`). `results/summary.csv` is merged by row (set, config), not overwritten; per-query CSVs and TREC runs are gitignored.
- **Tests.** `tests/fakes.py` provides a tiny color gallery and a fake encoder. `tests/test_smoke.py` serves fake COCO / Holidays / label files from a local HTTP server and runs the numbered scripts via `runpy`, with `get_encoder` monkeypatched, so it needs no network, GPU or weights.

## Cross-platform gotchas

The team runs Windows and a Linux server (conda env, Python 3.11; CI uses 3.12).
- Paths: build with `pathlib`, store with `.as_posix()`, and never assume `os.path.relpath` works (it raises on Windows across drives; see `_rel` in `scripts/07_judging.py`).
- Read and write text with `encoding="utf-8"` (Windows zh-TW defaults to cp950); the README also tells Windows users to set `PYTHONUTF8=1`. FAISS cannot write indexes to non-ASCII paths.
- Gradio 5 and 6 differ (`js` goes on `gr.Blocks` vs `launch()`); `app/app.py` checks the signature.
- open_clip's SigLIP tokenizer needs `transformers`; the tokenizer is built only on the first `encode_text`.

## Git Commit Guidelines

### Format
Follow Conventional Commits 1.0.0: `<type>(<scope>): <description>` (scope optional).

Types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.

### Message rules
- Write in English, imperative mood, lowercase type and scope.
- Subject ≤ 72 chars, no trailing period.
- Describe the actual change, not the task request. Avoid vague subjects like "update" or "fix bug".
- Add a body when the motivation or details are not obvious. Separate subject, body, and footer with blank lines.
- Mark breaking changes with `!` and a `BREAKING CHANGE:` footer.
- Never fabricate issue references or test results.

### Workflow
- Check `git status` and `git diff` before staging; review staged changes before committing.
- One logical change per commit; split unrelated changes.
- Never commit unrelated user modifications, secrets, credentials, or temp files.
- Run relevant tests before committing. Never bypass Git hooks.
- Never amend, rebase, or force-push without explicit permission.
- Pushing to a remote is allowed directly on working branches (feature/topic branches). Never push directly to main, master, or other default/protected branches without explicit permission.
