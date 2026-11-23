# Week 2 (Tags + BM25 + RRF, Demo, Photo Pairs) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the explainable semantic line (CLIP zero-shot tags → HW1-format inverted index → HW2 BM25 → RRF with CLIP kNN), a Gradio demo with add-to-gallery, and a measured photo-pair check.

**Architecture:** `src/tagging.py` builds label vocabularies and tags images from CLIP vectors. `src/sparse/` holds the copied HW1 posting format and a BM25 `TagIndex`. `src/rerank.py` holds RRF. `SearchEngine` gains tag and fusion retrievers plus `add_image`. The demo (`app/`) and `scripts/06_pairs.py` call the same `SearchEngine.search`.

**Tech Stack:** existing (Python 3.14 `.venv`, torch CPU, open_clip, timm, faiss-cpu, numpy, pytest), plus `gradio`.

**Spec:** `docs/superpowers/specs/2026-11-20-week2-tags-fusion-demo-design.md` (approved 2026-11-20). Project spec: `README.md`.

## Global Constraints

- All parameters live in `configs/default.yaml`, never hard-coded.
- Evaluation, demo and pairs script all go through `SearchEngine.search`.
- Latency is recorded in three stages, `embed` / `search` / `rerank`. Query tagging counts toward `embed`; BM25 and RRF count toward `search`. One CLIP query embedding per request is shared by clip, tags and fusion.
- BM25 = HW2: score(q, d) = Σ_t qtf(t) · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)), idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5)), k1 = 1.2, b = 0.75. Parity within `math.isclose(rel_tol=1e-9)`.
- Tagging: prompt "a photo of a {}.", softmax of 100 × cosine over the vocabulary, top 15, `tf = 1 + round((tf_max − 1) · p / p_top1)`, tf_max = 5.
- RRF: 1/(k + rank), rrf_k = 60, each line cut at depth 100, ties by image id ascending.
- Allowed representations — semantic: `clip, dinov2, phash, tags, fusion, tags_nococo, fusion_nococo`; similar: `clip, dinov2, phash`. Default semantic = `fusion`, similar = `dinov2`.
- User demo photos get ids ≥ 1,000,000,000, live in `data/user/`, and are never loaded by evaluation or the pairs script.
- HW1 code is copied into `src/sparse/hw1_format.py`, not imported (`hw1_indexer.py` imports lxml/nltk). HW2 `weighting.py` is vendored unchanged under `tests/reference/` for the parity test only.
- Never stem or split tag terms. Multi-word labels are one `_`-joined term on both sides.

## Review Focus

1. Label names with punctuation (e.g. Open Images "Ball (Object)", or names containing commas) → normalized terms contain only `[a-z0-9_]`, so HW1's `", "` / `": <"` / `"; "` separators can't break a record (Task 2 test).
2. A query whose tags match no gallery term → `tags` returns zero results without an error, and `fusion` falls back to the CLIP ranking (Task 5 test).
3. Fusion when the gallery holds fewer images than `fusion.depth` or `k` → no duplicate ids, at most `ntotal` results (Task 5 test).
4. Demo restart after adding photos → every user photo comes back with the same id, and the next new photo gets a fresh id (Task 7 test).
5. A disallowed task × representation from the demo dropdown → a readable error message in the UI, not a crash (Task 7 test). The engine itself raises `ValueError` (Task 5 test).

---

### Task 1: HW1 posting format + BM25 TagIndex with HW2 parity

**Files:**
- Create: `src/sparse/__init__.py`, `src/sparse/hw1_format.py`, `src/sparse/tag_index.py`
- Create: `tests/reference/__init__.py`, `tests/reference/hw2_weighting.py` (HW2 `weighting.py` unchanged, with an attribution header comment above the original docstring)
- Modify: `configs/default.yaml`: add `bm25: {k1: 1.2, b: 0.75}`
- Modify: `.gitignore`: add `team27_hw*.zip` (the 120 MB homework archives stay local)
- Test: `tests/test_tag_index.py`

**Interfaces:**
- Produces:
  - `src/sparse/hw1_format.py`: `PostingEntry`, `compute_skip_pointers`, `format_posting_standard`, `parse_posting_line`, copied verbatim from `hw1_indexer.py` (source: scratchpad `hw/team27_hw2/team27_hw2/hw1_indexer.py` lines 703–833, or `team27_hw2.zip`), with a comment naming the source file and team.
  - `class TagIndex(k1: float, b: float)`:
    - `add(image_id: int, tags: list[tuple[str, int]]) -> None`: tags in descending confidence; positions are the token slots of the tf-repeated sequence; raises `ValueError` if the id already exists.
    - `TagIndex.from_docs(docs: dict[int, list[tuple[str, int]]], k1, b) -> TagIndex`
    - `.N -> int`, `.avgdl -> float`, `.doc_len(image_id) -> int`, `.df(term) -> int`, `.ids -> list[int]`, `.doc_terms(image_id) -> list[str]` (confidence order)
    - `.score(query_tf: dict[str, int]) -> dict[int, float]` (only docs sharing ≥ 1 term)
    - `.search(query_tf: dict[str, int], k: int) -> list[tuple[int, float]]` (score desc, id asc on ties)
    - `.save(directory: Path)` writes `dictionary.txt` (`term, df` lines) and `postings.txt` (HW1 single-line records via `format_posting_standard`), terms sorted, docs sorted by id; `TagIndex.load(directory, k1, b) -> TagIndex` via `parse_posting_line`. Doc term order is rebuilt from the positions.

- [ ] **Step 1:** tests:
  - `test_positions_are_token_slots`: tags `[("cat",3),("sofa",1)]` → postings cat positions `[0,1,2]`, sofa `[3]`, `doc_len == 4`.
  - `test_bm25_matches_hw2_reference`: 4 docs with ids 1–4 and mixed tf, 3 query terms with qtf 1–3. The reference = Σ qtf · `Scheme(stub,"bm25").doc_weight_fn(t)(d, tf)`, where the stub has `N`, `df(t)`, `meta={"avgdl": …}` and `index_dir` holding `doc_lengths.tsv` (`d\t|d|`). Every doc must be `math.isclose(ours, ref, rel_tol=1e-9)`.
  - `test_search_ties_by_id_and_k_clipped`
  - `test_save_load_roundtrip_same_scores` (also asserts the `dictionary.txt` first line is `cat, 2` for a two-doc fixture)
  - `test_records_parse_with_hw1_parser` (parse each `postings.txt` line with `parse_posting_line`; df and tf match)
  - `test_add_duplicate_id_raises`
  - `test_add_updates_N_and_avgdl`
- [ ] **Step 2:** `pytest tests/test_tag_index.py -v` → FAIL (ImportError).
- [ ] **Step 3:** implement.
- [ ] **Step 4:** → PASS.
- [ ] **Step 5:** commit `feat: tag inverted index in HW1 format with HW2 BM25`.

### Task 2: Vocabulary + zero-shot tagger

**Files:**
- Create: `src/tagging.py`
- Modify: `src/encoders.py`: add `ClipEncoder.encode_text(texts: list[str]) -> np.ndarray` (L2-normalized float32, via `open_clip.get_tokenizer(arch)`, batched by `embed.batch_size`)
- Modify: `configs/default.yaml`: add `tags: {vocabularies: [coco, openimages, places365], prompts: ["a photo of a {}."], top_n: 15, tf_max: 5, logit_scale: 100, variants: {tags: [], tags_nococo: [coco]}}`, where `variants` maps a cache name to the sources whose *only-from* terms are dropped. Also add `paths.vocab: data/vocab`.
- Test: `tests/test_tagging.py`

**Interfaces:**
- Produces:
  - `normalize_label(name: str) -> str`: lower-case; Places `/a/x/y` → `x_y` (drop the leading `/<letter>/`); every run of chars outside `[a-z0-9]` → `_`; strip `_`.
  - `@dataclass Vocabulary(terms: list[str], sources: dict[str, set[str]])` with `.prompt_label(term) -> str` (`_` → space) and `.without_only(source: str) -> Vocabulary` (drops terms whose sources == {source}).
  - `load_vocabulary(sources: list[str], coco_instances: Path, vocab_dir: Path) -> Vocabulary`. Readers: COCO `categories[].name`; Open Images `oidv7-class-descriptions-boxable.csv` (header `LabelName,DisplayName`, use DisplayName via `csv`); Places `categories_places365.txt` (`/a/airfield 0`). Terms are sorted, merged across sources.
  - `VOCAB_URLS = {"openimages": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv", "places365": "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"}`
  - `class Tagger(terms: list[str], text_emb: np.ndarray, top_n: int, tf_max: int, logit_scale: float)`: `.tag(image_vecs: np.ndarray) -> list[list[tuple[str, float, int]]]` = (term, softmax prob, tf) for the top_n in descending prob. `.save(dir)` writes `vocab.json` (terms) + `text_emb.npy`; `Tagger.load(dir, cfg) -> Tagger`.
  - `build_text_embeddings(encoder, vocab: Vocabulary, prompts: list[str]) -> np.ndarray`: mean over prompts, re-normalized.

- [ ] **Step 1:** tests:
  - `test_normalize_label`: `"Dining table"→"dining_table"`, `"/c/church/indoor"→"church_indoor"`, `"Ball (Object)"→"ball_object"`, `"Hot-dog, grilled"→"hot_dog_grilled"`, `"/a/airport_terminal"→"airport_terminal"`.
  - `test_load_vocabulary_merges_sources` (tiny fixture files; "cat" in coco and openimages → `sources["cat"] == {"coco","openimages"}`).
  - `test_without_only_drops_coco_only_terms`.
  - `test_tagger_top_n_and_tf` (orthogonal text_emb rows; the image vector equals row 0 → term 0 first with tf == tf_max; all tf in [1, tf_max]; probabilities descending; len == top_n).
  - `test_tagger_save_load_roundtrip`.
  - `@pytest.mark.models test_clip_text_unit_norm` (shape (2, 512), norms ≈ 1) in `tests/test_encoders.py`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `pytest tests/test_tagging.py tests/test_encoders.py -v` → PASS.
- [ ] **Step 5:** commit `feat: label vocabularies and CLIP zero-shot tagger`.

### Task 3: RRF and shared tags

**Files:** Create `src/rerank.py`; Test `tests/test_rerank.py`

**Interfaces:**
- Produces: `rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]` (fused score desc, id asc on ties; an id in only one list gets one term); `shared_tags(query_terms: list[str], doc_terms: list[str]) -> list[str]` (query order).

- [ ] **Step 1:** tests: `test_rrf_known_scores` (`[[1,2,3],[3,1]]`, k=60 → id1 = 1/61+1/62, id3 = 1/63+1/61, id2 = 1/62; order 1,3,2); `test_rrf_tie_by_id` (`[[5],[4]]` → order 4,5); `test_rrf_empty_list_is_identity_order`; `test_shared_tags_query_order`.
- [ ] Steps 2–4; commit `feat: reciprocal rank fusion`.

### Task 4: Tag gallery script (`03_tag.py`)

**Files:**
- Create: `scripts/03_tag.py`
- Modify: `src/tagging.py` (add `build_tag_cache`)
- Test: `tests/test_tag_script.py` (calls `build_tag_cache` directly)

**Interfaces:**
- Consumes: Task 1 `TagIndex`, Task 2 `load_vocabulary`, `Tagger`, `build_text_embeddings`, `ClipEncoder.encode_text`.
- Produces: `src/tagging.py:build_tag_cache(variant: str, vocab: Vocabulary, text_emb: np.ndarray, image_vecs: np.ndarray, ids: np.ndarray, cfg: dict, out_dir: Path) -> TagIndex`. It writes `out_dir/{vocab.json, text_emb.npy, tags.json, dictionary.txt, postings.txt}`, where `tags.json` maps image_id → [[term, prob, tf], …]. Script: downloads missing vocab files to `paths.vocab` (requests, timeout 30 s), builds the full vocabulary once, encodes text once, and for each `tags.variants` entry selects the subset (rows of text_emb for the kept terms) and writes `cache/<variant>/`. It prints term count, mean tags per image, and the top 10 terms by df.

- [ ] **Step 1:** test `test_build_tag_cache_writes_all_files` (3 fake images, 5-term vocab, orthogonal embeddings → all five files exist, `TagIndex.load(out).N == 3`, tags.json keys are the ids as strings).
- [ ] Steps 2–4; then run `python scripts/03_tag.py` → Expected: `cache/tags/` and `cache/tags_nococo/` written, about 1,000 terms for `tags` and fewer for `tags_nococo`, 2,000 docs each. Step 5: commit `feat: tag the gallery into HW1-format indexes`.

### Task 5: Engine — tag/fusion retrievers, allowed reps, add_image

**Files:**
- Modify: `src/schema.py` (`SearchResult.shared_tags: list[str] = field(default_factory=list)`, `SearchResponse.query_tags: list[str] = field(default_factory=list)`), `src/index.py` (`FlatIndex.add(vectors, ids)`), `src/pipeline.py`, `src/evaluate.py` (`_check_index_matches` uses `engine.gallery_ids(rep)`), `configs/default.yaml` (`tasks.semantic.representation: fusion`, `tasks.semantic.allowed`, `tasks.similar.allowed`, `fusion: {rrf_k: 60, depth: 100}`, `user: {dir: data/user, first_id: 1000000000}`)
- Modify tests: `tests/test_config_schema.py` (new values), `tests/test_pipeline.py::test_search_uses_task_default_representation` (assert the default comes from `cfg["tasks"][task]["representation"]` by setting it to `"clip"` in the test's cfg; semantic default `fusion` is asserted in config tests)
- Test: `tests/test_engine_tags.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `REPRESENTATIONS = ("clip", "dinov2", "phash", "tags", "fusion", "tags_nococo", "fusion_nococo")`; `tag_variant(rep) -> str | None` (`fusion`→`tags`, `fusion_nococo`→`tags_nococo`, `tags*`→itself).
  - `SearchEngine(cfg, encoders, indexes, load_encoders=False, taggers: dict[str, Tagger] | None = None, tag_indexes: dict[str, TagIndex] | None = None)`. Dense reps use `indexes`. `tags*` need `encoders["clip"]` + tagger + tag index. `fusion*` need those plus `indexes["clip"]`.
  - `.available() -> list[str]` (reps whose parts are loaded); `.gallery_ids(rep) -> set[int]`.
  - `.search(req)`:
    - The `(task, rep)` pair must be in `tasks.<task>.allowed`, else `ValueError` naming the allowed list. *Plan ruling: the spec puts this check on `SearchRequest`, but `SearchRequest` has no config, so it lives in `search()`.* An unavailable rep → `ValueError`.
    - The CLIP vector is computed once. `tags`: BM25 with qtf from the query's tags. `fusion`: `rrf([clip top depth ids, bm25 top depth ids], rrf_k)`, cut to `k`; score = fused score.
    - `shared_tags` is set for tags/fusion results; `query_tags` is the query's terms.
  - `.add_image(img: Image.Image, image_id: int | None = None) -> int`: id = given, or max(`user.first_id`, max existing id ≥ first_id + 1). Preprocess like the query path, encode with every loaded dense rep (`FlatIndex.add`), tag with every loaded tagger (`TagIndex.add`). Raises `ValueError` if the id exists.
  - `from_cache` also loads `cache/<variant>/` for each `tags.variants` key that has `postings.txt` (`Tagger.load` + `TagIndex.load`).

- [ ] **Step 1:** tests in `tests/test_engine_tags.py`, with a fake CLIP encoder (the MeanColorEncoder from `tests/test_pipeline.py`), a 3-term tagger whose text_emb are the unit R, G and B axes, and a 4-image gallery:
  - `test_tags_rep_returns_bm25_ranking_with_shared_tags`
  - `test_fusion_rrf_of_clip_and_tags` (scores equal `rrf` of the two lists)
  - `test_fusion_falls_back_to_clip_when_no_tag_overlap` (Review Focus 2; `tags` → 0 results, no error)
  - `test_fusion_small_gallery_no_duplicates` (k=50 → ≤ 4 unique ids; Review Focus 3)
  - `test_disallowed_task_rep_raises` (similar + tags → `ValueError` mentioning "allowed")
  - `test_add_image_searchable_in_every_rep` (a new solid-color image → rank 1 for clip and tags with id 1_000_000_000; the second add gets 1_000_000_001)
  - `test_add_image_explicit_id_and_duplicate_raises`
  - `test_query_tags_and_single_clip_encode` (counts encoder calls == 1 for fusion)
  - `test_flat_index_add` in `tests/test_preprocess_index.py`
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** `pytest -q -m "not models"` → all PASS (Week-1 tests included).
- [ ] **Step 5:** commit `feat: tag and fusion retrieval in the search engine, add_image`.

### Task 6: Evaluation over allowed reps + README leakage note

**Files:**
- Modify: `scripts/05_evaluate.py` (loop reps × tasks only where rep ∈ `tasks.<task>.allowed` ∩ `engine.available()`; the default `--representations` becomes every available rep), `README.md`, `README.zh-TW.md` (Evaluation section: the leakage note from the spec; Getting started: add `03_tag.py`, `app/app.py`, `06_pairs.py`; Status line)
- Test: `tests/test_evaluate.py::test_semantic_eval_runs_for_fusion` (engine from Task 5 fixtures → summary has `ndcg@10`, `representation == "fusion"`)

- [ ] Steps 1–4 as usual; then run `python scripts/05_evaluate.py` → Expected: `summary.csv` has 10 rows (7 semantic, 3 similar). Record nDCG@10/mAP for tags, fusion and the nococo pair.
- [ ] Step 5: commit `feat: evaluate tag and fusion lines; leakage note`.

### Task 7: Gradio demo

**Files:**
- Create: `src/demo.py` (UI-free logic), `app/__init__.py`, `app/app.py` (Gradio wiring only)
- Modify: `requirements.txt` (+`gradio`), `configs/default.yaml` (`demo: {host: 127.0.0.1, port: 7860, hard_queries: 20}`)
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes: Task 5 engine, Task 6 per-query CSVs, Week-1 manifests.
- Produces in `src/demo.py`:
  - `class UserGallery(root: Path)`: `.records() -> list[dict]` (`{"image_id","file_name"}` from `gallery.json`; empty if missing); `.save(img, image_id) -> Path` (`images/<id>.jpg`, appends the manifest via temp file + rename).
  - `class DemoState(engine, cfg, coco_files: dict[int, Path], user: UserGallery)`; `DemoState.create(cfg) -> DemoState` (from_cache + re-add user photos with their ids); `.path_of(image_id) -> Path`.
  - `search_handler(state, image, task: str, rep: str, k: int) -> tuple[list[tuple[str, str]], str, str]` = (gallery items `(path, "#rank · score · tag, tag")`, query-tags text, timings text). On `ValueError`, it returns `([], "", "⚠ <message>")` (Review Focus 5); `image is None` → a prompt message.
  - `add_handler(state, image) -> str` ("Added as #<id>").
  - `reps_for_task(cfg, available, task) -> list[str]` (dropdown choices).
  - `hard_queries(cfg, rep) -> list[list]` (`[qid, ap]` rows, lowest AP first, `demo.hard_queries` rows; empty if the CSV is missing); `query_image_path(cfg, qid) -> Path`.
  - `app/app.py`: `build_app(state) -> gr.Blocks` following the spec sketch (crop via `gr.ImageEditor` or equivalent; the task radio updates the rep dropdown via `reps_for_task`); `main()` launches on `demo.host`/`demo.port`.

- [ ] **Step 1:** tests with the Task 5 fake engine, using `tmp_path` as `user.dir`:
  - `test_search_handler_items_and_captions`
  - `test_search_handler_disallowed_rep_returns_message`
  - `test_add_then_restart_keeps_ids` (Review Focus 4: add 2 photos → ids 1e9, 1e9+1; a new DemoState from the same dir re-adds them with the same ids; the next add → 1e9+2)
  - `test_reps_for_task_filters`
  - `test_hard_queries_sorted`
  - `test_build_app_returns_blocks` (imports gradio, no launch)
- [ ] Steps 2–4; then launch `python app/app.py` in the background, open `http://127.0.0.1:7860` in the browser pane, run one semantic search on a val image, and add one photo → Expected: a grid of 10 thumbnails with shared tags; "Added as #1000000000". Then delete `data/user/` (test artifact). Step 5: commit `feat: Gradio demo with add-to-gallery and hard queries`.

### Task 8: Photo pairs

**Files:**
- Create: `src/pairs.py`, `scripts/06_pairs.py`
- Modify: `src/edits.py` (`apply_edit_chain(img, edits: list[str], params: dict) -> Image`), `configs/default.yaml` (`pairs: {dir: data/pairs, synthetic: 10, synthetic_edit: {edits: [crop, rotation, color, jpeg], crop_frac: 0.6, rotate_deg: 15, hue_shift: 0.08, saturation: 1.5, jpeg_quality: 20}}`)
- Test: `tests/test_pairs.py`

**Interfaces:**
- Produces:
  - `@dataclass Pair(name: str, gallery: Path, query: Path)`; `load_pairs(root: Path) -> list[Pair]` (subfolders containing exactly one `gallery.*` and one `query.*` image; others skipped with a warning).
  - `make_synthetic_pairs(records: list[dict], image_dir: Path, out_root: Path, cfg: dict) -> list[Pair]` (writes `synth_<id>/gallery.jpg` + `query.jpg`; skips existing).
  - `evaluate_pairs(engine, pairs, cfg) -> list[dict]`: add every gallery photo via `engine.add_image`, then for each task and allowed ∩ available rep search with k = `search.eval_depth`. Rows are `{"pair","task","representation","rank"}` (rank `None` if absent).
  - `summarize_pairs(rows) -> list[dict]`: `{"task","representation","n","at1","top10"}`.
  - Script: `--synthetic N` (default `pairs.synthetic`). It selects N val2017 records not in `queries_semantic.json` (`select_subset`, seed), downloads them with `download_images` into `data/coco/val2017`, makes the pairs, loads every pair under `pairs.dir`, builds the engine with `from_cache` (no user photos), writes `results/pairs.csv`, and prints the summary.

- [ ] **Step 1:** tests: `test_apply_edit_chain_order` (crop then rotation: size == crop size); `test_load_pairs_skips_incomplete`; `test_evaluate_pairs_partner_rank1` (fake engine; gallery photo = query photo → rank 1 for each allowed rep); `test_summarize_pairs_counts`.
- [ ] Steps 2–4; then run `python scripts/06_pairs.py` → Expected: 10 synthetic pairs; a summary printed per task × rep; `results/pairs.csv` written. Step 5: commit `feat: photo-pair check with synthetic pairs`, then commit the regenerated `results/summary.csv` and the README numbers as `docs: week-2 results`.
