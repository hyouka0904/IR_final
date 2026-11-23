# Week 2 design: CLIP tags + BM25 + RRF fusion, web demo, photo pairs

Status: draft for approval (2026-11-20). Builds on the Week-1 MVP now on `main`.

## Intent

This is the "explainable" half of semantic search from README.md and the proposal:

> CLIP's top-15 concept tags per image form a "document" for our HW1 inverted index and HW2 BM25, fused with vectors by RRF.

Dense CLIP kNN finds similar images but can't say why. Tags give a second, sparse ranking, and the tags a result shares with the query explain the match. RRF fuses the two. The experiment question stays the same as the project's: which representation suits which task. This week adds the semantic ablation "CLIP only / tag BM25 only / RRF fusion".

The other half is the demo the course will see on 2026-12-23. The proposal adds a requirement the README's demo section lacks:

> Demo: 5–10 same-subject photo pairs; one joins the gallery, the other is the query, to confirm new images are reliably retrieved.

So the demo must be able to add a photo to the gallery, and the pairs need a measured result, not just a live click-through.

## Goals

1. **Zero-shot tagging.** Each gallery image gets its top-15 labels from fixed external label lists: COCO 80, Open Images V7 boxable (601), Places365. The score is CLIP image–text cosine with a prompt template. The tags are computed from the cached CLIP gallery vectors (`cache/clip/vectors.npy`), so nothing is re-embedded.
2. **Sparse retrieval.** An in-memory inverted index over the tag "documents" with Okapi BM25. Its scores must equal HW2's `weighting.Scheme('bm25')` exactly (choice C, picked by the user). The same formula and constants are pinned by a test that runs the HW2 code itself as the reference.
3. **RRF fusion.** Combines the CLIP kNN ranking and the tag BM25 ranking. Semantic results carry the tags they share with the query.
4. **Evaluation.** New semantic rows in the cross table: `tags`, `fusion`, plus the no-COCO-vocabulary ablations `tags_nococo` and `fusion_nococo` (see Constraints, leakage).
5. **Web demo** (Gradio):
   - Upload a photo and optionally crop it.
   - Choose Semantic or Similar.
   - See a thumbnail grid with scores and shared tags.
   - See the query's own tags.
   - Add the photo to the gallery.
   - A hard-queries panel.
6. **Photo pairs.** A script measures, for every pair, the rank at which the query photo retrieves its partner after the partner joins the gallery, for both tasks.

## Non-goals (this week)

- Rocchio, alpha-QE and MMR (Week 3); the demo's "mark relevant" button comes with Rocchio.
- Cluster pruning, IVF-PQ and HNSW (Week 3).
- SigLIP, INRIA Holidays, ROxford and phone-photo judging.
- Importing the HW1/HW2 modules at runtime. `hw1_indexer.py` pulls in `lxml` and `nltk` on import. The HW1 pieces we reuse are copied (see below), and HW2 `weighting.py` is used only as a test reference.
- Tags or fusion for the similar-image task. Tags describe categories, not copies.

## Anti-goals (failure even if it "works")

- **The demo ranks differently from what was evaluated.** The demo must call the same `SearchEngine.search`.
- **User-added demo photos leak into evaluation.** Evaluation must see only the COCO gallery. The Week-1 index/gallery id check must keep passing.
- **BM25 that silently differs from HW2's.** The parity test is the guard.
- **Tag terms broken apart.** A multi-word label such as "dining table" must stay one term (`dining_table`) on both the gallery and the query side.

## Constraints

- All parameters live in `configs/default.yaml` (Week-1 convention).
- Search on CPU; latency in three stages (`embed`, `search`, `rerank`). A CLIP query embedding is computed once per request and shared by the CLIP, tags and fusion lines.
- **Leakage.** The semantic ground truth comes from COCO categories, so the COCO-80 label list gives the tag line a head start. Open Images also covers most COCO concepts, so the no-COCO ablation reduces the head start without removing it. The results and report must say this.
- **Label-list downloads** (small text files):
  - COCO names come from the annotations we already have.
  - Open Images: `oidv7-class-descriptions-boxable.csv` from storage.googleapis.com/openimages/v7/.
  - Places365: `categories_places365.txt` from the CSAILVision/places365 GitHub repo.
- Gradio is a new dependency (available for Python 3.14).

## Approaches considered (sparse line)

| | Approach | Trade-off |
|---|---|---|
| A | Run HW1 builder + HW2 prep `--no-warc` + HW2 `Ranker` unchanged | Most reuse. HW2's query side runs the HW1 tokenizer/stemmer on query words, which would split or stem tag terms that the index stores whole, so matches would be silently lost. It also adds NLTK and untested Python 3.14 compatibility. |
| B | HW1 builder + HW2 `IndexReader` + HW2 `Scheme('bm25')`, with a thin tag ranker | Real HW code on the critical path, but keeps the on-disk format and the prep step. |
| **C (chosen)** | In-memory index with HW2's exact BM25 (parity test against HW2 `weighting.py`), stored in HW1's postings format | Simple and fast, and supports adding images live for the demo. The index on disk is HW1's Slide-5 format, read and written by HW1's own posting functions, so it can be inspected like HW1 output. BM25 reuse is at the formula level, made verifiable by the parity test. |

## How it works (decided parts)

**Label vocabulary.**
- Labels are lower-cased; spaces and hyphens become `_`. Duplicates across lists merge into one term, and the term records its source lists. Places365 names drop the leading letter folder: `/a/airport_terminal` → `airport_terminal`, `/c/church/indoor` → `church_indoor`. In prompts, `_` is shown as a space.
- `tags.vocabularies` selects which lists are used. The no-COCO variant drops terms that come *only* from COCO, and re-tags every image over that smaller vocabulary (its softmax differs), into its own cache.

**Tagging.**
- Text embeddings use the prompt `tags.prompt` ("a photo of a {}."), averaged when several templates are configured.
- Tag scores are the softmax of 100 × cosine over the vocabulary. The top `tags.top_n` = 15 are kept.
- Term frequency follows confidence: `tf = 1 + round((tf_max − 1) · p / p_top1)` with `tf_max` = 5. The top tag therefore gets tf 5 and weaker tags get 1–4.
- Document length = Σ tf. The query image is tagged the same way, and its tf values are the query term frequencies.

**BM25.** HW2's formula and defaults: score(q, d) = Σ_t qtf(t) · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)), with idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5)), k1 = 1.2, b = 0.75. N and avgdl are taken over all indexed images, and they are updated when a photo is added.

*Parity (my call):* "equal" means equal within `math.isclose(rel_tol=1e-9)`, not bit-for-bit. The test guards the formula and constants, and the scorer stays free to use numpy.

**RRF.**
- Each line contributes 1/(k + rank) with `fusion.rrf_k` = 60. Each line is cut at `fusion.depth` = 100 candidates.
- An image found by only one line still gets that line's contribution.
- Ties are broken by image id, ascending.

**Shared tags.** For semantic results from `tags` or `fusion`: the intersection of the query's and the result's tag sets, ordered by the query tag's score.

**Representations.**
- `clip`, `dinov2`, `phash` (Week 1), plus `tags`, `fusion`, `tags_nococo` and `fusion_nococo`.
- `tasks.semantic.representation` becomes `fusion`; the similar task stays `dinov2`.
- *Which representation runs on which task (my call):*
  - Semantic: `clip`, `dinov2`, `phash`, `tags`, `fusion`, `tags_nococo`, `fusion_nococo`.
  - Similar: `clip`, `dinov2`, `phash` only.
  - `tasks.<task>.allowed` in the config lists these. `05_evaluate.py`, the pairs script and the demo's dropdown follow it. `SearchRequest` with a disallowed pair raises `ValueError`.
  - Tagging the query counts toward the `embed` stage; BM25 and RRF count toward `search`.
- Caches, written by `scripts/03_tag.py`, one folder per variant (`cache/tags/`, `cache/tags_nococo/`):
  - `tags.json`: image_id → [[term, score, tf], …], for explanations and the demo.
  - `dictionary.txt` + `postings.txt`: the tag inverted index in HW1's format (below).
  - The BM25 statistics (N, |d|, avgdl) are derived from the postings on load. Loading takes milliseconds at 2k and well under a second at 40k.

**HW1 postings format (reused).**
- Each image's tag "document" is the token sequence formed by writing each tag tf times, tags in descending confidence order. A tag's positions are its token slots in that sequence. |d| = Σ tf = the number of positions.
- `postings.txt` has one record per term in HW1's Slide-5 single-line form, terms sorted, docs sorted by image id: `term, df: <doc, tf: <p1, p2, …>; …>`.
- `dictionary.txt` follows HW1's dictionary layout: one `term, df` line per term, in the same sorted order as `postings.txt` (e.g. `cat, 412`), as `hw1_indexer.py` writes it with `dict_include_df=True`.
- Reused from HW1, copied unchanged into `src/sparse/hw1_format.py` with an attribution comment: `PostingEntry`, `format_posting_standard`, `parse_posting_line`. These use only the standard library. `compute_skip_pointers` comes along because the parser calls it; skips aren't used for scoring.
- Not reused, and why: WARC/HTML parsing, tokenizer, stopwords and stemming (tags are clean fixed terms, and stemming would break `dining_table`); SPIMI block merging (the index fits in RAM); skip pointers, phrase and proximity queries (BM25 is OR scoring over full postings, and tag order carries no meaning); the vocabulary trie (no prefix lookup needed).
- Adding a demo photo appends its postings in memory. The on-disk files are rewritten only by `03_tag.py`, so evaluation always reads the COCO-only index.
- A test round-trips the index through the files and checks the BM25 scores are unchanged; another checks that a record written by our code parses with the copied HW1 parser.

**Demo** (`app/app.py`, Gradio, `127.0.0.1:7860` by default):

```
┌ Query ─────────────────────┐ ┌ Results ──────────────────────────────────────┐
│ [upload / crop photo]      │ │ ┌────┐┌────┐┌────┐┌────┐┌────┐                 │
│ Task: (•) Semantic ( ) Sim │ │ │ #1 ││ #2 ││ #3 ││ #4 ││ #5 │  … top k        │
│ Representation: [fusion ▾] │ │ └────┘└────┘└────┘└────┘└────┘                 │
│ k: [10]   [Search]         │ │ 0.83 · cat, sofa  (score · shared tags)        │
│ Query tags: cat, sofa, …   │ │ timings: embed 110 ms · search 1 ms · rerank 0 │
│ [Add this photo to gallery]│ └───────────────────────────────────────────────┘
└────────────────────────────┘ ┌ Hard queries (lowest AP) ─────────────────────┐
                               │ 12345  AP 0.02  [load]   …                     │
                               └───────────────────────────────────────────────┘
```

- **Adding a photo:**
  - The photo is stored under `data/user/images/` and listed in `data/user/gallery.json`.
  - It gets an id ≥ 1,000,000,000, so it can't clash with COCO ids.
  - It is embedded with every representation and tagged, and appended to the in-memory indexes.
  - On the next demo start, user photos are re-embedded and re-added. The evaluation script never loads them.
- **Hard-queries panel:** the 20 lowest-AP rows from `results/per_query/semantic_<rep>.csv`. Clicking one loads that val image as the query.

**Photo pairs** (`scripts/06_pairs.py`):
- Layout: `data/pairs/<pair_name>/gallery.<ext>` and `query.<ext>`.
- The script uses the COCO gallery plus the pair photos only. It never loads `data/user/` (*my call*: demo additions shouldn't change a measured result).
- It adds every gallery photo in memory without persisting it, then runs each query through each task's allowed representations.
- *Real pairs come later (user's decision, 2026-11-20).* This week the script also generates synthetic pairs: `--synthetic N` (default 10) picks N COCO val2017 images that are not semantic queries. Each one's original becomes `data/pairs/synth_<id>/gallery.jpg`, and a combined strong edit (crop 0.6 + rotation + color + JPEG, from `pairs.synthetic_edit` in config) becomes `query.jpg`. Real pairs dropped in later as `data/pairs/<name>/` run the same way.
- Output: `results/pairs.csv` with the partner's rank (empty if beyond `search.eval_depth`), plus a printed summary of how many partners are at rank 1 and in the top 10, per task and representation.

**Engine refactor.**
- The engine maps each representation to a retriever: dense (encoder + `FlatIndex`), tags (CLIP encoder + tagger + BM25 index), or fusion (dense CLIP + tags).
- `SearchEngine.add_image(img) -> int` adds one image to every loaded representation.
- `SearchResult` gains `shared_tags: list[str]` (empty by default). `SearchResponse` gains `query_tags: list[str]`.

## Results and leakage note

`README.md` (Evaluation section, both languages) states the COCO-vocabulary head start next to the tag/fusion numbers and points to the `_nococo` rows.

## Left to the builder

- Module layout inside `src/` (e.g. `src/tagging.py`, `src/sparse/`, `src/rerank.py`), within the README's planned layout.
- Gradio component choices and styling, within the sketch above.
- How the label lists are fetched and cached (`scripts/03_tag.py` or a helper). They should be downloaded once into `data/vocab/`.
- How the HW2 `weighting.py` reference gets into the tests. Vendor it under `tests/reference/` with attribution, unchanged.
