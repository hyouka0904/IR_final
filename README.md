# Zero-Shot Image-to-Image Search

**English** | [繁體中文](README.zh-TW.md)

An image-to-image search engine built for an Information Retrieval course final project. You upload any photo you took yourself, and the system returns a ranked list from a gallery of about 40,000 everyday photos. It supports two tasks over the same gallery:

- **Semantic search.** Finds images of the *same kind of thing*. A photo of your cat returns other cats, even though your cat is not in the gallery.
- **Similar-image search.** Finds images that are *the same picture or scene*: cropped, recolored, compressed or re-photographed copies, and other shots of the same scene.

No model is trained or fine-tuned. Every component is pre-trained, so any photo can be used as a query (zero-shot).

> **Status:** all components are implemented (November 2026): CLIP, SigLIP, DINOv2 and pHash; FAISS Flat, HW2 cluster pruning, IVF-PQ and HNSW; CLIP zero-shot tags in an HW1-format inverted index with HW2 BM25 and RRF fusion; alpha-QE, Rocchio and MMR; synthetic edits, INRIA Holidays, box queries, phone-photo judging with Cohen's kappa and the ROxford/RParis run; the Gradio demo, the photo-pair check, a data integrity check and CI on Windows and Linux. The 40k-image runs and the phone-photo judging are still to be done (see *Roadmap*).

## Why two tasks

The same query has different relevant results depending on the task. For a photo of a cat on a sofa, semantic search counts every cat as relevant, while similar-image search counts only copies of that photo. The project's core question is:

> **Which image representation suits which retrieval task?**

Both tasks share one gallery, one indexing layer and one evaluation framework. Only the representation and the definition of relevance change.

## How it works

```mermaid
flowchart LR
    subgraph Offline
        G[Gallery images] --> C[CLIP / SigLIP embedding]
        G --> D[DINOv2 embedding]
        G --> H[pHash]
        C --> T[CLIP zero-shot tags]
        C --> FC[(FAISS index)]
        D --> FD[(FAISS index)]
        T --> II[(Inverted index)]
    end
    subgraph Online
        Q[Query photo] --> P[EXIF fix, resize, optional crop]
        P --> S{Task}
        S -->|Semantic| R1["CLIP kNN + tag BM25, RRF fusion"]
        S -->|Similar| R2[DINOv2 kNN / pHash Hamming]
        R1 --> RR[Re-rank: Rocchio, alpha-QE, MMR]
        R2 --> OUT[Ranked results]
        RR --> OUT
    end
    FC -.-> R1
    II -.-> R1
    FD -.-> R2
```

| Component | Choice | Notes |
|---|---|---|
| Semantic representation | CLIP (OpenCLIP ViT-B/32), SigLIP | Trained with text supervision, so similarity follows human category concepts; SigLIP is a second semantic model to compare against CLIP |
| Visual representation | DINOv2 | Image-only self-supervised features, sensitive to appearance and layout |
| Classic baseline | pHash (64-bit) | Perceptual hash compared by Hamming distance; handles light edits only |
| Sparse line | CLIP zero-shot tags → inverted index → BM25 | Top-15 concept tags per image act as a "document". Labels come from fixed external class lists (COCO, Open Images, Places), not learned from the gallery |
| Fusion | Reciprocal Rank Fusion | Combines dense and sparse rankings; shared tags explain why a result matched |
| Feedback | Rocchio (user-marked), alpha-QE (automatic) | Rocchio moves the query vector toward marked results; alpha-QE is pseudo-relevance feedback |
| Diversity | MMR | Covers different readings of an ambiguous query ("the cat or the sofa?") |
| Indexing | FAISS Flat, cluster pruning (≈ IVF), IVF-PQ, HNSW | Measured for recall loss, latency and index size; IVF-PQ adds vector compression |

## Datasets

Both tasks share the main gallery:

- **Gallery:** a 40,000-image subset of MS-COCO train2017.
- **Semantic queries:** 1,000 val2017 images, none of which are in the gallery. Graded relevance is the share of COCO object categories a result has in common with the query.
- **Similar queries:**
  - 1,000 gallery images with synthetic edits (crop, rotation, color shift, JPEG, blur). The ground truth is the original image.
  - INRIA Holidays photos merged into the gallery, for real same-scene matches.
- **User study:** about 100 phone photos, judged by pooling (see Evaluation).
- **External benchmark (supplementary):** [ROxford5k / RParis6k](https://github.com/filipradenovic/revisitop) under the official revisitop protocol, for the similar-image task only. They are not merged into the shared gallery; they let the similar-image results be compared with published work.

Candidate open datasets, ranked by fit for this project (availability checked 2026-11-18):

| # | Dataset | Size | Labels | Tasks | Access / license |
|---|---|---|---|---|---|
| 1 | [MS-COCO 2017](https://cocodataset.org/) ([HF mirror](https://huggingface.co/datasets/detection-datasets/coco)) | train 117k, val 4.95k | 80 object classes, multi-label | Semantic + similar (synthetic) | Direct download; annotations CC BY 4.0 |
| 2 | [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) (subset) | val 41,620 (full 9.18M) | 20,638 image-level classes | Semantic + similar (synthetic) | `downloader.py` or FiftyOne; images mostly CC BY 2.0 |
| 3 | [Unsplash Lite](https://github.com/unsplash/datasets) | ~25k | Keywords only | Demo; similar (synthetic) | ~700 MB; commercial and non-commercial use, no redistribution of images |
| 4 | [INRIA Holidays](https://thoth.inrialpes.fr/~jegou/data.php.html) | 1,491 images, 500 queries | Same-scene groups | Similar (real) | FTP links listed; not yet tested |
| 5 | [INRIA Copydays](https://thoth.inrialpes.fr/~jegou/data.php.html) | Originals + crop/JPEG/strong attacks | Copy pairs | Similar (real edits) | HTTP archives listed |
| 6 | [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) | 30,607 images | 256 classes, single-label | Semantic + similar (synthetic) | 1.2 GB direct download; CC BY 4.0 |
| 7 | [Places365](https://github.com/CSAILVision/places365) | 365 scene classes | Scene labels | Semantic (scenes) | Tarball on data.csail.mit.edu; main site unreachable when checked |
| 8 | [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) (HF mirror) | 31,014 images | 5 captions each, no classes | Semantic (caption-derived) | 4.4 GB; non-commercial research |
| 9 | [California-ND](https://sites.google.com/site/vonikakis/datasets) | 701 personal photos | Near-duplicate probabilities from 10 annotators | Similar (real) | Encrypted zip, password by email |
| 10 | [DISC21](https://ai.meta.com/datasets/disc21-dataset/) | 1M references + 50k queries | Copy pairs | Similar (real edits) | ~350 GB; license not stated on page |

## Evaluation

Runs are written in TREC format and scored with `trec_eval`.

| Task | Metrics |
|---|---|
| Semantic search | nDCG@10, mAP (graded relevance from shared COCO categories) |
| Similar-image search | Recall@1, Recall@10, MRR, reported per edit type |
| Phone-photo queries | Pool the top 10 of every system; two annotators grade 0/1/2; nDCG@10 and Cohen's kappa |
| Efficiency | Median latency per stage (embedding, search, re-rank), index size, ANN recall loss against Flat. Search is always timed on CPU so that HNSW, which FAISS runs only on CPU, is compared fairly |
| Per-query diagnostics | AP for every query and configuration; a hard-query list sorted by AP, used for failure analysis and shown in the demo |

**Experiments** (how to run them: *Getting started*)

1. A cross table of 4 representations (CLIP, SigLIP, DINOv2, pHash) × 2 tasks: `05_evaluate.py`.
2. Semantic ablation: CLIP only, tag BM25 only, RRF fusion, then fusion with alpha-QE and with MMR: `05_evaluate.py --experiments` (configurations in `eval.experiments`).
3. Robustness of DINOv2 against pHash for each edit type: the `*_crop`, `*_rotation`, … columns of the `similar` rows.
4. Flat vs. cluster pruning vs. IVF-PQ vs. HNSW: the `@cluster`, `@ivfpq`, `@hnsw` rows with `recall_loss@10`, `index_mb` and latency.
5. Intent disambiguation: `box` (query cropped to one object) against `box_full` (the same query uncropped), with and without MMR.
6. Real same-scene matches: the `holidays` rows (mAP, Recall@1/10, MRR).
7. Supplementary: the similar-image task on ROxford/RParis (Easy / Medium / Hard mAP): `08_roxford.py`.

A configuration is labelled `representation[@index][+reranker…]`, e.g. `fusion`, `clip@hnsw` or `fusion+aqe+mmr`. Rocchio is scored with a simulated user: the set's own relevance judges the top 10 of the first ranking, the query is re-run with that feedback, and the judged results stay where they were (frozen ranking), so only what Rocchio adds below them counts. Its latency includes both rounds.

**Results so far** (2,000-image gallery, 200 semantic queries, before Holidays was merged into the gallery; `results/summary.csv`)

| Semantic | nDCG@10 | mAP |
|---|---|---|
| CLIP kNN | 0.652 | 0.236 |
| DINOv2 kNN | 0.684 | 0.274 |
| Tag BM25 | 0.656 | 0.279 |
| RRF fusion (CLIP + tags) | 0.676 | 0.277 |
| Tag BM25, no COCO labels | 0.653 | 0.276 |
| RRF fusion, no COCO labels | 0.675 | 0.275 |

**Photo pairs** (`scripts/06_pairs.py`, `results/pairs.csv`). 10 synthetic pairs for now (a COCO val image joins the gallery; its crop 0.6 + rotation + color + JPEG edit is the query). Partner at rank 1 / in the top 10: CLIP 9/10 · 10/10, DINOv2 9/10 · 10/10, fusion 7/10 · 10/10, tag BM25 4/10 · 7/10, pHash 0/10 · 0/10. Real phone-photo pairs go in `data/pairs/<name>/` and run the same way.

**Leakage note.** The semantic ground truth comes from COCO categories, so having the 80 COCO class names in the tag vocabulary gives the tag line a head start. The `_nococo` rows drop every term that comes only from COCO and re-tag the gallery over the smaller vocabulary (971 → 955 terms). Only 16 of the 80 COCO names go; the other 64 survive word for word through Open Images or Places365. So the ablation removes only a small part of the head start, and the small gap it shows (tags nDCG@10 0.656 → 0.653) is a lower bound on the leakage, not a measure of it.

## Project layout

```
.
├── configs/default.yaml      # every parameter: models, K, fusion, indexes, re-ranking, sources
├── data/                     # [not in git] COCO subset, Holidays, ROxford/RParis, phone photos
├── cache/                    # [not in git] embeddings, hashes, tags, indexes
├── src/
│   ├── encoders.py           # CLIP / SigLIP / DINOv2 / pHash behind one interface
│   ├── tagging.py            # CLIP zero-shot tags over a fixed label vocabulary
│   ├── sparse/               # inverted index + BM25 (reused from HW1/HW2)
│   ├── index.py              # FAISS Flat / IVF-PQ / HNSW, HW2 cluster pruning
│   ├── rerank.py             # RRF, alpha-QE, Rocchio, MMR
│   ├── edits.py              # synthetic edits for similar-image queries
│   ├── data.py               # COCO subset, Holidays, box queries, manifests
│   ├── download.py           # resumable downloads that check and repair data/
│   ├── integrity.py          # data and cache checks (scripts/check_data.py)
│   ├── schema.py             # SearchRequest / SearchResult, agreed by both members
│   ├── pipeline.py           # online search, shared by evaluation and demo
│   ├── evaluate.py           # every query set, configurations, Rocchio protocol
│   ├── revisited.py          # ROxford / RParis under the revisitop protocol
│   └── demo.py               # demo logic (app/app.py is the UI)
├── scripts/                  # numbered offline steps 01-08, check_data.py, search.py, makezip.py
├── app/                      # web demo
├── tests/                    # unit tests, smoke test (fake data), real-model tests (-m models)
└── results/
    ├── summary.csv           # metrics, staged latency, index size, recall loss per set and configuration
    ├── per_query/            # AP per query per configuration; source of the hard-query list
    ├── runs/                 # TREC runs and qrels
    ├── phone/                # judging sheet, kappa, qrels of the phone photos
    └── roxford/              # supplementary ROxford / RParis results
```

### Engineering conventions

- All parameters (models, K, fusion and feedback weights, index settings) live in `configs/default.yaml`, never hard-coded.
- Evaluation scripts and the demo call the same `src/pipeline.py`, so the demo shows exactly what was evaluated.
- `src/schema.py` fixes the data passed between the two members' modules.
- Latency is recorded in three stages, with search always on CPU.

## Getting started

The default config runs on a small slice: 2,000 gallery images, 200 semantic queries, 200 similar queries, plus the 1,491 INRIA Holidays images. Raise `data.*` in `configs/default.yaml` toward 40,000 / 1,000 / 1,000 to scale up. A GPU is needed only for embedding (02, 03, 08); search, evaluation and the demo run on CPU.

### 1. Environment

Python 3.11 or newer. Install PyTorch first, matching your GPU driver, then the rest:

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)
source .venv/bin/activate         # Linux / macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # see below
pip install -r requirements.txt
python -c "import torch, faiss, open_clip; print(torch.__version__, torch.cuda.is_available())"
```

| `CUDA Version` shown by `nvidia-smi` | `--index-url` ending |
|---|---|
| 12.8 or newer (13.x included) | `cu128` |
| 12.6, 12.7 | `cu126` |
| 12.4, 12.5 | `cu124`, with `"torch==2.6.*" "torchvision==0.21.*"` |
| older | update the driver first (`cu121` stops at torch 2.5.1, too old for open_clip) |
| no NVIDIA GPU | leave out `--index-url` (CPU build) |

With conda, create the environment from conda-forge so no Anaconda terms of service are needed: `conda create -n zsis -c conda-forge --override-channels python=3.12`.

### 2. Run the steps

```bash
python scripts/01_download.py --check-only --with-roxford   # what is complete, partial, broken
python scripts/01_download.py            # annotations, COCO subset, Holidays; resumes and repairs
python scripts/check_data.py             # data integrity (again after 02-04: caches vs gallery)
python scripts/02_embed.py --models clip siglip dinov2 phash
python scripts/03_tag.py                 # CLIP zero-shot tags -> cache/tags*/ (HW1 postings format)
python scripts/04_build_index.py --index flat cluster ivfpq hnsw
python scripts/05_evaluate.py            # cross table on every query set
python scripts/05_evaluate.py --experiments   # ablations, re-ranking, index comparison
python scripts/06_pairs.py               # photo-pair check -> results/pairs.csv
python scripts/07_judging.py pool        # phone photos: pool -> judge -> kappa -> qrels (below)
python scripts/01_download.py --with-roxford && python scripts/08_roxford.py   # supplementary
python scripts/search.py data/holidays/jpg/100000.jpg --task similar --index hnsw --rerank aqe mmr
python app/app.py                        # web demo on http://127.0.0.1:7860
```

- **01 checks before it downloads.** Complete files are kept, an interrupted download (`.part`) resumes, an archive smaller than the remote file (left by an older run) resumes too, a larger or unreadable one is fetched again, and broken images (empty, unreadable, truncated) are deleted and downloaded or extracted again. Rerun it after any interruption. `--check-only` reports without changing anything; `--skip-holidays` leaves Holidays out.
- **Holidays is on FTP**, which some networks block. 01 gives up after 60 seconds without data and carries on without it (the `holidays` set is then skipped). Put a mirror in `sources.holidays`, or fetch `jpg1.tar.gz` / `jpg2.tar.gz` elsewhere into `data/holidays/` and rerun 01.
- **05 options:** `--configs fusion+aqe clip@hnsw …` runs chosen configurations, `--tasks semantic box …` chosen query sets. Search is timed on one CPU thread.
- **Phone photos (user study):** put about 100 photos in `data/phone/` (convert HEIC first). `07_judging.py pool` writes `results/phone/judgments.csv` (shuffled, no system names) and `judgments.html` to view the images; each judge fills `judge_a` / `judge_b` with 0 / 1 / 2 (Excel is fine). Then `07_judging.py kappa`, `07_judging.py qrels` and `05_evaluate.py --tasks phone`.
- To re-score the TREC runs with `trec_eval`, pass `-l 2` (grades ≥ 2 are exactly the images counted as relevant for mAP), e.g. `trec_eval -l 2 -m map -m ndcg_cut.10 results/runs/semantic.qrels results/runs/semantic_clip.run`. trec_eval's nDCG uses the integer grades 0–3, so it differs slightly from the float-gain nDCG in `summary.csv`.

### Windows

Everything is Python; no bash, WSL, curl, wget, unzip or tar is needed. Use `/` in paths so the same commands work on both systems.

- Keep the repo on an ASCII-only path outside OneDrive (FAISS can't write indexes to non-ASCII paths, and the data plus caches exceed 10 GB), e.g. `C:\work\ir_image_search`.
- Set `PYTHONUTF8=1` so Python reads and writes UTF-8 instead of the system code page: `setx PYTHONUTF8 1` (new terminals), or with conda `conda env config vars set PYTHONUTF8=1`.
- Open the demo at `http://127.0.0.1:7860`; `0.0.0.0` is a listening address, not one a browser can open.

### On a Linux server, reachable from the LAN or a VPN

```bash
tmux new -s demo                                  # keeps running after you log out (Ctrl+B, D)
python app/app.py --host 0.0.0.0 --auth team27:<password>
hostname -I                                       # the server's addresses
sudo ufw allow from 192.168.0.0/16 to any port 7860 proto tcp   # if ufw is on; adjust the range
```

Other machines open `http://<server address>:7860` (the VPN interface's address over a VPN, e.g. Tailscale's `100.x.x.x`). The demo has no login of its own, so set `--auth` (or `IR_DEMO_AUTH=user:password`) whenever it is reachable from other machines. `--port` changes the port, `--share` adds a temporary public Gradio link. Run the long steps (01–05) inside tmux too.

### Tests and CI

```bash
python -m pytest -m "not models and not smoke"   # unit tests, a few seconds
python -m pytest -m smoke                         # scripts 01-07 end to end on fake data
python -m pytest -m models                        # real CLIP / SigLIP / DINOv2 weights
```

The smoke test serves a tiny fake COCO / Holidays set from a local HTTP server and replaces the models with a fake encoder, so it needs no network, GPU or weights; it also runs the data integrity check. GitHub Actions (`.github/workflows/ci.yml`) runs the syntax check, the unit tests and the smoke test on Ubuntu and Windows for every push and pull request, installing only `requirements-ci.txt` (no torch).

### Running from the zip

`python scripts/makezip.py` packs the project into `dist/ir_image_search-<date>.zip`. The zip holds the code, the gallery images (COCO and Holidays), the val images, the manifests, the label lists, the photo pairs, the embedding/tag/index caches and the per-query results, so nothing has to be downloaded or re-embedded. `--code-only` packs just the git files instead. Not included: `.venv`, demo photos in `data/user/`, and `instances_train2017.json`, which only `01_download.py` needs.

On the receiving PC:

```bash
cd ir_image_search
python -m venv .venv && .venv/Scripts/activate   # Windows; use .venv/bin/activate elsewhere
pip install -r requirements.txt
python -m pytest -m "not models"       # should all pass
python app/app.py                      # demo on http://127.0.0.1:7860
python scripts/05_evaluate.py          # reproduces results/summary.csv
```

The first search downloads the model weights from Hugging Face (about 1.5 GB with SigLIP, once).

## Demo

- Upload a photo, optionally crop it, and switch between **Semantic** and **Similar** search.
- Pick the **representation**, the **index** (flat or a built approximate index) and **re-ranking** (alpha-QE, MMR).
- Results appear as a thumbnail grid. Semantic results show the tags they share with the query; the timings line shows the embed / search / re-rank stages.
- **Rocchio feedback:** mark results as relevant or not relevant and re-run the query.
- **Compare** runs a second representation side by side on the same photo.
- **Add this photo to the gallery** (the photo-pair demo): it is searchable at once and kept across restarts, but never enters the evaluation.
- A **hard queries** panel lists the evaluation queries with the lowest AP; a loaded one marks its results ✓ / ✗ by the evaluation's relevance, for inspecting failures.
- The **🌓 Light / dark** button switches the theme (remembered by the browser; the OS setting decides until you choose).

## Roadmap

- [x] Week 1: download COCO; CLIP and DINOv2 embeddings, pHash, FAISS Flat; both tasks searchable from a script
- [x] Week 2: CLIP tags into the inverted index, BM25, RRF fusion; synthetic edit queries; web demo
- [x] Week 3 (code): Rocchio, alpha-QE, MMR; cluster pruning, IVF-PQ and HNSW; SigLIP; evaluation of every query set; judging scripts
- [ ] Week 3 (data): take and judge the phone photos
- [ ] Week 4: full 40k experiments, ROxford/RParis supplementary run, hard-query failure analysis, report and demo

## Team

| Member | Responsibilities |
|---|---|
| 陳長壻 | Semantic search: CLIP tags, inverted index and BM25 over tags, RRF fusion, Rocchio and alpha-QE, MMR; semantic evaluation |
| 鄧存正 | Similar-image search: DINOv2 and pHash, synthetic edits; FAISS and cluster pruning; web demo; similarity and efficiency evaluation |

Shared: data preprocessing, `src/pipeline.py` and `src/schema.py`, phone-photo judging, final report.

## Acknowledgements

- [ImgSearch](https://github.com/seamile/ImgSearch), the reference project, which showed that CLIP embeddings plus nearest-neighbor search work on a single machine.
- [OpenCLIP](https://github.com/mlfoundations/open_clip), [DINOv2](https://github.com/facebookresearch/dinov2), [FAISS](https://github.com/facebookresearch/faiss), [ImageHash](https://github.com/JohannesBuchner/imagehash).
- Ranking, inexact top-k and evaluation methods follow Manning, Raghavan and Schütze, *Introduction to Information Retrieval*.

## License

To be decided. Datasets keep their own licenses (see the table above); the images are not redistributed in this repository.
