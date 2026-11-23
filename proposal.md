# Zero-Shot Image-to-Image Search Engine

(Transcribed from `team27_proposal.pdf`.)

## Introduction

We build a zero-shot image-to-image search engine: a user uploads any photo, and the system ranks a gallery of about 40,000 everyday photos for two tasks. **Semantic search** finds the same kind of thing (your cat returns other cats). **Similar-image search** finds the same picture or scene (crops, edits, re-shots). Keyword search fails here: photos carry no text, and the two tasks define relevance differently. Building on ImgSearch, which has one notion of similarity and no evaluation, we ask: *which image representation suits which task?*

## Methods

- **Two notions of relevance:** semantic search uses CLIP and SigLIP (close to human categories); similar search uses DINOv2 (image-only), with pHash as the classic baseline. All are compared on both tasks.
- **Zero-shot:** pre-trained models only; tags come from a fixed external class list.
- **Explainability:** CLIP's top-15 concept tags per image form a "document" for our HW1 inverted index and HW2 BM25, fused with vectors by RRF.
- **Ambiguous intent** (the cat or the sofa?): box cropping, Rocchio, alpha-QE, MMR.
- **Robustness:** DINOv2 vs. pHash per edit type (crop, rotation, color, JPEG, blur).
- **Speed:** FAISS Flat vs. HW2 cluster pruning, IVF-PQ and HNSW (recall, latency, size).
- **Data:** a shared 40k MS-COCO 2017 gallery. Semantic queries: 1,000 val images, graded by shared categories. Similar queries: 1,000 edited gallery images plus INRIA Holidays.
- **Evaluation:** trec_eval (nDCG@10, mAP; Recall@1/10, MRR); pooled human judgments on 100 phone photos (kappa); per-query AP for failure analysis; supplementary ROxford run.
- **Engineering:** evaluation and demo share one pipeline.
- **Demo:** 5–10 same-subject photo pairs; one joins the gallery, the other is the query, to confirm new images are reliably retrieved.

## Team members

- **陳長壻 (115598095):** semantic search: tags + BM25, RRF, Rocchio, alpha-QE, MMR, evaluation.
- **鄧存正 (115598063):** similar search: DINOv2, pHash, edits, FAISS, demo, evaluation.
- **Shared:** data preparation, common pipeline, human judging, final report.

Preferred time slots: 2026/12/23
