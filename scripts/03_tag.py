"""Step 3: tag every gallery image with CLIP zero-shot labels and build the tag indexes.

    python scripts/03_tag.py
Needs cache/clip/vectors.npy (02_embed.py). Downloads the Open Images / Places365 label lists
into paths.vocab once, then writes cache/<variant>/ for every tags.variants entry:
vocab.json, text_emb.npy, tags.json, dictionary.txt, postings.txt (HW1 format).
"""
import time

import _bootstrap  # noqa: F401
import numpy as np
import requests

from src.config import load_config, resolve_path
from src.encoders import get_encoder
from src.tagging import (VOCAB_FILES, VOCAB_URLS, build_tag_cache, build_text_embeddings,
                         load_vocabulary)


def fetch_vocab_files(sources: list[str], vocab_dir, urls: dict | None = None) -> None:
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        if source not in VOCAB_URLS:
            continue
        path = vocab_dir / VOCAB_FILES[source]
        if path.exists():
            continue
        r = requests.get((urls or {}).get(source, VOCAB_URLS[source]), timeout=30)
        r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        print(f"downloaded {source} labels -> {path}")


def main():
    cfg = load_config()
    sources = cfg["tags"]["vocabularies"]
    vocab_dir = resolve_path(cfg, "vocab")
    fetch_vocab_files(sources, vocab_dir, cfg.get("sources"))
    coco_instances = resolve_path(cfg, "data") / "annotations" / "instances_val2017.json"
    vocab = load_vocabulary(sources, coco_instances, vocab_dir)
    counts = {s: sum(s in v for v in vocab.sources.values()) for s in sources}
    print(f"vocabulary: {len(vocab.terms)} terms {counts}")

    t0 = time.perf_counter()
    text_emb = build_text_embeddings(get_encoder("clip", cfg), vocab, cfg["tags"]["prompts"])
    print(f"text embeddings {text_emb.shape} in {time.perf_counter() - t0:.1f}s")

    cache = resolve_path(cfg, "cache")
    image_vecs = np.load(cache / "clip" / "vectors.npy")
    ids = np.load(cache / "clip" / "ids.npy")
    for variant in cfg["tags"]["variants"]:
        t0 = time.perf_counter()
        index = build_tag_cache(variant, vocab, text_emb, image_vecs, ids, cfg, cache / variant)
        top = sorted(index.postings, key=lambda t: (-index.df(t), t))[:10]
        print(f"{variant}: {index.N} images, {len(index.postings)} terms used, "
              f"mean tags/image {sum(len(index.doc_terms(i)) for i in index.ids) / index.N:.1f}, "
              f"avgdl {index.avgdl:.1f}, {time.perf_counter() - t0:.1f}s -> {cache / variant}")
        print("  top df: " + ", ".join(f"{t}({index.df(t)})" for t in top))


if __name__ == "__main__":
    main()
