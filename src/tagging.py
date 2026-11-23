"""Label vocabularies (COCO, Open Images V7 boxable, Places365) and CLIP zero-shot tagging."""
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VOCAB_URLS = {
    "openimages": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "places365": "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt",
}
VOCAB_FILES = {"openimages": "oidv7-class-descriptions-boxable.csv",
               "places365": "categories_places365.txt"}


def normalize_label(name: str) -> str:
    """One index term per label: lower-case, Places '/a/x/y' -> 'x_y', other runs of chars -> '_'."""
    s = re.sub(r"^/[a-z]/", "", name.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


@dataclass
class Vocabulary:
    terms: list[str]
    sources: dict[str, set[str]]

    def prompt_label(self, term: str) -> str:
        return term.replace("_", " ")

    def without_only(self, source: str) -> "Vocabulary":
        """Drop the terms that come only from `source`."""
        keep = [t for t in self.terms if self.sources[t] != {source}]
        return Vocabulary(keep, {t: set(self.sources[t]) for t in keep})


def _read_source(source: str, coco_instances: Path, vocab_dir: Path) -> list[str]:
    if source == "coco":
        with open(coco_instances, encoding="utf-8") as f:
            return [c["name"] for c in json.load(f)["categories"]]
    path = Path(vocab_dir) / VOCAB_FILES[source]
    if source == "openimages":
        with open(path, encoding="utf-8", newline="") as f:
            return [row["DisplayName"] for row in csv.DictReader(f)]
    if source == "places365":
        with open(path, encoding="utf-8") as f:
            return [line.split()[0] for line in f if line.strip()]
    raise ValueError(f"unknown vocabulary source {source!r}")


def load_vocabulary(sources: list[str], coco_instances: Path, vocab_dir: Path) -> Vocabulary:
    merged: dict[str, set[str]] = {}
    for source in sources:
        for name in _read_source(source, coco_instances, vocab_dir):
            term = normalize_label(name)
            if term:
                merged.setdefault(term, set()).add(source)
    return Vocabulary(sorted(merged), merged)


def build_text_embeddings(encoder, vocab: Vocabulary, prompts: list[str]) -> np.ndarray:
    """One unit vector per term: the mean over prompt templates, re-normalized."""
    total = None
    for prompt in prompts:
        emb = encoder.encode_text([prompt.format(vocab.prompt_label(t)) for t in vocab.terms])
        total = emb if total is None else total + emb
    total = total / np.linalg.norm(total, axis=1, keepdims=True)
    return total.astype(np.float32)


class Tagger:
    def __init__(self, terms: list[str], text_emb: np.ndarray, top_n: int, tf_max: int,
                 logit_scale: float):
        if len(terms) != len(text_emb):
            raise ValueError("terms and text embeddings differ in length")
        self.terms = list(terms)
        self.text_emb = np.asarray(text_emb, dtype=np.float32)
        self.top_n, self.tf_max, self.logit_scale = int(top_n), int(tf_max), float(logit_scale)

    def tag(self, image_vecs: np.ndarray) -> list[list[tuple[str, float, int]]]:
        """(term, softmax prob, tf) for the top_n terms of each image, by descending prob."""
        logits = self.logit_scale * (np.asarray(image_vecs, dtype=np.float32) @ self.text_emb.T)
        logits = logits.astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        n = min(self.top_n, len(self.terms))
        out = []
        for row in probs:
            top = sorted(range(len(row)), key=lambda j: (-row[j], j))[:n]
            p1 = row[top[0]]
            out.append([(self.terms[j], float(row[j]),
                         1 + int(round((self.tf_max - 1) * row[j] / p1))) for j in top])
        return out

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_bytes(json.dumps(self.terms).encode("utf-8"))
        np.save(directory / "text_emb.npy", self.text_emb)
        (directory / "tagger.json").write_bytes(json.dumps(self.settings()).encode("utf-8"))

    def settings(self) -> dict:
        return {"top_n": self.top_n, "tf_max": self.tf_max, "logit_scale": self.logit_scale}

    @classmethod
    def load(cls, directory: Path, cfg: dict) -> "Tagger":
        """Queries must be tagged with the settings the gallery was tagged with: if the cache
        records them and the config now differs, raise instead of silently mismatching."""
        directory = Path(directory)
        terms = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        t = cfg["tags"]
        wanted = {"top_n": int(t["top_n"]), "tf_max": int(t["tf_max"]),
                  "logit_scale": float(t["logit_scale"])}
        path = directory / "tagger.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached != wanted:
                raise ValueError(f"{directory} was tagged with {cached} but configs/default.yaml "
                                 f"now says {wanted}; rerun scripts/03_tag.py")
        return cls(terms, np.load(directory / "text_emb.npy"), wanted["top_n"],
                   wanted["tf_max"], wanted["logit_scale"])


def variant_vocabulary(variant: str, vocab: Vocabulary, cfg: dict) -> Vocabulary:
    """The vocabulary of one `tags.variants` entry: drop terms coming only from the listed sources."""
    for source in cfg["tags"]["variants"][variant]:
        vocab = vocab.without_only(source)
    return vocab


def build_tag_cache(variant: str, vocab: Vocabulary, text_emb: np.ndarray, image_vecs: np.ndarray,
                    ids: np.ndarray, cfg: dict, out_dir: Path):
    """Tag every gallery image over the variant's vocabulary; write tagger, tags.json and the
    HW1-format index into out_dir. `vocab`/`text_emb` are the full vocabulary and its embeddings."""
    from src.sparse.tag_index import TagIndex

    sub = variant_vocabulary(variant, vocab, cfg)
    row = {t: i for i, t in enumerate(vocab.terms)}
    t = cfg["tags"]
    tagger = Tagger(sub.terms, text_emb[[row[x] for x in sub.terms]], t["top_n"], t["tf_max"],
                    t["logit_scale"])
    tags = tagger.tag(image_vecs)
    out_dir = Path(out_dir)
    tagger.save(out_dir)
    doc_tags = {int(i): tg for i, tg in zip(ids, tags)}
    (out_dir / "tags.json").write_bytes(json.dumps(
        {str(i): [[term, round(p, 6), tf] for term, p, tf in tg] for i, tg in doc_tags.items()}
    ).encode("utf-8"))
    index = TagIndex.from_docs({i: [(term, tf) for term, _, tf in tg] for i, tg in doc_tags.items()},
                               cfg["bm25"]["k1"], cfg["bm25"]["b"])
    index.save(out_dir)
    return index
