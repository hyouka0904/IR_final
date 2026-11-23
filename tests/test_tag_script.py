import json

import numpy as np

from src.sparse.tag_index import TagIndex
from src.tagging import Tagger, Vocabulary, build_tag_cache

CFG = {"tags": {"top_n": 3, "tf_max": 5, "logit_scale": 100,
                "variants": {"tags": [], "tags_nococo": ["coco"]}},
       "bm25": {"k1": 1.2, "b": 0.75}}
TERMS = ["cat", "dog", "lamp", "sofa", "tree"]
SOURCES = {"cat": {"coco", "openimages"}, "dog": {"coco"}, "lamp": {"openimages"},
           "sofa": {"coco"}, "tree": {"places365"}}


def _inputs():
    vocab = Vocabulary(TERMS, SOURCES)
    text_emb = np.eye(5, dtype=np.float32)
    image_vecs = np.eye(5, dtype=np.float32)[[0, 1, 2]]
    return vocab, text_emb, image_vecs, np.array([11, 22, 33])


def test_build_tag_cache_writes_all_files(tmp_path):
    vocab, text_emb, image_vecs, ids = _inputs()
    out = tmp_path / "tags"
    idx = build_tag_cache("tags", vocab, text_emb, image_vecs, ids, CFG, out)
    for name in ("vocab.json", "text_emb.npy", "tags.json", "dictionary.txt", "postings.txt"):
        assert (out / name).exists(), name
    loaded = TagIndex.load(out, 1.2, 0.75)
    assert loaded.N == 3 == idx.N
    tags = json.loads((out / "tags.json").read_text(encoding="utf-8"))
    assert sorted(tags) == ["11", "22", "33"]
    assert tags["11"][0][0] == "cat" and tags["11"][0][2] == 5
    assert Tagger.load(out, CFG).terms == TERMS


def test_build_tag_cache_variant_drops_only_from_terms(tmp_path):
    vocab, text_emb, image_vecs, ids = _inputs()
    out = tmp_path / "tags_nococo"
    build_tag_cache("tags_nococo", vocab, text_emb, image_vecs, ids, CFG, out)
    tagger = Tagger.load(out, CFG)
    assert tagger.terms == ["cat", "lamp", "tree"]
    np.testing.assert_array_equal(tagger.text_emb, text_emb[[0, 2, 4]])
