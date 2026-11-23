import json
import re

import numpy as np
import pytest

from src.tagging import Tagger, Vocabulary, load_vocabulary, normalize_label


@pytest.mark.parametrize("raw,term", [
    ("Dining table", "dining_table"),
    ("/c/church/indoor", "church_indoor"),
    ("Ball (Object)", "ball_object"),
    ("Hot-dog, grilled", "hot_dog_grilled"),
    ("/a/airport_terminal", "airport_terminal"),
])
def test_normalize_label(raw, term):
    assert normalize_label(raw) == term
    assert re.fullmatch(r"[a-z0-9_]+", term)


def _vocab_files(tmp_path):
    coco = tmp_path / "instances.json"
    coco.write_text(json.dumps({"categories": [{"id": 1, "name": "cat"},
                                               {"id": 2, "name": "dining table"}]}))
    vdir = tmp_path / "vocab"
    vdir.mkdir()
    (vdir / "oidv7-class-descriptions-boxable.csv").write_text(
        'LabelName,DisplayName\n/m/01,Cat\n/m/02,"Ball (Object)"\n/m/03,"Hot-dog, grilled"\n',
        encoding="utf-8")
    (vdir / "categories_places365.txt").write_text("/a/airfield 0\n/c/church/indoor 1\n",
                                                  encoding="utf-8")
    return coco, vdir


def test_load_vocabulary_merges_sources(tmp_path):
    coco, vdir = _vocab_files(tmp_path)
    v = load_vocabulary(["coco", "openimages", "places365"], coco, vdir)
    assert v.terms == sorted(["cat", "dining_table", "ball_object", "hot_dog_grilled",
                              "airfield", "church_indoor"])
    assert v.sources["cat"] == {"coco", "openimages"}
    assert v.sources["church_indoor"] == {"places365"}
    assert v.prompt_label("dining_table") == "dining table"


def test_without_only_drops_coco_only_terms(tmp_path):
    coco, vdir = _vocab_files(tmp_path)
    v = load_vocabulary(["coco", "openimages", "places365"], coco, vdir).without_only("coco")
    assert "dining_table" not in v.terms
    assert "cat" in v.terms and v.sources["cat"] == {"coco", "openimages"}
    assert set(v.sources) == set(v.terms)


def _tagger(top_n=3, tf_max=5):
    terms = ["a", "b", "c", "d"]
    return Tagger(terms, np.eye(4, dtype=np.float32), top_n=top_n, tf_max=tf_max, logit_scale=100)


def test_tagger_top_n_and_tf():
    img = np.array([[1.0, 0.0, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0]], dtype=np.float32)
    out = _tagger().tag(img)
    assert len(out) == 2 and all(len(t) == 3 for t in out)
    assert out[0][0][0] == "a" and out[0][0][2] == 5
    assert out[1][0][0] == "b" and out[1][0][2] == 5
    for tags in out:
        probs = [p for _, p, _ in tags]
        assert probs == sorted(probs, reverse=True)
        assert all(1 <= tf <= 5 for _, _, tf in tags)
    # a distant second gets tf 1
    assert 1 <= out[1][1][2] < 5 and out[1][1][0] == "a"


def test_tagger_save_load_roundtrip(tmp_path):
    t = _tagger()
    t.save(tmp_path)
    cfg = {"tags": {"top_n": 3, "tf_max": 5, "logit_scale": 100}}
    loaded = Tagger.load(tmp_path, cfg)
    assert loaded.terms == t.terms
    v = np.array([[0.2, 0.9, 0.1, 0.3]], dtype=np.float32)
    assert loaded.tag(v) == t.tag(v)


def test_tagger_settings_travel_with_the_cache(tmp_path):
    _tagger().save(tmp_path)
    settings = json.loads((tmp_path / "tagger.json").read_text(encoding="utf-8"))
    assert settings == {"top_n": 3, "tf_max": 5, "logit_scale": 100.0}
    changed = {"tags": {"top_n": 15, "tf_max": 5, "logit_scale": 100}}
    with pytest.raises(ValueError, match="03_tag"):
        Tagger.load(tmp_path, changed)


def test_tagger_load_cache_without_settings_uses_config(tmp_path):
    _tagger().save(tmp_path)
    (tmp_path / "tagger.json").unlink()  # caches written before the settings file existed
    loaded = Tagger.load(tmp_path, {"tags": {"top_n": 2, "tf_max": 4, "logit_scale": 50}})
    assert (loaded.top_n, loaded.tf_max, loaded.logit_scale) == (2, 4, 50.0)
