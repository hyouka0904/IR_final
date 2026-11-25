import pytest
from PIL import Image

from src.config import load_config
from src.schema import TASKS, SearchRequest, SearchResponse, SearchResult


def test_load_config_has_required_keys():
    cfg = load_config()
    assert set(cfg["paths"]) >= {"data", "cache", "results"}
    assert cfg["data"] == {"gallery_size": 40000, "semantic_queries": 1000,
                           "similar_queries": 1000, "seed": 27}
    assert cfg["models"]["clip"] == {"arch": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"}
    assert cfg["models"]["dinov2"] == {"timm_name": "vit_small_patch14_dinov2.lvd142m",
                                       "image_size": 224}
    assert cfg["models"]["phash"] == {"hash_size": 8}
    assert cfg["models"]["device"] == "auto"
    assert cfg["embed"]["batch_size"] == 32
    assert cfg["preprocess"]["max_side"] == 512
    assert cfg["tasks"]["semantic"]["representation"] == "fusion"
    assert cfg["tasks"]["semantic"]["allowed"] == ["clip", "siglip", "dinov2", "phash", "tags",
                                                   "fusion", "tags_nococo", "fusion_nococo"]
    assert cfg["tasks"]["similar"]["representation"] == "dinov2"
    assert cfg["tasks"]["similar"]["allowed"] == ["clip", "siglip", "dinov2", "phash"]
    assert cfg["models"]["siglip"] == {"arch": "ViT-B-16-SigLIP", "pretrained": "webli"}
    assert set(cfg["index"]) == {"cluster", "ivfpq", "hnsw"}
    assert set(cfg["rerank"]) == {"aqe", "rocchio", "mmr"}
    assert cfg["fusion"] == {"rrf_k": 60, "depth": 100}
    assert cfg["user"] == {"dir": "data/user", "first_id": 1000000000}
    assert cfg["bm25"] == {"k1": 1.2, "b": 0.75}
    assert cfg["paths"]["vocab"] == "data/vocab"
    assert cfg["tags"] == {"vocabularies": ["coco", "openimages", "places365"],
                           "prompts": ["a photo of a {}."], "top_n": 15, "tf_max": 5,
                           "logit_scale": 100,
                           "variants": {"tags": [], "tags_nococo": ["coco"]}}
    assert cfg["search"] == {"k": 10, "eval_depth": 100}
    assert cfg["eval"]["semantic_binary_threshold"] == 0.5
    assert cfg["edits"] == {"crop_frac": 0.7, "rotate_deg": 15, "hue_shift": 0.08,
                            "saturation": 1.5, "jpeg_quality": 20, "blur_radius": 2.0}


def test_search_request_rejects_bad_task():
    with pytest.raises(ValueError):
        SearchRequest(image=Image.new("RGB", (4, 4)), task="keyword")


def test_search_request_rejects_k_zero():
    with pytest.raises(ValueError):
        SearchRequest(image=Image.new("RGB", (4, 4)), task="semantic", k=0)


def test_schema_types_construct():
    req = SearchRequest(image=Image.new("RGB", (4, 4)), task="similar", k=5)
    assert req.task in TASKS and req.representation is None and req.crop is None
    resp = SearchResponse(results=[SearchResult(image_id=1, score=0.5, rank=1)],
                          representation="dinov2", timings_ms={"embed": 1.0})
    assert resp.results[0].rank == 1


def test_search_request_index_and_rerank():
    img = Image.new("RGB", (4, 4))
    req = SearchRequest(image=img, task="semantic", index="hnsw", rerank=("mmr", "aqe"))
    assert req.rerank == ("aqe", "mmr")  # normalized to the fixed order
    with pytest.raises(ValueError):
        SearchRequest(image=img, task="semantic", index="lsh")
    with pytest.raises(ValueError):
        SearchRequest(image=img, task="semantic", rerank=("bogus",))
