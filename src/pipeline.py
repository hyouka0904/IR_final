"""線上檢索流程（README 3.2），評估腳本與 demo 共用同一份。

config_id = {method}-{index}-{post}
  method : clip | siglip | dinov2 | phash | bm25 | {clip|siglip|dinov2}+bm25
  index  : flat | hnsw | ivfpq | cp（向量部分）；bm25 單獨使用時為 inv；phash 只有 flat
  post   : none，或 aqe / rocchio / mmr 以 + 串接（依此順序執行），只能用在向量模型

延遲分三段：
  embed  ：抽 query 特徵（向量、pHash、CLIP tag）；評估時改用快取特徵，此段由 03 腳本另外量測
  search ：索引搜尋、BM25、RRF 融合
  rerank ：alpha-QE、Rocchio（擴展 query 後重搜）、MMR
"""
import os

import numpy as np
from PIL import Image

from src import features
from src.data.sets import load_set
from src.eval.metrics import StageTimer
from src.index import faiss_index
from src.models.names import VECTOR_MODELS
from src.preprocess.crop import crop_user_box
from src.rerank.aqe import alpha_qe
from src.rerank.mmr import mmr
from src.rerank.rocchio import rocchio
from src.rerank.rrf import rrf
from src.schema import SearchRequest, SearchResult
from src.text.bm25 import BM25
from src.text.inverted_index import InvertedIndex

POSTS = ("aqe", "rocchio", "mmr")
METHODS = (*VECTOR_MODELS, "phash", "bm25", *(f"{m}+bm25" for m in VECTOR_MODELS))


def parse_config_id(config_id: str) -> dict:
    try:
        method, index, post = config_id.split("-")
    except ValueError:
        raise ValueError(f"Invalid config_id: {config_id}") from None
    parts = method.split("+")
    vec = next((p for p in parts if p != "bm25"), None)
    text = "bm25" in parts
    steps = [] if post == "none" else post.split("+")
    ok = (method in METHODS
          and (index == "inv" if vec is None else index in faiss_index.INDEX_KINDS)
          and (vec != "phash" or (index == "flat" and not steps))
          and all(s in POSTS for s in steps) and len(set(steps)) == len(steps)
          and steps == sorted(steps, key=POSTS.index)
          and (not steps or vec in VECTOR_MODELS))
    if not ok:
        raise ValueError(f"Invalid config_id: {config_id}")
    return {"method": method, "vec": vec, "text": text, "index": index, "post": post,
            "aqe": "aqe" in steps, "rocchio": "rocchio" in steps, "mmr": "mmr" in steps}


def without_post(config_id: str, step: str) -> str:
    """去掉某個後處理步驟，例如 clip-flat-aqe+rocchio → clip-flat-aqe（Rocchio 的初次排序）。"""
    method, index, post = config_id.split("-")
    steps = [s for s in post.split("+") if s not in (step, "none")]
    return f"{method}-{index}-{'+'.join(steps) or 'none'}"


def text_index_path(cache_root: str, gallery: str) -> str:
    return os.path.join(cache_root, "text", gallery, "tags_index.pkl")


class Pipeline:
    """單一圖庫的檢索流程。特徵、索引、encoder 都在第一次用到時才載入。"""

    def __init__(self, gallery: str, cfg: dict, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.cache_root = cfg["paths"]["cache_root"]
        self.gallery = load_set(self.cache_root, gallery)
        self.name = gallery
        self._db, self._indexes, self._encoders = {}, {}, {}
        self._bm25 = self._vocab = self._text_embs = self._tags = None

    # ---------- 延遲載入 ----------

    def db_features(self, rep: str) -> np.ndarray:
        if rep not in self._db:
            self._db[rep] = np.load(features.feature_path(self.cache_root, self.name, rep))
        return self._db[rep]

    def index(self, rep: str, kind: str):
        key = (rep, kind)
        if key not in self._indexes:
            path = faiss_index.index_file(self.cache_root, self.name, rep, kind)
            self._indexes[key] = faiss_index.load_index(path, kind, self.cfg, binary=rep == "phash")
        return self._indexes[key]

    def encoder(self, model: str):
        if model not in self._encoders:
            from src.models.encoders import load_encoder
            self._encoders[model] = load_encoder(model, self.cfg, self.device)
        return self._encoders[model]

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            c = self.cfg["bm25"]
            self._bm25 = BM25(InvertedIndex.load(text_index_path(self.cache_root, self.name)), c["k1"], c["b"], c["k3"])
        return self._bm25

    @property
    def vocab(self) -> list[str]:
        if self._vocab is None:
            from src.models.tagger import load_vocab
            self._vocab = load_vocab(self.cfg["tags"]["vocab"])
        return self._vocab

    @property
    def gallery_tags(self) -> features.TagSet:
        if self._tags is None:
            self._tags = features.TagSet.load(self.cache_root, self.name, self.vocab, self.cfg["tags"])
        return self._tags

    def text_embs(self) -> np.ndarray:
        if self._text_embs is None:
            from src.models import tagger
            tcfg = self.cfg["tags"]
            path = features.text_emb_path(self.cache_root, tcfg)
            if os.path.exists(path):
                self._text_embs = np.load(path)
            else:
                self._text_embs = tagger.text_embeddings(self.encoder(tcfg["model"]), self.vocab, tcfg["templates"])
        return self._text_embs

    # ---------- query 特徵 ----------

    def prepare_query(self, request: SearchRequest) -> Image.Image:
        if request.image is None:
            raise ValueError("SearchRequest needs an image (or cached query features)")
        image = request.image.convert("RGB")
        return crop_user_box(image, request.user_box) if request.user_box is not None else image

    def query_features(self, image: Image.Image, c: dict) -> dict:
        """{rep: 向量或 pHash, "tags": (top, probs)}，只算這組設定用得到的。"""
        from src.models import phash, tagger
        q = {}
        if c["vec"] == "phash":
            q["phash"] = phash.phash_bits(image, self.cfg["models"]["phash"]["hash_size"])
        elif c["vec"] is not None:
            q[c["vec"]] = self.encoder(c["vec"]).encode([image])[0]
        if c["text"]:
            tcfg = self.cfg["tags"]
            enc = self.encoder(tcfg["model"])
            emb = q[tcfg["model"]] if tcfg["model"] in q else enc.encode([image])[0]
            top, probs = tagger.tag_images(emb[None], self.text_embs(), tcfg["top_n"], tagger.logit_scale(enc))
            q["tags"] = (top[0], probs[0])
        return q

    def tag_terms(self, tags) -> dict[str, int]:
        from src.models.tagger import tag_terms
        tcfg = self.cfg["tags"]
        return tag_terms(tags[0], tags[1], self.vocab, tcfg["tf_mode"], tcfg["tf_scale"])

    # ---------- 流程 ----------

    def search(self, request: SearchRequest, query_feats: dict | None = None) -> SearchResult:
        """query_feats：評估時傳入快取的 query 特徵，省略 embed 階段。"""
        c = parse_config_id(request.config_id)
        k = request.k or self.cfg["retrieval"]["k"]
        exclude = np.asarray(request.exclude, dtype=np.int64)
        timer = StageTimer()
        image = None
        if query_feats is None:
            image = self.prepare_query(request)
            with timer.stage("embed"):
                query_feats = self.query_features(image, c)

        # 多取 len(exclude) 名，過濾後仍有 k 名
        depth = min(k + len(exclude), len(self.gallery))
        keep = lambda ids, s: (ids[(ids >= 0) & ~np.isin(ids, exclude)], s[(ids >= 0) & ~np.isin(ids, exclude)])
        rankings, vec_ids, vec_scores, extra = [], None, None, {}

        # 1. 向量檢索（含 alpha-QE、Rocchio 的擴展 query 重搜）
        if c["vec"] is not None:
            rep = c["vec"]
            q = query_feats[rep]
            idx = self.index(rep, c["index"])
            n_qe = self.cfg["aqe"]["n"]
            with timer.stage("search"):
                scores, ids = faiss_index.search(idx, q, max(depth, n_qe + len(exclude)))
            vec_ids, vec_scores = keep(ids[0], scores[0])
            if c["aqe"]:
                with timer.stage("rerank"):
                    a = self.cfg["aqe"]
                    q = alpha_qe(q, self.db_features(rep), vec_ids, a["n"], a["alpha"])
                    scores, ids = faiss_index.search(idx, q, depth)
                    vec_ids, vec_scores = keep(ids[0], scores[0])
            if c["rocchio"]:
                if request.feedback is not None:
                    with timer.stage("rerank"):
                        r = self.cfg["rocchio"]
                        q = rocchio(q, self.db_features(rep), *request.feedback, r["alpha"], r["beta"], r["gamma"])
                        scores, ids = faiss_index.search(idx, q, depth)
                        vec_ids, vec_scores = keep(ids[0], scores[0])
                else:
                    extra["rocchio"] = "skipped: no feedback"
            rankings.append(vec_ids)
            extra["query_vector"] = q

        # 2. tag BM25
        query_tags = None
        if c["text"]:
            with timer.stage("search"):
                terms = self.tag_terms(query_feats["tags"])
                txt_ids, txt_scores = keep(*self.bm25.search(terms, depth))
            rankings.append(txt_ids)
            top, probs = query_feats["tags"]
            query_tags = [(self.vocab[i], float(p)) for i, p in zip(top, probs)]
            extra["query_terms"] = terms

        # 3. 融合
        if c["vec"] is not None and c["text"]:
            with timer.stage("search"):
                ids, scores = rrf(rankings, self.cfg["rrf"]["k"])
        elif c["text"]:
            ids, scores = txt_ids, txt_scores
        else:
            ids, scores = vec_ids, vec_scores

        # 4. MMR（用向量模型的 embedding 衡量結果之間的相似度）
        if c["mmr"]:
            with timer.stage("rerank"):
                m = self.cfg["mmr"]
                ids, scores = mmr(ids, scores, self.db_features(c["vec"]), m["lambda"], m["depth"])

        return SearchResult(request.config_id, ids[:k], np.asarray(scores[:k], dtype=np.float32), timer.as_dict(),
                            query_image=image, query_tags=query_tags, extra=extra)

    def matched_tags(self, query_terms: dict[str, int], db_idx: int) -> list[str]:
        """explainability：query 與某張圖庫圖片共有的 tag。"""
        return [t.replace("_", " ") for t in self.bm25.index.matched_terms(query_terms, db_idx)]
