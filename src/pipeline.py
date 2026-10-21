"""線上檢索流程（README 3.2），評估腳本與 demo 共用同一份。

延遲分三段：
- embed：抽 query embedding（評估時改用快取的 embedding，此段由 02 腳本另外量測）
- search：初次索引搜尋，加上 ensemble 的聯集、重算與融合
- rerank：alpha-QE（擴展 query 與第二次搜尋）＋ 幾何驗證
"""
import itertools
import os

import numpy as np
from PIL import Image

from src.data import revisited
from src.index import faiss_index
from src.models import ensemble
from src.models.encoders import MODEL_NAMES
from src.preprocess.crop import crop_bbx, crop_user_box
from src.preprocess.multiscale import extract_multiscale, scales_for
from src.rerank import geometric
from src.rerank.aqe import alpha_qe
from src.eval.metrics import StageTimer
from src.schema import SearchRequest, SearchResult

SCALES = ("ss", "ms")
RERANKS = ("none", "aqe", "gv", "aqe+gv")


def parse_config_id(config_id: str) -> dict:
    model, scale, index, rerank = config_id.split("-")
    if model not in (*MODEL_NAMES, "ens") or scale not in SCALES or index not in faiss_index.INDEX_KINDS or rerank not in RERANKS:
        raise ValueError(f"Invalid config_id: {config_id}")
    return {
        "model": model,
        "models": list(MODEL_NAMES) if model == "ens" else [model],
        "scale": scale,
        "index": index,
        "rerank": rerank,
        "aqe": "aqe" in rerank,
        "gv": "gv" in rerank,
    }


def all_config_ids() -> list[str]:
    """4 × 2 × 3 × 4 = 96 組。"""
    return ["-".join(c) for c in itertools.product((*MODEL_NAMES, "ens"), SCALES, faiss_index.INDEX_KINDS, RERANKS)]


def embedding_path(cache_root: str, dataset: str, model: str, scale: str, split: str) -> str:
    """split: db | q | q_nocrop"""
    return os.path.join(cache_root, "embeddings", dataset, f"{model}_{scale}_{split}.npy")


def index_path(cache_root: str, dataset: str, model: str, scale: str, kind: str) -> str:
    return os.path.join(cache_root, "indexes", dataset, f"{model}_{scale}_{kind}.index")


def sift_path(cache_root: str, dataset: str) -> str:
    return os.path.join(cache_root, "sift", dataset, "db_sift.h5")


def inlier_cache_path(cache_root: str, dataset: str) -> str:
    return os.path.join(cache_root, "inliers", dataset, "inliers.h5")


class Pipeline:
    """單一資料集的檢索流程。embedding、索引、SIFT、encoder 都在第一次用到時才載入。"""

    def __init__(self, dataset: str, cfg: dict, device: str = "cpu"):
        self.dataset = dataset
        self.cfg = cfg
        self.device = device
        self.data_root = cfg["paths"]["data_root"]
        self.cache_root = cfg["paths"]["cache_root"]
        self.gnd = revisited.load_gnd(dataset, self.data_root)
        self.db_paths = revisited.db_image_paths(self.gnd, dataset, self.data_root)
        self.q_paths = revisited.query_image_paths(self.gnd, dataset, self.data_root)
        self._db_embs, self._indexes, self._encoders = {}, {}, {}
        self._db_sift = None
        self._inlier_cache = None

    # ---------- 延遲載入 ----------

    def db_embs(self, model: str, scale: str) -> np.ndarray:
        key = (model, scale)
        if key not in self._db_embs:
            self._db_embs[key] = np.load(embedding_path(self.cache_root, self.dataset, model, scale, "db"))
        return self._db_embs[key]

    def index(self, model: str, scale: str, kind: str):
        key = (model, scale, kind)
        if key not in self._indexes:
            self._indexes[key] = faiss_index.load_index(index_path(self.cache_root, self.dataset, *key), kind, self.cfg)
        return self._indexes[key]

    def encoder(self, model: str):
        if model not in self._encoders:
            from src.models.encoders import load_encoder
            self._encoders[model] = load_encoder(model, self.cfg, self.device)
        return self._encoders[model]

    @property
    def db_sift(self) -> geometric.DbSift:
        if self._db_sift is None:
            self._db_sift = geometric.DbSift(sift_path(self.cache_root, self.dataset))
        return self._db_sift

    @property
    def inlier_cache(self) -> geometric.InlierCache:
        if self._inlier_cache is None:
            self._inlier_cache = geometric.InlierCache(
                inlier_cache_path(self.cache_root, self.dataset), len(self.q_paths), len(self.db_paths))
        return self._inlier_cache

    # ---------- 流程 ----------

    def prepare_query(self, request: SearchRequest) -> Image.Image:
        """依來源裁切：官方 query 用 bbx，上傳圖片用使用者框選（沒有框選就用整張）。"""
        if request.qidx is not None:
            image = revisited.load_image(self.q_paths[request.qidx])
            if request.crop:
                image = crop_bbx(image, revisited.query_bbx(self.gnd, request.qidx))
        else:
            if request.image is None:
                raise ValueError("SearchRequest needs either image or qidx")
            image = request.image.convert("RGB")
        if request.user_box is not None:
            image = crop_user_box(image, request.user_box)
        return image

    def embed(self, image: Image.Image, model: str, scale: str) -> np.ndarray:
        return extract_multiscale(self.encoder(model), [image], scales_for(scale, self.cfg))[0]

    def search(self, request: SearchRequest, query_embs: dict[str, np.ndarray] | None = None) -> SearchResult:
        """query_embs：評估時傳入快取的 query embedding（{model: (D,)}），省略 embed 階段。"""
        c = parse_config_id(request.config_id)
        k = request.k or self.cfg["retrieval"]["k"]
        timer = StageTimer()
        image = None
        if query_embs is None or c["gv"]:
            image = self.prepare_query(request)

        # 1–2. 前處理與抽 embedding
        if query_embs is None:
            with timer.stage("embed"):
                query_embs = {m: self.embed(image, m, c["scale"]) for m in c["models"]}
        query_embs = {m: query_embs[m] for m in c["models"]}

        # 3–4. 初次檢索；alpha-QE 用初次檢索的前 n 名擴展 query 後再搜一次
        n_qe = self.cfg["aqe"]["n"]
        per_model_ids, per_model_scores = {}, {}
        for m, q in query_embs.items():
            idx = self.index(m, c["scale"], c["index"])
            with timer.stage("search"):
                scores, ids = faiss_index.search(idx, q, max(k, n_qe) if c["aqe"] else k)
            if c["aqe"]:
                with timer.stage("rerank"):
                    q = alpha_qe(q, self.db_embs(m, c["scale"]), ids[0], n_qe, self.cfg["aqe"]["alpha"])
                    scores, ids = faiss_index.search(idx, q, k)
                query_embs[m] = q
            per_model_ids[m], per_model_scores[m] = ids[0][:k], scores[0][:k]

        # 5. Ensemble：聯集 → 精確重算 → 加權融合
        if c["model"] == "ens":
            with timer.stage("search"):
                cand = ensemble.union_candidates(per_model_ids)
                db = {m: self.db_embs(m, c["scale"]) for m in c["models"]}
                ids, scores = ensemble.fuse(ensemble.rescore(query_embs, db, cand), self.cfg["ensemble"]["weights"], cand, k)
        else:
            m = c["models"][0]
            ids, scores = per_model_ids[m], per_model_scores[m]
            valid = ids >= 0
            ids, scores = ids[valid], scores[valid]

        # 6. 幾何驗證：官方且有裁切的 query 才使用 inlier 快取
        inliers = None
        if c["gv"]:
            use_cache = request.qidx is not None and request.crop and request.user_box is None
            new_ids, inliers, cost = geometric.rerank(
                image, ids, self.db_sift, self.cfg,
                cache=self.inlier_cache if use_cache else None,
                qidx=request.qidx if use_cache else None)
            # 依新順序對齊分數
            pos = {int(i): j for j, i in enumerate(ids)}
            scores = scores[[pos[int(i)] for i in new_ids]]
            ids = new_ids
            timer.add("rerank", cost)

        result = SearchResult(request.config_id, ids, scores, timer.as_dict(), query_image=image, inliers=inliers)
        if request.full_ranking:
            result.full_ranking = self._full_ranking(ids, query_embs, c)
        return result

    def _full_ranking(self, top_ids: np.ndarray, query_embs: dict[str, np.ndarray], c: dict) -> np.ndarray:
        """revisitop 的 compute_map 需要整個圖庫的排序：前 K 名用本次結果，其餘依精確分數補齊。

        所有設定的尾段都以同樣方式補齊，因此設定之間的差異只來自前 K 名。
        """
        weights = self.cfg["ensemble"]["weights"] if c["model"] == "ens" else {c["models"][0]: 1.0}
        exact = sum(weights[m] * (self.db_embs(m, c["scale"]) @ q) for m, q in query_embs.items())
        rest = np.argsort(-exact, kind="stable")
        rest = rest[~np.isin(rest, top_ids)]
        return np.concatenate([top_ids, rest]).astype(np.int64)

    def inlier_matches(self, query_image: Image.Image, db_idx: int) -> np.ndarray:
        """demo 畫連線用：現場重算一對影像的 inlier 匹配點 (M, 4)。"""
        g = self.cfg["geometric"]
        q_feat = geometric.extract_sift(query_image, g["max_side"], g["max_keypoints"])
        return geometric.match_pair(*q_feat, *self.db_sift.get(db_idx), self.cfg)
