"""評估流程：對一個 query set 跑一組設定，逐 query 計算指標。評估與 demo 共用 src/pipeline.py。

Rocchio 的評估協定（模擬使用者回饋）：
  1. 先跑去掉 rocchio 的設定，取得初次排序
  2. 以 qrels 判斷前 feedback_depth 名（等級 >= rel_threshold 為相關），作為使用者回饋
  3. 帶著回饋重跑完整設定
  4. frozen ranking：已判斷的前幾名維持原位，其後接上新排序中尚未出現的結果
     （使用者已經看過的圖不該算成 Rocchio 的進步）
"""
import numpy as np

from src.data.sets import CategoryQrels, ImageSet, Qrels
from src.eval.metrics import category_coverage, query_metrics
from src.pipeline import Pipeline, parse_config_id, without_post
from src.schema import SearchRequest


def simulate_feedback(initial: np.ndarray, grades: np.ndarray, depth: int, threshold: int) -> tuple[list[int], list[int]]:
    judged = np.asarray(initial[:depth], dtype=np.int64)
    rel = grades[judged] >= threshold
    return judged[rel].tolist(), judged[~rel].tolist()


def frozen_ranking(initial: np.ndarray, new: np.ndarray, depth: int, k: int) -> np.ndarray:
    head = np.asarray(initial[:depth], dtype=np.int64)
    tail = np.asarray(new, dtype=np.int64)
    return np.concatenate([head, tail[~np.isin(tail, head)]])[:k]


def run_queries(pipe: Pipeline, queries: ImageSet, config_id: str, query_feats: list[dict], qrels: Qrels | None,
                depth: int) -> tuple[list[np.ndarray], list[dict], list[dict]]:
    """回傳 (每個 query 的排序, 每個 query 的指標, 每個 query 的延遲)。query_feats[i] 為快取的 query 特徵。"""
    c = parse_config_id(config_id)
    rcfg = pipe.cfg["rocchio"]
    rankings, per_query, latencies = [], [], []
    for i in range(len(queries)):
        exclude = [queries.exclude(i)] if queries.exclude(i) is not None else []
        grades = qrels.grades(i) if qrels is not None else None
        req = SearchRequest(config_id, k=depth, exclude=exclude)
        if c["rocchio"]:
            if grades is None:
                raise ValueError("Rocchio evaluation needs qrels to simulate feedback")
            init = pipe.search(SearchRequest(without_post(config_id, "rocchio"), k=depth, exclude=exclude), query_feats[i])
            req.feedback = simulate_feedback(init.ids, grades, rcfg["feedback_depth"], rcfg["rel_threshold"])
            res = pipe.search(req, query_feats[i])
            ids = frozen_ranking(init.ids, res.ids, rcfg["feedback_depth"], depth)
            lat = {s: init.latency_ms.get(s, 0.0) + res.latency_ms.get(s, 0.0) for s in ("search", "rerank")}
        else:
            res = pipe.search(req, query_feats[i])
            ids, lat = res.ids, res.latency_ms
        rankings.append(ids)
        latencies.append(lat)
        if grades is not None:
            m = query_metrics(ids, grades)
            if isinstance(qrels, CategoryQrels):
                m["cat_coverage_10"] = category_coverage(ids, qrels.q_cats[i], qrels.g_cats)
            per_query.append(m)
    return rankings, per_query, latencies
