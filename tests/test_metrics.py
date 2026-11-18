"""指標：與 pytrec_eval（trec_eval 的 Python 綁定）逐 query 比對。"""
import numpy as np
import pytest

from src.data.sets import CategoryQrels, SparseQrels
from src.eval.metrics import METRICS, category_coverage, cohen_kappa, mean_metrics, query_metrics, recall_loss
from src.eval.trec import to_pytrec

pytrec_eval = pytest.importorskip("pytrec_eval")
TREC_NAMES = {"ndcg_cut_10": "ndcg_cut_10", "map": "map", "recall_1": "recall_1", "recall_10": "recall_10", "recip_rank": "recip_rank"}


def _compare(qrels, rankings, n_docs):
    doc_ids = [f"d{i}" for i in range(n_docs)]
    q_ids = [f"q{i}" for i in range(len(rankings))]
    qrel, run = to_pytrec(q_ids, rankings, qrels, doc_ids)
    ev = pytrec_eval.RelevanceEvaluator(qrel, {"ndcg_cut.10", "map", "recall.1,10", "recip_rank"})
    ref = ev.evaluate(run)
    for qi, qid in enumerate(q_ids):
        ours = query_metrics(rankings[qi], qrels.grades(qi))
        if qid not in ref:
            assert all(np.isnan(v) for v in ours.values())
            continue
        for m in METRICS:
            assert ours[m] == pytest.approx(ref[qid][TREC_NAMES[m]], abs=1e-6), (qid, m)


def test_matches_trec_eval_graded():
    rng = np.random.default_rng(0)
    n_docs, n_cat = 300, 6
    q_cats = (rng.random((25, n_cat)) < 0.3).astype(np.uint8)
    g_cats = (rng.random((n_docs, n_cat)) < 0.2).astype(np.uint8)
    qrels = CategoryQrels(q_cats, g_cats, max_grade=2)
    rankings = [rng.permutation(n_docs)[:rng.integers(5, 120)] for _ in range(25)]
    _compare(qrels, rankings, n_docs)


def test_matches_trec_eval_single_relevant():
    rng = np.random.default_rng(1)
    n_docs = 200
    rels = [{int(rng.integers(n_docs)): 1} for _ in range(30)]
    rankings = [rng.permutation(n_docs)[:50] for _ in range(30)]
    _compare(SparseQrels(rels, n_docs), rankings, n_docs)


def test_hand_example():
    grades = np.array([0, 2, 1, 0, 0])
    m = query_metrics(np.array([0, 1, 3, 2]), grades)
    assert m["map"] == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert m["recip_rank"] == pytest.approx(0.5)
    assert m["recall_1"] == 0 and m["recall_10"] == 1
    dcg = 2 / np.log2(3) + 1 / np.log2(5)
    idcg = 2 / np.log2(2) + 1 / np.log2(3)
    assert m["ndcg_cut_10"] == pytest.approx(dcg / idcg)


def test_no_relevant_is_skipped_in_mean():
    a = query_metrics(np.array([0]), np.array([1, 0]))
    b = query_metrics(np.array([0]), np.array([0, 0]))
    assert mean_metrics([a, b])["map"] == pytest.approx(1.0)


def test_coverage_recall_loss_kappa():
    g_cats = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    assert category_coverage(np.array([0, 1]), np.array([1, 1, 0]), g_cats, k=2) == 0.5
    assert category_coverage(np.array([0, 2]), np.array([1, 1, 0]), g_cats, k=2) == 1.0
    assert recall_loss([np.array([1, 2, -1])], [np.array([1, 2, 3])]) == pytest.approx(1 / 3)
    assert cohen_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0
    # 教科書範例：觀察一致 0.7、期望一致 0.5 → kappa 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa([0, 1, 2], [0, 1, 2], "linear") == 1.0
