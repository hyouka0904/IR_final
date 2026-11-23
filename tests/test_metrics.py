import math

import numpy as np
import pytest

from src.metrics import (average_precision, mean_ignore_nan, ndcg_at_k, reciprocal_rank,
                         recall_at_k, semantic_gains)
from src.trec import write_qrels, write_run


def test_semantic_gains_share_of_query_categories():
    gains = semantic_gains({1, 2}, [{1}, {1, 2, 3}, {4}])
    np.testing.assert_allclose(gains, [0.5, 1.0, 0.0])


def test_semantic_gains_empty_query_is_zero():
    np.testing.assert_allclose(semantic_gains(set(), [{1}, {2}]), [0.0, 0.0])


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([1.0, 0.5, 0.0], [0.0, 0.5, 1.0], 3) == pytest.approx(1.0)


def test_ndcg_known_value():
    assert ndcg_at_k([0.0, 1.0], [1.0, 0.0], 2) == pytest.approx(1 / math.log2(3))


def test_ndcg_truncates_at_k():
    # relevant item at rank 3 is outside k=2
    assert ndcg_at_k([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 2) == 0.0


def test_ndcg_no_relevant_is_nan():
    assert math.isnan(ndcg_at_k([0.0, 0.0], [0.0, 0.0], 2))


def test_ap_known_value():
    assert average_precision([True, False, True], 3) == pytest.approx((1 + 2 / 3) / 3)


def test_ap_zero_relevant_nan():
    assert math.isnan(average_precision([False, False], 0))


def test_recall_and_rr():
    ranked = [5, 7, 9]
    assert recall_at_k(ranked, 7, 1) == 0.0
    assert recall_at_k(ranked, 7, 2) == 1.0
    assert reciprocal_rank(ranked, 9) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, 42) == 0.0


def test_mean_ignore_nan_counts_skipped():
    mean, skipped = mean_ignore_nan([1.0, float("nan"), 0.0])
    assert mean == pytest.approx(0.5) and skipped == 1


def test_mean_ignore_nan_all_nan():
    mean, skipped = mean_ignore_nan([float("nan")])
    assert math.isnan(mean) and skipped == 1


def test_write_run_format(tmp_path):
    p = tmp_path / "x.run"
    write_run(p, [("q1", 42, 1, 0.9), ("q1", 7, 2, 0.25)], tag="clip")
    assert p.read_text().splitlines() == ["q1 Q0 42 1 0.900000 clip",
                                          "q1 Q0 7 2 0.250000 clip"]


def test_write_qrels_format(tmp_path):
    p = tmp_path / "x.qrels"
    write_qrels(p, [("q1", 42, 3)])
    assert p.read_text().splitlines() == ["q1 0 42 3"]


def test_write_run_breaks_ties_in_rank_order(tmp_path):
    # trec_eval re-sorts by score (ties by docno), so printed scores must strictly decrease
    p = tmp_path / "t.run"
    write_run(p, [("q", 2, 1, -3.0), ("q", 9, 2, -3.0), ("q", 4, 3, -3.0000001),
                  ("r", 5, 1, 0.5)], tag="phash")
    lines = [l.split() for l in p.read_text().splitlines()]
    q_scores = [float(l[4]) for l in lines if l[0] == "q"]
    assert q_scores == sorted(q_scores, reverse=True) and len(set(q_scores)) == 3
    assert q_scores[0] == -3.0 and abs(q_scores[2] + 3.0) < 1e-5
    assert lines[3][4] == "0.500000"  # a new query starts fresh
