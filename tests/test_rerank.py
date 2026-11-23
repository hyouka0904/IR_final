import pytest

from src.rerank import rrf, shared_tags


def test_rrf_known_scores():
    out = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [d for d, _ in out] == [1, 3, 2]
    s = dict(out)
    assert s[1] == pytest.approx(1 / 61 + 1 / 62)
    assert s[3] == pytest.approx(1 / 63 + 1 / 61)
    assert s[2] == pytest.approx(1 / 62)


def test_rrf_tie_by_id():
    assert [d for d, _ in rrf([[5], [4]], k=60)] == [4, 5]


def test_rrf_empty_list_is_identity_order():
    out = rrf([[7, 3, 9], []], k=60)
    assert [d for d, _ in out] == [7, 3, 9]
    assert rrf([[], []], k=60) == []


def test_shared_tags_query_order():
    assert shared_tags(["sofa", "cat", "lamp"], ["cat", "dog", "sofa"]) == ["sofa", "cat"]
    assert shared_tags(["x"], []) == []
