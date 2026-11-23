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


def _unit(rows):
    import numpy as np
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_alpha_qe_moves_toward_neighbors():
    import numpy as np
    from src.rerank import alpha_qe
    q = _unit([1, 0, 0])
    nb = _unit([[0.8, 0.6, 0], [0.8, 0.6, 0]])
    out = alpha_qe(q, nb, n=2, alpha=1.0)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out[1] > 0 and out @ nb[0] > q @ nb[0]
    np.testing.assert_allclose(alpha_qe(q, nb[:0], 3, 3.0), q)  # no neighbours: unchanged


def test_rocchio_formula():
    import numpy as np
    from src.rerank import rocchio
    q, rel, non = _unit([1, 0, 0]), _unit([[0, 1, 0]]), _unit([[0, 0, 1]])
    out = rocchio(q, rel, non, alpha=1.0, beta=0.5, gamma=0.25)
    np.testing.assert_allclose(out, _unit([1, 0.5, -0.25]), rtol=1e-6)
    np.testing.assert_allclose(rocchio(q, rel[:0], non[:0], 1, 1, 1), q)


def test_mmr_promotes_a_different_result():
    import numpy as np
    from src.rerank import mmr
    hits = [(1, 0.9), (2, 0.89), (3, 0.5)]
    vecs = _unit([[1, 0], [1, 0.01], [0, 1]])  # 2 is a near-duplicate of 1
    out = mmr(hits, vecs, lam=0.5, depth=3)
    assert [d for d, _ in out] == [1, 3, 2]
    assert [d for d, _ in mmr(hits, vecs, lam=1.0, depth=3)] == [1, 2, 3]
    deep = mmr(hits + [(4, 0.1)], np.vstack([vecs, _unit([[1, 1]])]), lam=0.5, depth=3)
    assert deep[-1][0] == 4  # beyond depth keeps its place
