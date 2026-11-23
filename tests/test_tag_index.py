import math

import pytest

from src.sparse.hw1_format import parse_posting_line
from src.sparse.tag_index import TagIndex
from tests.reference.hw2_weighting import Scheme

K1, B = 1.2, 0.75

DOCS = {
    1: [("cat", 3), ("sofa", 1)],
    2: [("dog", 5), ("cat", 2), ("grass", 1)],
    3: [("sofa", 4), ("lamp", 2)],
    4: [("dog", 1), ("grass", 3), ("sky", 2), ("cat", 1)],
}


def test_positions_are_token_slots():
    idx = TagIndex(K1, B)
    idx.add(7, [("cat", 3), ("sofa", 1)])
    assert idx.postings["cat"][7] == [0, 1, 2]
    assert idx.postings["sofa"][7] == [3]
    assert idx.doc_len(7) == 4


class _Hw2Stub:
    """The parts of HW2's IndexReader that Scheme('bm25') reads."""

    def __init__(self, idx: TagIndex, index_dir):
        self.N = idx.N
        self._idx = idx
        self.meta = {"avgdl": idx.avgdl}
        self.index_dir = str(index_dir)
        with open(index_dir / "doc_lengths.tsv", "w", encoding="utf-8", newline="\n") as f:
            for d in idx.ids:
                f.write(f"{d}\t{idx.doc_len(d)}\n")

    def df(self, term):
        return self._idx.df(term)


def test_bm25_matches_hw2_reference(tmp_path):
    idx = TagIndex.from_docs(DOCS, K1, B)
    scheme = Scheme(_Hw2Stub(idx, tmp_path), "bm25", k1=K1, b=B)
    query = {"cat": 1, "dog": 2, "grass": 3}
    ours = idx.score(query)
    for d, tags in DOCS.items():
        tf = dict(tags)
        ref = sum(qtf * scheme.doc_weight_fn(t)(d, tf[t]) for t, qtf in query.items() if t in tf)
        if ref == 0:
            assert d not in ours
        else:
            assert math.isclose(ours[d], ref, rel_tol=1e-9), (d, ours[d], ref)
    assert 3 not in ours  # shares no term


def test_search_ties_by_id_and_k_clipped():
    idx = TagIndex.from_docs({9: [("a", 1)], 4: [("a", 1)], 6: [("b", 1)]}, K1, B)
    hits = idx.search({"a": 1}, k=10)
    assert [d for d, _ in hits] == [4, 9]
    assert hits[0][1] == hits[1][1]
    assert len(idx.search({"a": 1}, k=1)) == 1
    assert idx.search({"zzz": 1}, k=5) == []


def test_save_load_roundtrip_same_scores(tmp_path):
    idx = TagIndex.from_docs({1: [("cat", 2), ("sofa", 1)], 2: [("dog", 1), ("cat", 1)]}, K1, B)
    idx.save(tmp_path)
    lines = (tmp_path / "dictionary.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "cat, 2"
    loaded = TagIndex.load(tmp_path, K1, B)
    q = {"cat": 1, "dog": 1}
    assert loaded.score(q) == pytest.approx(idx.score(q), rel=1e-12)
    assert loaded.doc_terms(1) == ["cat", "sofa"]
    assert loaded.doc_terms(2) == ["dog", "cat"]
    assert loaded.N == 2 and loaded.avgdl == idx.avgdl


def test_records_parse_with_hw1_parser(tmp_path):
    idx = TagIndex.from_docs(DOCS, K1, B)
    idx.save(tmp_path)
    lines = (tmp_path / "postings.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(idx.postings)
    for line in lines:
        term, df, entries = parse_posting_line(line)
        assert df == idx.df(term) == len(entries)
        assert [e.doc_id for e in entries] == sorted(e.doc_id for e in entries)
        for e in entries:
            assert e.tf == dict(DOCS[e.doc_id])[term] == len(e.positions)


def test_add_duplicate_id_raises():
    idx = TagIndex.from_docs({1: [("a", 1)]}, K1, B)
    with pytest.raises(ValueError):
        idx.add(1, [("b", 1)])


def test_add_updates_N_and_avgdl():
    idx = TagIndex.from_docs({1: [("a", 2)]}, K1, B)
    assert idx.N == 1 and idx.avgdl == 2
    idx.add(2, [("a", 1), ("b", 3)])
    assert idx.N == 2 and idx.avgdl == 3
    assert idx.df("a") == 2 and idx.df("b") == 1 and idx.df("zzz") == 0
    assert idx.ids == [1, 2]


@pytest.mark.parametrize("tags", [
    [("cat", 0)],                     # tf < 1
    [("cat", 2), ("cat", 1)],         # repeated term
    [("Cat (pet)", 1)],               # would break the HW1 record separators
    [],                               # no tags: the image would vanish on save/load
])
def test_add_rejects_tags_it_cannot_store(tags):
    idx = TagIndex(K1, B)
    with pytest.raises(ValueError):
        idx.add(1, tags)
    assert idx.N == 0 and idx.postings == {}
