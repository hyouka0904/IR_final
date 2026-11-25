import numpy as np
import pytest

from src.metrics import cohen_kappa
from src.revisited import full_ranking, protocol_gnd, query_box


def test_cohen_kappa_known_values():
    assert cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    # textbook 2x2 example: po = 0.7, pe = 0.5 -> kappa 0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    # a 0-vs-2 disagreement costs more than 0-vs-1 under linear weights
    near = cohen_kappa([0, 1, 2, 0], [1, 1, 2, 0], "linear")
    far = cohen_kappa([0, 1, 2, 0], [2, 1, 2, 0], "linear")
    assert near > far


def test_protocol_gnd_definitions():
    g = [{"easy": [1], "hard": [2], "junk": [3]}]
    e, m, h = (protocol_gnd(g, p)[0] for p in "EMH")
    assert e["ok"].tolist() == [1] and sorted(e["junk"].tolist()) == [2, 3]
    assert sorted(m["ok"].tolist()) == [1, 2] and m["junk"].tolist() == [3]
    assert h["ok"].tolist() == [2] and sorted(h["junk"].tolist()) == [1, 3]


def test_full_ranking_ip_and_hamming():
    db = np.eye(3, dtype=np.float32)
    assert full_ranking(db, db[1:2], "ip")[:, 0].tolist()[0] == 1
    codes = np.array([[0b00000000], [0b11111111], [0b00000001]], dtype=np.uint8)
    assert full_ranking(codes, codes[:1], "hamming")[:, 0].tolist() == [0, 2, 1]


def test_query_box_rounds_outward():
    assert query_box({"gnd": [{"bbx": [1.5, 2.2, 10.1, 20.9]}]}, 0) == (1, 2, 11, 21)


def test_judging_rel_falls_back_across_drives(tmp_path, monkeypatch):
    """Windows: data on another drive than the repo has no relative path (os.path.relpath
    raises ValueError); the CSV gets the absolute path and the HTML a file:// URL."""
    import importlib.util
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("judging", root / "scripts" / "07_judging.py")
    judging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judging)
    photo = tmp_path / "a b.jpg"
    assert judging._rel(photo, tmp_path) == "a b.jpg"

    def other_drive(path, start=None):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")
    monkeypatch.setattr(os.path, "relpath", other_drive)
    assert judging._rel(photo, root) == photo.resolve().as_posix()
    assert judging._rel(photo, root, url=True) == photo.resolve().as_uri()
