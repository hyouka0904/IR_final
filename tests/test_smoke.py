"""Smoke test: the numbered scripts end to end on a tiny fake dataset (marker `smoke`).

A local HTTP server stands in for cocodataset.org, the Holidays mirror and the label lists, and
a deterministic fake encoder replaces CLIP / SigLIP / DINOv2 (pHash stays real), so this runs
without network, GPU or model weights. It checks that the scripts fit together and write what
the next step reads, not retrieval quality. ROxford (08) is left out: it needs revisitop's code.

    python -m pytest -m smoke
"""
import contextlib
import functools
import http.server
import io
import json
import runpy
import sys
import tarfile
import threading
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import ROOT, load_config

pytestmark = pytest.mark.smoke
SCRIPTS = ROOT / "scripts"
CATS = {1: ("cat", (220, 40, 40)), 3: ("dog", (40, 200, 40)), 5: ("car", (40, 40, 220)),
        7: ("tree", (200, 200, 40))}
DIM = 64


class FakeModelEncoder:
    """Coarse 8x8 colour layout through a fixed random projection; deterministic per name."""
    metric = "ip"

    def __init__(self, name: str):
        self.name = name
        self.proj = np.random.default_rng(zlib.crc32(name.encode())).standard_normal((192, DIM))

    def encode(self, images):
        x = np.stack([np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32).ravel() / 255
                      for im in images])
        v = (x - 0.5) @ self.proj
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    def encode_text(self, texts):
        v = np.stack([np.random.default_rng(zlib.crc32(t.encode())).standard_normal(DIM)
                      for t in texts])
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def fake_get_encoder(name, cfg):
    from src.encoders import PHashEncoder
    if name == "phash":
        return PHashEncoder(cfg["models"]["phash"]["hash_size"])
    return FakeModelEncoder(name)


def _jpeg(color, rng, size=(48, 40)) -> bytes:
    a = np.clip(np.array(color) + rng.normal(0, 25, (size[1], size[0], 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def _make_sources(srv: Path, base: str) -> None:
    rng = np.random.default_rng(0)
    (srv / "img").mkdir()
    cat_ids = list(CATS)

    def instances(n, offset):
        images, anns = [], []
        for i in range(n):
            iid = offset + i
            cats = [cat_ids[i % 4]] + ([cat_ids[(i + 1) % 4]] if i % 3 == 0 else [])
            fn = f"{iid:012d}.jpg"
            (srv / "img" / fn).write_bytes(_jpeg(CATS[cats[0]][1], rng))
            images.append({"id": iid, "file_name": fn, "coco_url": f"{base}/img/{fn}",
                           "width": 48, "height": 40})
            for c in cats:
                anns.append({"image_id": iid, "category_id": c, "bbox": [4, 4, 30, 24],
                             "area": 720, "iscrowd": 0, "segmentation": [[4, 4, 34, 4, 34, 28]]})
        return {"images": images, "annotations": anns,
                "categories": [{"id": c, "name": n} for c, (n, _) in CATS.items()]}

    with zipfile.ZipFile(srv / "ann.zip", "w") as z:
        z.writestr("annotations/instances_train2017.json", json.dumps(instances(40, 1)))
        z.writestr("annotations/instances_val2017.json", json.dumps(instances(16, 5001)))
    for part, groups in (("jpg1", (1000, 1001)), ("jpg2", (1002,))):
        with tarfile.open(srv / f"{part}.tar.gz", "w:gz") as t:
            for g in groups:
                for j in range(2):
                    data = _jpeg(CATS[cat_ids[g % 4]][1], rng)
                    info = tarfile.TarInfo(f"jpg/{g}{j:02d}.jpg")
                    info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
    (srv / "oi.csv").write_text("LabelName,DisplayName\n/m/1,Cat\n/m/2,Bicycle\n/m/3,Tree\n",
                                encoding="utf-8")
    (srv / "places.txt").write_text("/b/beach 0\n/f/forest/broadleaf 1\n", encoding="utf-8")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_HEAD(self):
        f = Path(self.directory) / self.path.lstrip("/")
        self.send_response(200 if f.exists() else 404)
        if f.exists():
            self.send_header("Content-Length", str(f.stat().st_size))
        self.end_headers()


def run(script: str, *args: str) -> str:
    """Run a script in-process (so the fake encoder patch applies); returns its stdout."""
    out = io.StringIO()
    argv = sys.argv
    sys.argv = [script, *args]
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    except SystemExit as e:
        assert e.code in (0, None), f"{script} {' '.join(args)} exited with {e.code}:\n{out.getvalue()}"
    finally:
        sys.argv = argv
    return out.getvalue()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    work = tmp_path_factory.mktemp("smoke")
    srv = work / "srv"
    srv.mkdir()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Handler, directory=str(srv)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    _make_sources(srv, base)

    cfg = load_config()
    for key in ("data", "cache", "results", "vocab", "holidays", "revisited", "revisitop", "phone"):
        cfg["paths"][key] = str(work / {"data": "data/coco"}.get(key, key))
    cfg["user"]["dir"] = str(work / "user")
    cfg["pairs"]["dir"] = str(work / "pairs")
    cfg["data"].update(gallery_size=30, semantic_queries=8, similar_queries=10)
    cfg["sources"].update(coco_annotations=f"{base}/ann.zip",
                          holidays=[f"{base}/jpg1.tar.gz", f"{base}/jpg2.tar.gz"],
                          openimages=f"{base}/oi.csv", places365=f"{base}/places.txt")
    cfg["integrity"].update(coco_images={"train2017": 40, "val2017": 16}, coco_categories=4,
                            holidays={"images": 6, "queries": 3})
    cfg["eval"]["experiments"] = {"semantic": ["fusion+aqe", "fusion+mmr", "clip+rocchio",
                                               "clip@hnsw", "siglip@ivfpq", "clip@cluster"],
                                  "box": ["fusion+mmr"], "similar": ["dinov2@hnsw"],
                                  "holidays": ["dinov2+rocchio"]}
    cfg["judging"]["systems"] = ["fusion", "dinov2"]
    phone = work / "phone"
    phone.mkdir()
    rng = np.random.default_rng(1)
    for i, (_, color) in enumerate(list(CATS.values())[:3]):
        (phone / f"IMG_{i}.jpg").write_bytes(_jpeg(color, rng))
    config = work / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    with pytest.MonkeyPatch.context() as mp:
        import src.encoders
        import src.pipeline
        mp.setenv("IR_CONFIG", str(config))
        mp.setattr(src.encoders, "get_encoder", fake_get_encoder)
        mp.setattr(src.pipeline, "get_encoder", fake_get_encoder)
        mp.syspath_prepend(str(SCRIPTS))
        logs = {
            "download": run("01_download.py"),
            "check_before": run("check_data.py", "--only", "partials", "annotations", "manifests"),
            "embed": run("02_embed.py", "--models", "clip", "siglip", "dinov2", "phash"),
            "tag": run("03_tag.py"),
            "index": run("04_build_index.py", "--index", "flat", "cluster", "ivfpq", "hnsw"),
            "eval": run("05_evaluate.py"),
            "experiments": run("05_evaluate.py", "--experiments"),
            "pairs": run("06_pairs.py", "--synthetic", "2"),
            "search": run("search.py", str(phone / "IMG_0.jpg"), "--task", "semantic",
                          "--index", "hnsw", "--rerank", "mmr"),
        }
        run("07_judging.py", "pool")
        jpath = work / "results" / "phone" / "judgments.csv"
        lines = jpath.read_text(encoding="utf-8").splitlines()
        header, rows = lines[0], lines[1:]
        filled = [r.rstrip(",") + f",{i % 3},{(i + i // 4) % 3}" for i, r in enumerate(rows)]
        jpath.write_text("﻿" + "\n".join([header] + filled) + "\n", encoding="utf-8")  # Excel BOM
        logs["kappa"] = run("07_judging.py", "kappa")
        logs["qrels"] = run("07_judging.py", "qrels")
        logs["phone"] = run("05_evaluate.py", "--tasks", "phone")
        logs["check_after"] = run("check_data.py")
        from app.app import build_app
        from src.demo import DemoState, run_search
        state = DemoState.create(load_config())
        demo = build_app(state)
        with Image.open(phone / "IMG_0.jpg") as img:
            found = run_search(state, img.copy(), "semantic", "fusion", 5, "hnsw", ["aqe", "mmr"])
        yield {"work": work, "cfg": load_config(config), "logs": logs, "demo": demo,
               "found": found}
    server.shutdown()


def _summary(work):
    import csv
    with open(work / "results" / "summary.csv", newline="", encoding="utf-8") as f:
        return {(r["task"], r["config"]): r for r in csv.DictReader(f)}


def test_download_writes_every_manifest(pipeline):
    root = pipeline["work"] / "data" / "coco"
    gallery = json.loads((root / "gallery.json").read_text())
    assert len(gallery) == 30 + 6 and sum(r.get("source") == "holidays" for r in gallery) == 6
    assert len(json.loads((root / "queries_semantic.json").read_text())) == 8
    assert json.loads((root / "queries_box.json").read_text())
    assert len(json.loads((root / "queries_holidays.json").read_text())) == 3
    assert "failed 0" in pipeline["logs"]["check_before"]


def test_caches_and_indexes_cover_the_gallery(pipeline):
    log = pipeline["logs"]["check_after"]
    assert "failed 0" in log and "cache siglip" in log
    cache = pipeline["work"] / "cache"
    for rep in ("clip", "siglip", "dinov2"):
        for kind in ("flat", "cluster", "ivfpq", "hnsw"):
            assert (cache / rep / kind / "meta.json").exists(), (rep, kind)
    assert (cache / "tags" / "postings.txt").exists() and (cache / "tags_nococo" / "postings.txt").exists()


def test_cross_table_and_experiments(pipeline):
    s = _summary(pipeline["work"])
    for task in ("semantic", "box", "box_full", "similar", "holidays"):
        assert (task, "clip") in s, task
    assert ("similar", "fusion") not in s  # tags / fusion are semantic-only
    for key in [("semantic", "fusion+aqe"), ("semantic", "clip+rocchio"), ("semantic", "clip@hnsw"),
                ("semantic", "siglip@ivfpq"), ("box", "fusion+mmr"), ("similar", "dinov2@hnsw"),
                ("holidays", "dinov2+rocchio")]:
        assert key in s, key
    assert float(s[("semantic", "clip")]["recall_loss@10"]) == 0
    assert 0 <= float(s[("semantic", "siglip@ivfpq")]["recall_loss@10"]) <= 1
    assert float(s[("semantic", "clip@hnsw")]["index_mb"]) > 0
    for row in s.values():
        for k in ("ndcg@10", "map", "recall@1", "mrr"):
            if row.get(k) not in (None, "", "nan"):
                assert 0 <= float(row[k]) <= 1, (row["task"], row["config"], k)
    per_query = pipeline["work"] / "results" / "per_query"
    assert (per_query / "semantic_fusion+mmr.csv").exists() and (per_query / "holidays_dinov2.csv").exists()


def test_judging_kappa_and_phone_eval(pipeline):
    work = pipeline["work"]
    kappa = json.loads((work / "results" / "phone" / "kappa.json").read_text())
    assert kappa["n_pairs"] > 0 and -1 <= kappa["kappa"] <= 1
    assert (work / "results" / "phone" / "judgments.html").exists()
    assert (work / "results" / "phone" / "qrels.txt").read_text().strip()
    assert ("phone", "fusion") in _summary(work)


def test_pairs_search_cli_and_demo(pipeline):
    assert "partner at rank 1" in pipeline["logs"]["pairs"]
    assert "index=hnsw rerank=mmr" in pipeline["logs"]["search"]
    items, tags, timings, choices = pipeline["found"]
    assert len(items) == 5 and len(choices) == 5 and "@hnsw" in timings and "+aqe +mmr" in timings
    import gradio as gr
    assert isinstance(pipeline["demo"], gr.Blocks)
