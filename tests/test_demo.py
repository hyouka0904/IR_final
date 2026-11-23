import csv

import pytest
from PIL import Image

from src.config import load_config
from src.demo import (DemoState, UserGallery, add_handler, hard_queries, reps_for_task,
                      search_handler)
from tests.fakes import COLORS, solid, tag_engine

FIRST = 1_000_000_000


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config()
    cfg["user"]["dir"] = str(tmp_path / "user")
    cfg["paths"]["results"] = str(tmp_path / "results")
    return cfg


@pytest.fixture
def state(cfg, tmp_path):
    files = {}
    for i, c in COLORS.items():
        files[i] = tmp_path / f"{i}.png"
        solid(c).save(files[i])
    return DemoState(tag_engine(cfg), cfg, files, UserGallery(tmp_path / "user"))


def test_search_handler_items_and_captions(state):
    items, qtags, timings = search_handler(state, solid(COLORS[1]), "semantic", "fusion", 3)
    assert len(items) == 3
    path, caption = items[0]
    assert path.endswith(".png") and caption.startswith("#1 · ")
    assert "red" in caption
    assert qtags == "red, green"
    assert "embed" in timings and "search" in timings and "rerank" in timings


def test_search_handler_similar_has_no_tags(state):
    items, qtags, _ = search_handler(state, solid(COLORS[2]), "similar", "dinov2", 2)
    assert items[0][1].count("·") == 1 and qtags == ""


def test_search_handler_disallowed_rep_returns_message(state):
    items, qtags, msg = search_handler(state, solid(COLORS[1]), "similar", "tags", 5)
    assert items == [] and qtags == "" and msg.startswith("⚠") and "allowed" in msg


def test_search_handler_no_image(state):
    items, _, msg = search_handler(state, None, "semantic", "fusion", 5)
    assert items == [] and msg


def test_add_then_restart_keeps_ids(cfg, state):
    assert add_handler(state, solid((0, 200, 200))) == f"Added as #{FIRST}"
    assert add_handler(state, solid((90, 90, 90))) == f"Added as #{FIRST + 1}"
    assert [r["image_id"] for r in state.user.records()] == [FIRST, FIRST + 1]
    assert state.path_of(FIRST).exists()

    restarted = DemoState.create(cfg, engine=tag_engine(cfg))
    assert {FIRST, FIRST + 1} <= restarted.engine.gallery_ids("clip")
    assert restarted.engine.tag_indexes["tags"].N == 6
    items, _, _ = search_handler(restarted, solid((0, 200, 200)), "similar", "clip", 1)
    assert items[0][0] == str(restarted.path_of(FIRST))
    assert add_handler(restarted, solid((5, 5, 5))) == f"Added as #{FIRST + 2}"


def test_reps_for_task_filters(cfg):
    avail = ["clip", "dinov2", "tags", "fusion"]
    assert reps_for_task(cfg, avail, "semantic") == ["clip", "dinov2", "tags", "fusion"]
    assert reps_for_task(cfg, avail, "similar") == ["clip", "dinov2"]


def test_hard_queries_sorted(cfg, tmp_path):
    d = tmp_path / "results" / "per_query"
    d.mkdir(parents=True)
    with open(d / "semantic_fusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "ap", "ndcg@10"])
        w.writerows([[5, 0.4, 0.5], [6, "nan", 0.1], [7, 0.01, 0.2], [8, 0.2, 0.3]])
    cfg["demo"]["hard_queries"] = 2
    assert hard_queries(cfg, "fusion") == [[7, 0.01], [8, 0.2]]
    assert hard_queries(cfg, "clip") == []


def test_build_app_returns_blocks(state):
    import gradio as gr

    from app.app import build_app
    assert isinstance(build_app(state), gr.Blocks)


def test_search_waits_for_add_lock(state):
    """Search and add share one engine; a running add must block a search (and vice versa)."""
    import threading
    out = []
    with state.lock:
        t = threading.Thread(target=lambda: out.append(
            search_handler(state, solid(COLORS[1]), "semantic", "fusion", 3)))
        t.start()
        t.join(0.3)
        assert t.is_alive() and out == []
    t.join(5)
    assert len(out[0][0]) == 3
    with state.lock:
        t = threading.Thread(target=lambda: out.append(add_handler(state, solid((1, 99, 1)))))
        t.start()
        t.join(0.3)
        assert t.is_alive()
    t.join(5)
    assert out[-1].startswith("Added as #")


def test_saved_user_photo_is_the_preprocessed_image(state):
    """A phone photo with EXIF rotation (and a transparent editor layer) must be saved as the
    engine saw it, so a restart re-embeds the same pixels."""
    import io
    img = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    msg = add_handler(state, Image.open(buf))
    image_id = int(msg.split("#")[1])
    with Image.open(state.path_of(image_id)) as saved:
        assert saved.size == (20, 40)
    rgba = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    image_id = int(add_handler(state, rgba).split("#")[1])
    with Image.open(state.path_of(image_id)) as saved:
        assert saved.convert("RGB").getpixel((0, 0))[0] > 240  # white, not black


def test_launch_allows_gallery_and_user_dirs(cfg):
    from app.app import launch_kwargs
    from src.config import resolve_path
    kw = launch_kwargs(cfg)
    assert kw["server_name"] == cfg["demo"]["host"] and kw["server_port"] == cfg["demo"]["port"]
    assert str(resolve_path(cfg, "data")) in kw["allowed_paths"]
    assert cfg["user"]["dir"] in kw["allowed_paths"]  # absolute tmp dir in this fixture


def test_add_save_failure_leaves_engine_unchanged(state, monkeypatch):
    before = set(state.engine.all_ids())

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(state.user, "write_image", broken)
    msg = add_handler(state, solid((0, 200, 200)))
    assert msg.startswith("⚠") and "disk full" in msg
    assert set(state.engine.all_ids()) == before and state.user.records() == []
    monkeypatch.undo()
    assert add_handler(state, solid((0, 200, 200))) == f"Added as #{FIRST}"


def test_restart_skips_missing_or_corrupt_user_photos(cfg, state):
    for c in ((0, 200, 200), (200, 0, 200), (9, 9, 9)):
        add_handler(state, solid(c))
    state.path_of(FIRST).unlink()
    state.path_of(FIRST + 2).write_bytes(b"not an image")
    restarted = DemoState.create(cfg, engine=tag_engine(cfg))
    assert restarted.skipped == [FIRST, FIRST + 2]
    assert FIRST + 1 in restarted.engine.all_ids() and FIRST not in restarted.engine.all_ids()
    # a skipped photo's id is never handed out again
    assert add_handler(restarted, solid((5, 5, 5))) == f"Added as #{FIRST + 3}"


def test_search_handler_result_without_a_file(cfg, state):
    state.coco_files.pop(1)
    items, _, msg = search_handler(state, solid(COLORS[1]), "similar", "clip", 4)
    assert len(items) == 3 and "1 result" in msg


def test_hard_queries_label_names_the_semantic_task():
    from src.demo import hard_queries_label
    assert "semantic" in hard_queries_label("semantic", "fusion")
    lab = hard_queries_label("similar", "dinov2")
    assert "semantic" in lab and "dinov2" in lab
