"""Web demo (Gradio): query photo -> semantic or similar search, add photos, hard queries.

    python app/app.py        # http://127.0.0.1:7860 (demo.host / demo.port)
UI wiring only; the logic lives in src/demo.py and every search goes through SearchEngine.search.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.demo import (DemoState, _abs, add_handler, hard_queries,  # noqa: E402
                      hard_queries_label, query_image_path, reps_for_task, search_handler)


def editor_image(value) -> Image.Image | None:
    """The (cropped) photo from a gr.ImageEditor value."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("composite") if value.get("composite") is not None \
            else value.get("background")
    if value is None:
        return None
    return value if isinstance(value, Image.Image) else Image.fromarray(value)


def build_app(state: DemoState) -> gr.Blocks:
    cfg = state.cfg
    available = state.engine.available()
    sem_reps = reps_for_task(cfg, available, "semantic")
    sem_default = cfg["tasks"]["semantic"]["representation"]
    sem_default = sem_default if sem_default in sem_reps else (sem_reps or [None])[0]

    def on_task(task):
        reps = reps_for_task(cfg, available, task)
        default = cfg["tasks"][task]["representation"]
        return gr.update(choices=reps, value=default if default in reps else (reps or [None])[0])

    def on_rep(task, rep):
        return gr.update(value=hard_queries(cfg, rep) if rep else [],
                         label=hard_queries_label(task, rep or "-"))

    def on_search(value, task, rep, k):
        return search_handler(state, editor_image(value), task, rep, k)

    def on_add(value):
        return add_handler(state, editor_image(value))

    def on_hard(evt: gr.SelectData):
        row = evt.row_value if getattr(evt, "row_value", None) else None
        if not row:
            return gr.update()
        return Image.open(query_image_path(cfg, int(row[0]))).convert("RGB")

    with gr.Blocks(title="Zero-shot image search") as demo:
        gr.Markdown("## Zero-shot image-to-image search")
        with gr.Row():
            with gr.Column(scale=1):
                query = gr.ImageEditor(type="pil", label="Query photo (crop in the editor)")
                task = gr.Radio(["semantic", "similar"], value="semantic", label="Task")
                rep = gr.Dropdown(sem_reps, value=sem_default, label="Representation")
                k = gr.Slider(1, 50, value=cfg["search"]["k"], step=1, label="k")
                search_btn = gr.Button("Search", variant="primary")
                qtags = gr.Textbox(label="Query tags", interactive=False)
                add_btn = gr.Button("Add this photo to the gallery")
                add_msg = gr.Markdown()
            with gr.Column(scale=2):
                results = gr.Gallery(label="Results (#rank · score · shared tags)", columns=5,
                                     height="auto")
                timings = gr.Markdown()
                hard = gr.Dataframe(value=hard_queries(cfg, sem_default) if sem_default else [],
                                    headers=["qid", "AP"], interactive=False,
                                    label=hard_queries_label("semantic", sem_default or "-"))
        task.change(on_task, task, rep)
        rep.change(on_rep, [task, rep], hard)
        search_btn.click(on_search, [query, task, rep, k], [results, qtags, timings])
        add_btn.click(on_add, query, add_msg)
        hard.select(on_hard, None, query)
    return demo


def launch_kwargs(cfg: dict) -> dict:
    """Gradio serves only files under allowed_paths (plus cwd/temp): allow the gallery and the
    user photos wherever they live, so the demo works from any working directory."""
    return {"server_name": cfg["demo"]["host"], "server_port": cfg["demo"]["port"],
            "allowed_paths": [str(resolve_path(cfg, "data")), str(_abs(cfg["user"]["dir"]))]}


def main():
    cfg = load_config()
    state = DemoState.create(cfg)
    print(f"gallery loaded; available: {', '.join(state.engine.available())}; "
          f"user photos: {len(state.user.records()) - len(state.skipped)}"
          + (f"; skipped unreadable user photos: {state.skipped}" if state.skipped else ""))
    build_app(state).launch(**launch_kwargs(cfg))


if __name__ == "__main__":
    main()
