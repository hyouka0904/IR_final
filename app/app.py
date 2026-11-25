"""Web demo (Gradio): query photo -> semantic or similar search, add photos, hard queries.

    python app/app.py                                  # http://127.0.0.1:7860 (demo.host / demo.port)
    python app/app.py --host 0.0.0.0 --auth user:pass  # reachable from the LAN / VPN, with a login
    python app/app.py --share                          # temporary public Gradio link
UI wiring only; the logic lives in src/demo.py and every search goes through SearchEngine.search.
"""
import argparse
import inspect
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.demo import (DemoState, _abs, add_handler, hard_queries,  # noqa: E402
                      hard_queries_label, query_image_path, reps_for_task, run_search)

AUTO_RERANK = ["aqe", "mmr"]  # Rocchio runs from the marked results instead

# Light / dark switch: Gradio styles the page dark when <body> has the "dark" class. The choice is
# remembered in the browser; with no choice yet the OS setting decides.
THEME_JS = """
() => {
  let mode = null;
  try { mode = localStorage.getItem('ir-demo-theme'); } catch (e) {}
  if (!mode) mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.body.classList.toggle('dark', mode === 'dark');
}
"""
TOGGLE_JS = """
() => {
  const dark = !document.body.classList.contains('dark');
  document.body.classList.toggle('dark', dark);
  try { localStorage.setItem('ir-demo-theme', dark ? 'dark' : 'light'); } catch (e) {}
}
"""


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


def _js_on_launch() -> bool:
    """Gradio 6 takes js on launch(); Gradio 5 only on gr.Blocks (which warns that it moves)."""
    return "js" in inspect.signature(gr.Blocks.launch).parameters


def build_app(state: DemoState) -> gr.Blocks:
    cfg = state.cfg
    engine = state.engine
    available = engine.available()
    sem_reps = reps_for_task(cfg, available, "semantic")
    sem_default = cfg["tasks"]["semantic"]["representation"]
    sem_default = sem_default if sem_default in sem_reps else (sem_reps or [None])[0]

    def index_update(rep, value="flat"):
        kinds = engine.index_kinds(rep) if rep else ["flat"]
        return gr.update(choices=kinds, value=value if value in kinds else "flat")

    def rerank_update(rep):
        ok = bool(rep) and engine.can_rerank(rep)
        return gr.update(interactive=True) if ok else gr.update(interactive=False, value=[])

    def on_task(task):
        reps = reps_for_task(cfg, available, task)
        default = cfg["tasks"][task]["representation"]
        rep = default if default in reps else (reps or [None])[0]
        return (gr.update(choices=reps, value=rep), index_update(rep), rerank_update(rep),
                gr.update(choices=reps, value=None))

    def on_rep(task, rep):
        return (gr.update(value=hard_queries(cfg, rep) if rep else [],
                          label=hard_queries_label(task, rep or "-")),
                index_update(rep), rerank_update(rep))

    def search(value, task, rep, index, rerank, k, qid, cmp_rep, cmp_index):
        img = editor_image(value)
        items, tags, timings, choices = run_search(state, img, task, rep, k, index, rerank,
                                                   query_qid=qid)
        if cmp_rep:
            cmp_items, _, cmp_timings, _ = run_search(state, img, task, cmp_rep, k, cmp_index,
                                                      query_qid=qid)
        else:
            cmp_items, cmp_timings = [], ""
        marks = gr.update(choices=choices, value=[])
        return items, tags, timings, marks, gr.update(choices=choices, value=[]), \
            cmp_items, cmp_timings

    def rocchio(value, task, rep, index, rerank, k, qid, rel, nonrel):
        if not rel and not nonrel:
            return gr.update(), gr.update(), "⚠ Mark at least one result as relevant or not relevant."
        if not engine.can_rerank(rep):
            return gr.update(), gr.update(), (f"⚠ Rocchio needs a vector representation; "
                                              f"{rep} has none.")
        items, tags, timings, _ = run_search(state, editor_image(value), task, rep, k, index,
                                             list(rerank or []) + ["rocchio"],
                                             feedback=(list(rel), list(nonrel)), query_qid=qid)
        return items, tags, timings

    def on_add(value):
        return add_handler(state, editor_image(value))

    def on_hard(evt: gr.SelectData):
        row = evt.row_value if getattr(evt, "row_value", None) else None
        if not row:
            return gr.update(), gr.update()
        qid = int(row[0])
        return Image.open(query_image_path(cfg, qid)).convert("RGB"), qid

    blocks_kw = {"title": "Zero-shot image search"}
    if not _js_on_launch():
        blocks_kw["js"] = THEME_JS
    with warnings.catch_warnings():  # Gradio 5: "js ... will be removed in Gradio 6.0" (handled)
        warnings.filterwarnings("ignore", message=".*'js' parameter.*", category=DeprecationWarning)
        demo = gr.Blocks(**blocks_kw)
    with demo:
        with gr.Row():
            gr.Markdown("## Zero-shot image-to-image search")
            theme_btn = gr.Button("🌓 Light / dark", size="sm", scale=0, min_width=140)
        qid = gr.State(None)  # the evaluation query on screen, if loaded from the panel
        with gr.Row():
            with gr.Column(scale=1):
                query = gr.ImageEditor(type="pil", label="Query photo (crop in the editor)")
                task = gr.Radio(["semantic", "similar"], value="semantic", label="Task")
                rep = gr.Dropdown(sem_reps, value=sem_default, label="Representation")
                with gr.Row():
                    index = gr.Dropdown(engine.index_kinds(sem_default) if sem_default else ["flat"],
                                        value="flat", label="Index")
                    rerank = gr.CheckboxGroup(AUTO_RERANK, value=[], label="Re-rank",
                                              interactive=bool(sem_default)
                                              and engine.can_rerank(sem_default))
                k = gr.Slider(1, 50, value=cfg["search"]["k"], step=1, label="k")
                search_btn = gr.Button("Search", variant="primary")
                qtags = gr.Textbox(label="Query tags", interactive=False)
                add_btn = gr.Button("Add this photo to the gallery")
                add_msg = gr.Markdown()
            with gr.Column(scale=2):
                results = gr.Gallery(label="Results (#rank · score · shared tags)", columns=5,
                                     height="auto")
                timings = gr.Markdown()
                with gr.Accordion("Rocchio feedback: mark results, then re-run", open=False):
                    with gr.Row():
                        rel = gr.CheckboxGroup([], label="Relevant")
                        nonrel = gr.CheckboxGroup([], label="Not relevant")
                    rocchio_btn = gr.Button("Re-run with feedback")
                with gr.Accordion("Compare with another representation", open=False):
                    with gr.Row():
                        cmp_rep = gr.Dropdown(sem_reps, value=None, label="Representation B "
                                              "(empty = off)")
                        cmp_index = gr.Dropdown(["flat"], value="flat", label="Index B")
                    cmp_results = gr.Gallery(label="Results B", columns=5, height="auto")
                    cmp_timings = gr.Markdown()
                hard = gr.Dataframe(value=hard_queries(cfg, sem_default) if sem_default else [],
                                    headers=["qid", "AP"], interactive=False,
                                    label=hard_queries_label("semantic", sem_default or "-"))
        theme_btn.click(None, None, None, js=TOGGLE_JS)
        task.change(on_task, task, [rep, index, rerank, cmp_rep])
        rep.change(on_rep, [task, rep], [hard, index, rerank])
        cmp_rep.change(lambda r: index_update(r), cmp_rep, cmp_index)
        query.upload(lambda: None, None, qid)  # a new photo is not an evaluation query
        search_btn.click(search, [query, task, rep, index, rerank, k, qid, cmp_rep, cmp_index],
                         [results, qtags, timings, rel, nonrel, cmp_results, cmp_timings])
        rocchio_btn.click(rocchio, [query, task, rep, index, rerank, k, qid, rel, nonrel],
                          [results, qtags, timings])
        add_btn.click(on_add, query, add_msg)
        hard.select(on_hard, None, [query, qid])
    return demo


def launch_kwargs(cfg: dict, host: str | None = None, port: int | None = None,
                  auth: str | None = None, share: bool = False) -> dict:
    """Gradio serves only files under allowed_paths (plus cwd/temp): allow the gallery, Holidays
    and the user photos wherever they live, so the demo works from any working directory."""
    kw = {"server_name": host or cfg["demo"]["host"], "server_port": port or cfg["demo"]["port"],
          "allowed_paths": [str(resolve_path(cfg, "data")), str(_abs(cfg["user"]["dir"])),
                            str(resolve_path(cfg, "holidays"))],
          "share": share}
    if auth:
        user, sep, password = auth.partition(":")
        if not sep or not user:
            raise SystemExit("--auth must look like user:password")
        kw["auth"] = (user, password)
    if _js_on_launch():
        kw["js"] = THEME_JS
    return kw


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", help=f"default demo.host ({cfg['demo']['host']}); 0.0.0.0 = every "
                                   "network interface (LAN / VPN)")
    ap.add_argument("--port", type=int, help=f"default demo.port ({cfg['demo']['port']})")
    ap.add_argument("--auth", default=os.environ.get("IR_DEMO_AUTH"),
                    help="user:password login (or the IR_DEMO_AUTH environment variable); "
                         "set it whenever the demo is reachable from other machines")
    ap.add_argument("--share", action="store_true", help="also create a public Gradio link")
    args = ap.parse_args()
    state = DemoState.create(cfg)
    print(f"gallery loaded; available: {', '.join(state.engine.available())}; "
          f"user photos: {len(state.user.records()) - len(state.skipped)}"
          + (f"; skipped unreadable user photos: {state.skipped}" if state.skipped else ""))
    if not state.engine.available():
        print("!! nothing to search yet: run scripts/01_download.py, 02_embed.py, 03_tag.py and "
              "04_build_index.py first")
    kw = launch_kwargs(cfg, args.host, args.port, args.auth, args.share)
    if kw["server_name"] == "0.0.0.0":
        print(f"listening on every interface: open http://127.0.0.1:{kw['server_port']} here, or "
              f"http://<this machine's IP>:{kw['server_port']} from another machine")
    build_app(state).launch(**kw)


if __name__ == "__main__":
    main()
