"""Gradio demo 進入點：上傳任意照片、選任務與設定、並排比較、框選、Rocchio 回饋、failure analysis。

需要先完成 scripts/02–04（圖庫特徵與索引）；failure analysis 分頁需要 scripts/05 的 per_query 結果。
檢索一律呼叫 src/pipeline.py，與評估腳本是同一份程式。
"""
import sys
from pathlib import Path

# 以 `python app/app.py` 執行時，app/ 在 sys.path 上（components 可直接匯入），repo 根目錄則需要補上
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch

from components.compare_view import (build_config_panel, build_result_view, config_id_of, panel_updates,
                                     render_result, result_choices)
from components.hard_queries import available_configs, load_hard_queries
from components.query_input import SOURCES, build_query_input, on_click, preview
from src.config import load_config
from src.data.sets import list_sets, load_qrels, load_set
from src.eval.metrics import METRICS, query_metrics
from src.features import load_query_image
from src.pipeline import Pipeline, parse_config_id
from src.schema import SearchRequest

CFG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = CFG["paths"]["cache_root"]
N_SHOW = CFG["demo"]["n_show"]
PRESETS = CFG["demo"]["presets"]
_pipelines, _sets, _qrels = {}, {}, {}


def get_pipeline(gallery: str) -> Pipeline:
    if gallery not in _pipelines:
        _pipelines[gallery] = Pipeline(gallery, CFG, device=DEVICE)
    return _pipelines[gallery]


def get_set(name: str):
    if name not in _sets:
        _sets[name] = load_set(CACHE, name)
    return _sets[name]


def get_qrels(name: str):
    if name not in _qrels:
        qs = get_set(name)
        _qrels[name] = load_qrels(CACHE, qs, get_pipeline(qs.meta["gallery"]).gallery, CFG)
    return _qrels[name]


def query_sets_of(gallery: str) -> list[str]:
    return [n for n in list_sets(CACHE, "queries") if get_set(n).meta.get("gallery") == gallery]


EMPTY = (None, None, [], None, None)          # raw, set_box, points, box, view


def on_gallery(gallery):
    sets = query_sets_of(gallery)
    return gr.update(choices=sets, value=sets[0] if sets else None), gr.update(choices=[], value=None), *EMPTY


def on_source(source, upload):
    from_set = source == SOURCES[1]
    states = EMPTY if from_set else (upload, None, [], None, preview(upload, None, None))
    return gr.update(visible=not from_set), gr.update(visible=from_set), *states


def on_qset(qset):
    qs = get_set(qset)
    return gr.update(choices=[(f"{i}: {q}", i) for i, q in enumerate(qs.ids)], value=None), *EMPTY


def load_set_query(qset, qidx):
    if qset is None or qidx is None:
        return EMPTY
    qs = get_set(qset)
    raw = load_query_image(qs.paths[int(qidx)])
    set_box = qs.box(int(qidx))
    return raw, set_box, [], None, preview(raw, set_box, None)


def on_upload(image):
    return image, None, [], None, preview(image, None, None)


def make_request(config_id, source, qset, qidx, raw, set_box, box, feedback=None):
    """回傳 (SearchRequest, 該 query 的 qrels 等級或 None, 最高等級)。"""
    if source == SOURCES[1] and qset is not None and qidx is not None:
        qs, i = get_set(qset), int(qidx)
        ex = qs.exclude(i)
        qrels = get_qrels(qset)
        req = SearchRequest(config_id, image=raw, user_box=box or set_box, exclude=[ex] if ex is not None else [],
                            feedback=feedback)
        if qrels is None:
            return req, None, 1
        return req, qrels.grades(i), qrels.max_grade
    return SearchRequest(config_id, image=raw, user_box=box, feedback=feedback), None, 1


def run_one(gallery, config_id, query, feedback=None):
    try:
        parse_config_id(config_id)
    except ValueError:
        raise gr.Error(f"不支援的設定組合：{config_id}")
    pipe = get_pipeline(gallery)
    req, grades, max_grade = make_request(config_id, *query, feedback=feedback)
    result = pipe.search(req)
    metrics = query_metrics(result.ids, grades) if grades is not None and (grades >= 1).any() else None
    return result, render_result(pipe, result, N_SHOW, grades, max_grade, metrics)


def search(gallery, source, qset, qidx, raw, set_box, box, compare, *panels):
    if raw is None:
        raise gr.Error("請先上傳圖片或選擇 query")
    query = (source, qset, qidx, raw, set_box, box)
    res_a, (gal_a, info_a) = run_one(gallery, config_id_of(*panels[:3]), query)
    gal_b, info_b = [], ""
    if compare:
        _, (gal_b, info_b) = run_one(gallery, config_id_of(*panels[3:]), query)
    choices = result_choices(get_pipeline(gallery), res_a, N_SHOW)
    return gal_a, info_a, gal_b, info_b, gr.update(choices=choices, value=[]), gr.update(choices=choices, value=[])


def search_feedback(gallery, source, qset, qidx, raw, set_box, box, rel, nonrel, *panel):
    """Rocchio：以勾選的相關 / 不相關結果更新 query，重新搜尋設定 A（自動加上 rocchio）。"""
    if raw is None:
        raise gr.Error("請先搜尋一次")
    if not rel and not nonrel:
        raise gr.Error("請至少勾選一張相關或不相關的圖")
    method, index, post = panel
    if method in ("bm25", "phash"):
        raise gr.Error("Rocchio 需要向量模型（clip / siglip / dinov2，或與 bm25 融合）")
    cid = config_id_of(method, index, list(post or []) + ["rocchio"])
    _, (gal, info) = run_one(gallery, cid, (source, qset, qidx, raw, set_box, box), feedback=(list(rel), list(nonrel)))
    return gal, info


def refresh_hard(qset, config_id, metric):
    if not qset or not config_id:
        return None
    df = load_hard_queries(CFG["paths"]["results_root"], qset, config_id, metric)
    if df is None:
        gr.Warning("找不到 per_query 結果，請先執行 scripts/05_run_eval.py")
    return df


def on_hard_set(qset):
    configs = available_configs(CFG["paths"]["results_root"], qset) if qset else []
    return gr.update(choices=configs, value=configs[0] if configs else None)


def pick_hard(qset, df, evt: gr.SelectData):
    """點選困難 query：切到該 query set 並載入那張 query（再回到「搜尋」分頁）。"""
    qidx = int(df.iloc[evt.index[0]]["qidx"])
    qs = get_set(qset)
    choices = [(f"{i}: {q}", i) for i, q in enumerate(qs.ids)]
    return (gr.update(value=qs.meta["gallery"]), gr.update(value=SOURCES[1]), gr.update(visible=False),
            gr.update(visible=True), gr.update(choices=query_sets_of(qs.meta["gallery"]), value=qset),
            gr.update(choices=choices, value=qidx), *load_set_query(qset, qidx))


def build_app() -> gr.Blocks:
    galleries = list_sets(CACHE, "gallery")
    all_qsets = list_sets(CACHE, "queries")
    with gr.Blocks(title="Zero-Shot Image Search") as demo:
        gr.Markdown("# Zero-Shot Image-to-Image Search\n上傳任意照片：**semantic** 找同一類東西，**similar** 找同一張圖或同一個場景。")
        with gr.Tab("搜尋"):
            with gr.Row():
                with gr.Column(scale=1):
                    q = build_query_input(galleries, CFG["demo"]["default_gallery"])
                with gr.Column(scale=2):
                    task = gr.Radio(list(PRESETS), value="semantic", label="任務（帶入建議設定）")
                    panel_a = build_config_panel("設定 A", PRESETS["semantic"])
                    compare = gr.Checkbox(value=True, label="並排比較")
                    panel_b = build_config_panel("設定 B", PRESETS["similar"])
                    go = gr.Button("搜尋", variant="primary")
            with gr.Row():
                gal_a, info_a = build_result_view("設定 A")
                gal_b, info_b = build_result_view("設定 B")
            with gr.Accordion("Rocchio 回饋（針對設定 A 的結果）", open=False):
                with gr.Row():
                    rel = gr.CheckboxGroup([], label="相關")
                    nonrel = gr.CheckboxGroup([], label="不相關")
                fb = gr.Button("以回饋重新搜尋設定 A")

        with gr.Tab("Failure analysis"):
            with gr.Row():
                hard_set = gr.Dropdown(all_qsets, value=all_qsets[0] if all_qsets else None, label="Query set")
                hard_cfg = gr.Dropdown(available_configs(CFG["paths"]["results_root"], all_qsets[0]) if all_qsets else [],
                                       label="設定")
                metric = gr.Radio(list(METRICS), value="map", label="排序依據")
            hard_btn = gr.Button("載入")
            hard_df = gr.Dataframe(interactive=False, label="由低到高（點選列即載入該 query，再回到「搜尋」分頁）")

        states = [q["raw"], q["set_box"], q["points"], q["box"], q["view"]]
        query = [q["source"], q["qset"], q["qidx"], q["raw"], q["set_box"], q["box"]]
        # 一律用 .input（只有使用者操作才觸發），避免程式更新元件時連鎖觸發重設
        q["gallery"].input(on_gallery, q["gallery"], [q["qset"], q["qidx"], *states])
        q["source"].input(on_source, [q["source"], q["upload"]], [q["upload"], q["set_row"], *states])
        q["qset"].input(on_qset, q["qset"], [q["qidx"], *states])
        q["qidx"].input(load_set_query, [q["qset"], q["qidx"]], states)
        q["upload"].upload(on_upload, q["upload"], states)
        q["view"].select(on_click, [q["raw"], q["set_box"], q["points"]], [q["points"], q["box"], q["view"]])
        q["clear"].click(lambda raw, sb: ([], None, preview(raw, sb, None)), [q["raw"], q["set_box"]],
                         [q["points"], q["box"], q["view"]])
        task.input(lambda t: panel_updates(PRESETS[t]), task, panel_a)
        go.click(search, [q["gallery"], *query, compare, *panel_a, *panel_b], [gal_a, info_a, gal_b, info_b, rel, nonrel])
        fb.click(search_feedback, [q["gallery"], *query, rel, nonrel, *panel_a], [gal_a, info_a])
        hard_set.input(on_hard_set, hard_set, hard_cfg)
        hard_btn.click(refresh_hard, [hard_set, hard_cfg, metric], hard_df)
        hard_df.select(pick_hard, [hard_set, hard_df],
                       [q["gallery"], q["source"], q["upload"], q["set_row"], q["qset"], q["qidx"], *states])
        demo.load(on_gallery, q["gallery"], [q["qset"], q["qidx"], *states])
    return demo


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    # 預設只給本機用，印出的網址可以直接點開；要讓同網段的其他電腦連線再加 --host 0.0.0.0
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="產生 Gradio 的公開網址（給不在同一網段的人看）")
    args = p.parse_args()
    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)
