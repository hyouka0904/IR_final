"""Gradio demo 進入點：組合各元件並啟動介面。

需要先完成 scripts/02–05（embedding、索引、SIFT、per_query 結果）。
"""
import sys
from pathlib import Path

# 以 `python app/app.py` 執行時，app/ 在 sys.path 上（components 可直接匯入），repo 根目錄則需要補上
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch

from components.compare_view import build_config_panel, build_result_view, config_id_of, render_result
from components.hard_queries import load_hard_queries
from components.inlier_view import draw_inliers
from components.query_input import SOURCES, build_query_input, on_click, preview
from src.config import load_config
from src.data import revisited
from src.eval.metrics import query_ap
from src.pipeline import Pipeline, parse_config_id
from src.schema import SearchRequest

CFG = load_config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_pipelines: dict[str, Pipeline] = {}


def get_pipeline(dataset: str) -> Pipeline:
    if dataset not in _pipelines:
        _pipelines[dataset] = Pipeline(dataset, CFG, device=DEVICE)
    return _pipelines[dataset]


def query_choices(dataset: str):
    names = get_pipeline(dataset).gnd["qimlist"]
    return gr.update(choices=[(n, i) for i, n in enumerate(names)], value=0)


def load_official(dataset: str, qidx):
    """選官方 query：顯示原圖並以綠框標出官方 bbx。"""
    if qidx is None:
        return None, None, [], None, None
    pipe = get_pipeline(dataset)
    raw = revisited.load_image(pipe.q_paths[int(qidx)])
    bbx = revisited.query_bbx(pipe.gnd, int(qidx))
    return raw, bbx, [], None, preview(raw, bbx, None)


def load_upload(image):
    return image, None, [], None, preview(image, None, None)


def switch_source(source: str, dataset: str, qidx, upload):
    """切換來源時一併重設 query，避免沿用另一個來源的圖。"""
    official = source == SOURCES[0]
    states = load_official(dataset, qidx) if official else load_upload(upload)
    return gr.update(visible=official), gr.update(visible=not official), *states


def run_one(pipe: Pipeline, config_id: str, source: str, qidx, raw, box):
    if source == SOURCES[0]:
        # 官方 query：沒有框選時依官方協定用 bbx 裁切；有框選時改用框選區域
        req = SearchRequest(pipe.dataset, config_id, qidx=int(qidx), crop=box is None, user_box=box, full_ranking=True)
    else:
        req = SearchRequest(pipe.dataset, config_id, image=raw, user_box=box)
    result = pipe.search(req)
    ap = None
    if req.qidx is not None:
        ap = query_ap(result.full_ranking, pipe.gnd["gnd"][req.qidx], "M")
    return result, render_result(pipe, result, req.qidx, CFG["demo"]["n_show"], ap)


def search(dataset, source, qidx, raw, box, compare, *panels):
    if raw is None:
        raise gr.Error("請先選擇官方 query 或上傳圖片")
    pipe = get_pipeline(dataset)
    res_a, (gal_a, info_a) = run_one(pipe, config_id_of(*panels[:4]), source, qidx, raw, box)
    gal_b, info_b = [], ""
    if compare:
        _, (gal_b, info_b) = run_one(pipe, config_id_of(*panels[4:]), source, qidx, raw, box)
    return gal_a, info_a, gal_b, info_b, res_a.query_image, res_a.ids


def show_inliers(dataset, query_image, ids, evt: gr.SelectData):
    """點選左側結果：畫出 query 與該結果之間的 inlier 連線。"""
    if query_image is None or ids is None:
        return None, ""
    pipe = get_pipeline(dataset)
    db_idx = int(ids[evt.index])
    matches = pipe.inlier_matches(query_image, db_idx)
    ok = len(matches) >= CFG["geometric"]["min_inliers"]
    msg = f"inlier 數：**{len(matches)}**（門檻 {CFG['geometric']['min_inliers']}，{'通過' if ok else '未通過'}驗證）"
    return draw_inliers(query_image, revisited.load_image(pipe.db_paths[db_idx]), matches), msg


def refresh_hard(dataset, protocol, *panel):
    df = load_hard_queries(CFG["paths"]["results_root"], dataset, config_id_of(*panel), protocol)
    if df is None:
        gr.Warning("找不到 per_query 結果，請先執行 scripts/05_run_grid.py")
    return df


def pick_hard(df, evt: gr.SelectData):
    """點選困難 query：切到官方 query 並選中它。"""
    qidx = int(df.iloc[evt.index[0]]["qidx"])
    return gr.update(value=SOURCES[0]), gr.update(value=qidx)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Landmark Retrieval Diagnostics") as demo:
        gr.Markdown("# 地標檢索診斷工具")
        with gr.Tab("搜尋"):
            with gr.Row():
                with gr.Column(scale=1):
                    q = build_query_input(CFG["datasets"])
                with gr.Column(scale=2):
                    panel_a = build_config_panel("設定 A（baseline）")
                    compare = gr.Checkbox(value=True, label="並排比較")
                    panel_b = build_config_panel("設定 B", model="ens", scale="ms", rerank="aqe+gv")
                    go = gr.Button("搜尋", variant="primary")
            with gr.Row():
                gal_a, info_a = build_result_view("設定 A")
                gal_b, info_b = build_result_view("設定 B")
            with gr.Accordion("幾何驗證視覺化（點選設定 A 的結果）", open=True):
                inlier_msg = gr.Markdown()
                inlier_img = gr.Image(label="SIFT inlier 匹配", interactive=False)
            last_query, last_ids = gr.State(None), gr.State(None)

        with gr.Tab("困難 query"):
            with gr.Row():
                hard_panel = build_config_panel("依此設定排序", rerank="none")
                protocol = gr.Radio(["E", "M", "H"], value="M", label="協定")
            hard_btn = gr.Button("載入")
            hard_df = gr.Dataframe(interactive=False, label="AP 由低到高（點選列即載入該 query，再回到「搜尋」分頁）")

        states = [q["raw"], q["bbx"], q["points"], q["box"], q["view"]]
        q["dataset"].change(query_choices, q["dataset"], q["official"])
        q["official"].change(load_official, [q["dataset"], q["official"]], states)
        q["upload"].upload(load_upload, q["upload"], states)
        q["source"].change(switch_source, [q["source"], q["dataset"], q["official"], q["upload"]],
                           [q["official"], q["upload"], *states])
        q["view"].select(on_click, [q["raw"], q["bbx"], q["points"]], [q["points"], q["box"], q["view"]])
        q["clear"].click(lambda raw, bbx: ([], None, preview(raw, bbx, None)), [q["raw"], q["bbx"]],
                         [q["points"], q["box"], q["view"]])
        go.click(search, [q["dataset"], q["source"], q["official"], q["raw"], q["box"], compare, *panel_a, *panel_b],
                 [gal_a, info_a, gal_b, info_b, last_query, last_ids])
        gal_a.select(show_inliers, [q["dataset"], last_query, last_ids], [inlier_img, inlier_msg])
        hard_btn.click(refresh_hard, [q["dataset"], protocol, *hard_panel], hard_df)
        hard_df.select(pick_hard, hard_df, [q["source"], q["official"]])
        demo.load(query_choices, q["dataset"], q["official"])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0")
