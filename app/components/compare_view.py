"""並排比較：同一個 query 以兩組設定同時搜尋，左右並列。"""
import os

import gradio as gr

from src.data.revisited import gt_label
from src.models.encoders import MODEL_NAMES
from src.index.faiss_index import INDEX_KINDS
from src.pipeline import RERANKS, SCALES

LABEL_MARK = {"positive": "✅", "junk": "➖", "negative": "❌"}


def build_config_panel(title: str, model: str = "dinov2", scale: str = "ss", index: str = "flat", rerank: str = "none") -> list:
    """回傳 [model, scale, index, rerank] 四個 Dropdown，用 config_id_of 組成 config_id。"""
    with gr.Group():
        gr.Markdown(f"**{title}**")
        with gr.Row():
            return [
                gr.Dropdown([*MODEL_NAMES, "ens"], value=model, label="模型"),
                gr.Dropdown(list(SCALES), value=scale, label="尺度"),
                gr.Dropdown(list(INDEX_KINDS), value=index, label="索引"),
                gr.Dropdown(list(RERANKS), value=rerank, label="Re-ranking"),
            ]


def config_id_of(model: str, scale: str, index: str, rerank: str) -> str:
    return f"{model}-{scale}-{index}-{rerank}"


def build_result_view(title: str) -> tuple:
    with gr.Column():
        info = gr.Markdown(f"### {title}")
        gallery = gr.Gallery(columns=5, height=520, label=title, allow_preview=False)
    return gallery, info


def render_result(pipe, result, qidx: int | None, n_show: int, ap: float | None) -> tuple[list, str]:
    """結果 → (Gallery 項目, 說明文字)。選官方 query 時另外標示 ground truth 與 AP。"""
    items = []
    for r, (db_idx, score) in enumerate(zip(result.ids[:n_show], result.scores[:n_show])):
        caption = f"#{r + 1} {os.path.basename(pipe.db_paths[db_idx])[:-4]} | {score:.3f}"
        if result.inliers is not None:
            caption += f" | inliers {int(result.inliers[r])}"
        if qidx is not None:
            caption = f"{LABEL_MARK[gt_label(pipe.gnd, qidx, int(db_idx))]} {caption}"
        items.append((pipe.db_paths[db_idx], caption))
    lat = " · ".join(f"{k} {v:.0f} ms" for k, v in result.latency_ms.items())
    lines = [f"### `{result.config_id}`", f"延遲：{lat or '-'}"]
    if ap is not None:
        lines.append(f"AP（Medium）：**{ap:.2f}**　✅ positive ➖ junk ❌ negative")
    return items, "\n\n".join(lines)
