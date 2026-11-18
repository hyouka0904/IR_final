"""設定面板與結果呈現；並排比較時兩組設定各用一份。"""
import gradio as gr

from src.index.faiss_index import INDEX_KINDS
from src.pipeline import METHODS, POSTS



def grade_mark(grade: int, max_grade: int) -> str:
    """最高等級為相關，介於中間為部分相關（只有分級 qrels 才會出現）。"""
    return "✅" if grade >= max_grade else ("🟡" if grade >= 1 else "❌")


def build_config_panel(title: str, config_id: str) -> list:
    """回傳 [method, index, post] 三個元件，用 config_id_of 組成 config_id。"""
    method, index, post = config_id.split("-")
    with gr.Group():
        gr.Markdown(f"**{title}**")
        with gr.Row():
            return [
                gr.Dropdown(list(METHODS), value=method, label="表示法"),
                gr.Dropdown([*INDEX_KINDS, "inv"], value=index, label="索引（bm25 單獨用 inv）"),
                gr.CheckboxGroup(list(POSTS), value=[] if post == "none" else post.split("+"), label="後處理"),
            ]


def config_id_of(method: str, index: str, post: list[str]) -> str:
    """後處理只能用在向量模型：bm25 與 phash 會忽略勾選的後處理。"""
    steps = [p for p in POSTS if p in (post or [])]
    if method == "bm25":
        index, steps = "inv", []
    elif method == "phash":
        index, steps = "flat", []
    elif index == "inv":
        index = "flat"
    return f"{method}-{index}-{'+'.join(steps) or 'none'}"


def panel_updates(config_id: str) -> list:
    method, index, post = config_id.split("-")
    return [gr.update(value=method), gr.update(value=index), gr.update(value=[] if post == "none" else post.split("+"))]


def build_result_view(title: str) -> tuple:
    with gr.Column():
        info = gr.Markdown(f"### {title}")
        gallery = gr.Gallery(columns=5, height=520, label=title, allow_preview=True)
    return gallery, info


def render_result(pipe, result, n_show: int, grades=None, max_grade: int = 1, metrics: dict | None = None) -> tuple[list, str]:
    """結果 → (Gallery 項目, 說明文字)。query 來自 query set 時另外標示 qrels 等級與該 query 的指標。"""
    terms = result.extra.get("query_terms")
    items = []
    for r, (idx, score) in enumerate(zip(result.ids[:n_show], result.scores[:n_show])):
        idx = int(idx)
        caption = f"#{r + 1} {pipe.gallery.ids[idx]} | {score:.3f}"
        if grades is not None:
            caption = f"{grade_mark(int(grades[idx]), max_grade)} {caption}"
        if terms:
            matched = pipe.matched_tags(terms, idx)[:3]
            if matched:
                caption += " | " + ", ".join(matched)
        items.append((pipe.gallery.paths[idx], caption))
    lat = " · ".join(f"{k} {v:.0f} ms" for k, v in result.latency_ms.items())
    lines = [f"### `{result.config_id}`", f"延遲：{lat or '-'}"]
    if result.query_tags:
        lines.append("Query tags：" + "、".join(f"{t} ({p:.2f})" for t, p in result.query_tags[:8]))
    if result.extra.get("rocchio"):
        lines.append("Rocchio：尚未提供回饋，這次等同沒有 Rocchio（勾選下方的相關 / 不相關後重新搜尋）")
    if metrics is not None:
        lines.append(f"此 query（前 {len(result.ids)} 名）：nDCG@10 **{metrics['ndcg_cut_10']:.3f}** · AP **{metrics['map']:.3f}** · "
                     f"RR {metrics['recip_rank']:.3f}　✅ 相關{' 🟡 部分相關' if max_grade > 1 else ''} ❌ 不相關")
    return items, "\n\n".join(lines)


def result_choices(pipe, result, n_show: int) -> list[tuple[str, int]]:
    """Rocchio 回饋用的勾選清單：(顯示文字, 圖庫索引)。"""
    return [(f"#{r + 1} {pipe.gallery.ids[int(i)]}", int(i)) for r, i in enumerate(result.ids[:n_show])]
