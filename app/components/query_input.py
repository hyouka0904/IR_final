"""Query 來源：上傳 / 選官方 query / 框選。

框選方式：在圖上依序點兩下（左上角、右下角），預覽圖上會畫出框。
官方 query 預設以 bbx 裁切（圖上以綠框標示）；若另外框選，則改用框選區域。
"""
import gradio as gr
from PIL import Image, ImageDraw

SOURCES = ("官方 query", "上傳圖片")


def draw_box(image: Image.Image, box, color: str) -> Image.Image:
    out = image.copy()
    ImageDraw.Draw(out).rectangle(box, outline=color, width=max(2, image.width // 200))
    return out


def preview(image: Image.Image | None, bbx, user_box) -> Image.Image | None:
    if image is None:
        return None
    if bbx is not None:
        image = draw_box(image, bbx, "lime")
    if user_box is not None:
        image = draw_box(image, user_box, "red")
    return image


def build_query_input(datasets: list[str]) -> dict:
    """必須在 gr.Blocks 內呼叫；回傳各元件與 state，事件由 app.py 綁定。"""
    with gr.Column():
        dataset = gr.Dropdown(list(datasets), value=datasets[0], label="圖庫")
        source = gr.Radio(list(SOURCES), value=SOURCES[0], label="Query 來源")
        official = gr.Dropdown([], label="官方 query", interactive=True)
        upload = gr.Image(type="pil", label="上傳圖片", sources=["upload", "clipboard"], visible=False)
        view = gr.Image(type="pil", label="Query（點兩下框選：左上 → 右下）", interactive=False)
        clear = gr.Button("清除框選", size="sm")
    return {
        "dataset": dataset, "source": source, "official": official, "upload": upload, "view": view, "clear": clear,
        "raw": gr.State(None),       # 未裁切的原圖
        "bbx": gr.State(None),       # 官方 bbx
        "points": gr.State([]),      # 已點選的角點
        "box": gr.State(None),       # 框選結果 (x1, y1, x2, y2)
    }


def on_click(raw, bbx, points, evt: gr.SelectData):
    """點一下記一個角點；滿兩點即形成框選並更新預覽。"""
    points = (points + [tuple(evt.index)])[-2:]
    box = (*points[0], *points[1]) if len(points) == 2 else None
    return points, box, preview(raw, bbx, box)
