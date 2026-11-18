"""Query 來源：上傳任意照片 / 從評估用的 query set 選一張；兩者都可以框選（box cropping）。

框選方式：在圖上依序點兩下（左上角、右下角），預覽圖上會畫出紅框。
query set 中本來就帶框的 query（coco_semantic_box）以綠框標示，沒有另外框選時就用它。
"""
import gradio as gr
from PIL import Image, ImageDraw

SOURCES = ("上傳圖片", "評估 query set")


def draw_box(image: Image.Image, box, color: str) -> Image.Image:
    out = image.copy()
    ImageDraw.Draw(out).rectangle(box, outline=color, width=max(2, image.width // 200))
    return out


def preview(image: Image.Image | None, set_box, user_box) -> Image.Image | None:
    if image is None:
        return None
    if set_box is not None:
        image = draw_box(image, set_box, "lime")
    if user_box is not None:
        image = draw_box(image, user_box, "red")
    return image


def build_query_input(galleries: list[str], default: str) -> dict:
    """必須在 gr.Blocks 內呼叫；回傳各元件與 state，事件由 app.py 綁定。"""
    with gr.Column():
        gallery = gr.Dropdown(galleries, value=default if default in galleries else (galleries[0] if galleries else None), label="圖庫")
        source = gr.Radio(list(SOURCES), value=SOURCES[0], label="Query 來源")
        upload = gr.Image(type="pil", label="上傳圖片", sources=["upload", "clipboard", "webcam"])
        with gr.Row(visible=False) as set_row:
            qset = gr.Dropdown([], label="Query set", interactive=True)
            qidx = gr.Dropdown([], label="Query", interactive=True)
        view = gr.Image(type="pil", label="Query（點兩下框選：左上 → 右下）", interactive=False)
        clear = gr.Button("清除框選", size="sm")
    return {
        "gallery": gallery, "source": source, "upload": upload, "set_row": set_row, "qset": qset, "qidx": qidx,
        "view": view, "clear": clear,
        "raw": gr.State(None),       # 未裁切的原圖
        "set_box": gr.State(None),   # query set 自帶的框
        "points": gr.State([]),      # 已點選的角點
        "box": gr.State(None),       # 使用者框選 (x1, y1, x2, y2)
    }


def on_click(raw, set_box, points, evt: gr.SelectData):
    """點一下記一個角點；滿兩點即形成框選並更新預覽。"""
    points = (points + [tuple(evt.index)])[-2:]
    box = (*points[0], *points[1]) if len(points) == 2 else None
    return points, box, preview(raw, set_box, box)
