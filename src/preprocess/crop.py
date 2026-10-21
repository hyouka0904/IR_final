"""bbx 裁切與使用者框選裁切；線上與離線共用同一份。"""


def crop_bbx(image, bbx: tuple[float, float, float, float]):
    ...


def crop_user_box(image, box: tuple[int, int, int, int]):
    ...
