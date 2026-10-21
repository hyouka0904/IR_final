"""多尺度抽取與合併；線上與離線共用同一份。"""


def rescale(image, scale: float):
    ...


def extract_multiscale(encoder, image, scales: list[float]) -> "np.ndarray":
    ...
