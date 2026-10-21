"""讀取 revisitop 的 gnd、imlist、bbx，並解析圖片路徑。"""


def load_gnd(dataset: str, data_root: str) -> dict:
    ...


def db_image_paths(gnd: dict, dataset: str, data_root: str) -> list[str]:
    ...


def query_image_paths(gnd: dict, dataset: str, data_root: str) -> list[str]:
    ...


def query_bbx(gnd: dict, qidx: int) -> tuple[float, float, float, float]:
    ...
