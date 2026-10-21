"""SIFT 比對 + RANSAC 幾何驗證；先查 inlier 快取，沒有才計算。"""


def extract_sift(image, max_keypoints: int | None = None) -> tuple["np.ndarray", "np.ndarray"]:
    ...


def count_inliers(q_kp, q_desc, db_kp, db_desc, cfg: dict) -> tuple[int, "np.ndarray"]:
    ...


def load_db_sift(path: str, db_idx: int) -> tuple["np.ndarray", "np.ndarray"]:
    ...


class InlierCache:
    def __init__(self, path: str):
        ...

    def get(self, qidx: int, db_idx: int) -> int | None:
        ...

    def put(self, qidx: int, db_idx: int, n_inliers: int) -> None:
        ...


def rerank(query_image, candidates: "np.ndarray", cfg: dict, qidx: int | None = None) -> tuple["np.ndarray", "np.ndarray"]:
    ...
