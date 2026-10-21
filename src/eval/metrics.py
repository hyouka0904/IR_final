"""包裝 revisitop compute_map，並計算 recall 損失與分階段延遲。"""


def evaluate(ranks: "np.ndarray", gnd: list[dict], ks: list[int]) -> dict:
    ...


def per_query_ap(ranks: "np.ndarray", gnd: list[dict], protocol: str) -> "np.ndarray":
    ...


def recall_loss(ann_ids: "np.ndarray", flat_ids: "np.ndarray") -> float:
    ...


class StageTimer:
    def __init__(self):
        ...

    def stage(self, name: str):
        ...

    def as_dict(self) -> dict[str, float]:
        ...
