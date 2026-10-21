"""CLIP / SigLIP / DINOv2 的統一介面，輸出 L2-normalized 向量。"""


class Encoder:
    name: str
    dim: int

    def preprocess(self, image):
        ...

    def encode(self, images) -> "np.ndarray":
        ...


def load_encoder(name: str, cfg: dict, device: str = "cpu") -> Encoder:
    ...
