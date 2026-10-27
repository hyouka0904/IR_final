"""CLIP / SigLIP / DINOv2 的統一介面，輸出 L2-normalized 向量。"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from src.preprocess.multiscale import scale_image

MODEL_NAMES = ("clip", "siglip", "dinov2")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Encoder:
    def __init__(self, name: str, model: torch.nn.Module, forward, base_size: int, mean, std, patch: int | None, device: str):
        self.name = name
        self.model = model.eval().to(device)
        self._forward = forward
        self.base_size = base_size
        self.mean, self.std = list(mean), list(std)
        self.patch = patch          # 有值表示彈性解析度（DINOv2）
        self.device = device
        self.dim = int(self.encode([Image.new("RGB", (base_size, base_size))]).shape[1])

    def preprocess(self, image: Image.Image, scale: float = 1.0) -> torch.Tensor:
        image = scale_image(image, self.base_size, scale, self.patch)
        return TF.normalize(TF.to_tensor(image), self.mean, self.std)

    @torch.no_grad()
    def encode(self, images: list[Image.Image], scale: float = 1.0) -> np.ndarray:
        x = torch.stack([self.preprocess(im, scale) for im in images]).to(self.device)
        feats = self._forward(x).float()
        return F.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)


def load_encoder(name: str, cfg: dict, device: str = "cpu") -> Encoder:
    mcfg = cfg["models"][name]
    if name in ("clip", "siglip"):
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(mcfg["arch"], pretrained=mcfg["pretrained"])
        pp = open_clip.get_model_preprocess_cfg(model)
        size = pp["size"]
        size = size[0] if isinstance(size, (tuple, list)) else size
        return Encoder(name, model, model.encode_image, size, pp["mean"], pp["std"], None, device)
    if name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", mcfg["arch"])
        # forward() 回傳 CLS token 經過 head（Identity）後的向量
        return Encoder(name, model, model, mcfg["image_size"], IMAGENET_MEAN, IMAGENET_STD, model.patch_size, device)
    raise ValueError(f"Unknown model: {name}")
