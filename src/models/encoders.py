"""CLIP / SigLIP / DINOv2 的統一介面，輸出 L2-normalized 向量。

前處理一律用各模型官方的設定：open_clip 回傳的 preprocess（SigLIP 為直接縮放、CLIP 為短邊縮放 + 中央裁切），
DINOv2 則用官方範例的 Resize(256) + CenterCrop(224)。
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Encoder:
    def __init__(self, name: str, model: torch.nn.Module, forward, transform, device: str, tokenizer=None):
        self.name = name
        self.model = model.eval().to(device)
        self._forward = forward
        self.transform = transform
        self.device = device
        self.tokenizer = tokenizer          # 只有 CLIP / SigLIP 有，tagger 用
        self.dim = int(self.encode([Image.new("RGB", (64, 64))]).shape[1])

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        feats = self._forward(x).float()
        return F.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)


def load_encoder(name: str, cfg: dict, device: str = "cpu") -> Encoder:
    mcfg = cfg["models"][name]
    if name in ("clip", "siglip"):
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(mcfg["arch"], pretrained=mcfg["pretrained"])
        return Encoder(name, model, model.encode_image, preprocess, device, open_clip.get_tokenizer(mcfg["arch"]))
    if name == "dinov2":
        from torchvision import transforms as T
        size = mcfg["image_size"]
        transform = T.Compose([
            T.Resize(round(size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        model = torch.hub.load("facebookresearch/dinov2", mcfg["arch"])
        # forward() 回傳 CLS token 經過 head（Identity）後的向量
        return Encoder(name, model, model, transform, device)
    raise ValueError(f"Unknown model: {name}")
