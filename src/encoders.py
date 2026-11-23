"""CLIP / SigLIP / DINOv2 / pHash behind one interface. Pre-trained weights only."""
from typing import Protocol

import imagehash
import numpy as np
from PIL import Image


class Encoder(Protocol):
    name: str
    metric: str  # "ip" (L2-normalized float32) or "hamming" (packed uint8 bits)

    def encode(self, images: list[Image.Image]) -> np.ndarray: ...


def resolve_device(spec: str) -> str:
    if spec != "auto":
        return spec
    try:
        import torch
    except ImportError:  # search-only installs (CI, the demo on a CPU box) have no torch
        return "cpu"
    if torch.cuda.is_available():
        try:  # the wheel may not ship kernels for an old GPU
            (torch.ones(2, device="cuda") * 2).sum().item()
            return "cuda"
        except Exception:
            pass
    return "cpu"


class PHashEncoder:
    name = "phash"
    metric = "hamming"

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        bits = [imagehash.phash(img, hash_size=self.hash_size).hash.flatten() for img in images]
        return np.packbits(np.asarray(bits, dtype=bool), axis=1)


class _TorchEncoder:
    metric = "ip"

    def __init__(self, model, transform, device: str):
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device

    def _features(self, batch):
        raise NotImplementedError

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        import torch
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        with torch.inference_mode():
            feats = self._features(batch).float()
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)


class ClipEncoder(_TorchEncoder):
    """OpenCLIP image/text towers. SigLIP is loaded through the same class (name="siglip")."""
    name = "clip"

    def __init__(self, arch: str, pretrained: str, device: str = "cpu", batch_size: int = 32,
                 name: str = "clip"):
        import open_clip
        model, _, transform = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        super().__init__(model, transform, device)
        self.name = name
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.batch_size = batch_size

    def _features(self, batch):
        return self.model.encode_image(batch)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """L2-normalized CLIP text embeddings, float32, in batches of batch_size."""
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[i:i + self.batch_size]).to(self.device)
                feats = self.model.encode_text(tokens).float()
                out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class Dinov2Encoder(_TorchEncoder):
    name = "dinov2"

    def __init__(self, timm_name: str, image_size: int, device: str = "cpu"):
        import timm
        model = timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=image_size)
        data_cfg = timm.data.resolve_data_config({"input_size": (3, image_size, image_size)},
                                                 model=model)
        transform = timm.data.create_transform(**data_cfg)
        super().__init__(model, transform, device)

    def _features(self, batch):
        return self.model(batch)  # pooled CLS token (num_classes=0)


def get_encoder(name: str, cfg: dict) -> Encoder:
    m = cfg["models"]
    if name == "phash":
        return PHashEncoder(m["phash"]["hash_size"])
    if name == "clip":
        return ClipEncoder(m["clip"]["arch"], m["clip"]["pretrained"], resolve_device(m["device"]),
                           cfg["embed"]["batch_size"])
    if name == "siglip":
        return ClipEncoder(m["siglip"]["arch"], m["siglip"]["pretrained"],
                           resolve_device(m["device"]), cfg["embed"]["batch_size"], name="siglip")
    if name == "dinov2":
        return Dinov2Encoder(m["dinov2"]["timm_name"], m["dinov2"]["image_size"],
                             resolve_device(m["device"]))
    raise ValueError(f"unknown representation {name!r}; expected clip, siglip, dinov2 or phash")
