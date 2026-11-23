"""Offline gallery embedding."""
from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocess import preprocess


def embed_paths(encoder, paths: list[Path], batch_size: int, max_side: int,
                progress: bool = False) -> np.ndarray:
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for p in paths[start:start + batch_size]:
            with Image.open(p) as im:
                batch.append(preprocess(im, max_side))
        chunks.append(encoder.encode(batch))
        if progress:
            print(f"  {encoder.name}: {min(start + batch_size, len(paths))}/{len(paths)}",
                  end="\r", flush=True)
    if progress:
        print()
    return np.concatenate(chunks, axis=0)
