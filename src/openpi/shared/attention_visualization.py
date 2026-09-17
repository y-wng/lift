"""Optional attention-mask rendering, independent of model construction."""

from pathlib import Path

import numpy as np


def save_attention_mask(mask, file_path: str | Path) -> None:
    from PIL import Image

    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.where(np.asarray(mask, dtype=bool), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(destination)
