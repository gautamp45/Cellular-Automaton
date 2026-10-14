"""Image and state helpers for the NCA (numpy, channel-first float32)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cellauto._types import FloatArray


def load_target(path: str | Path, size: int) -> FloatArray:
    """Load an image as ``(4, size, size)`` RGBA in [0, 1] with RGB premultiplied by alpha."""
    with Image.open(path) as im:
        img = im.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / np.float32(255.0)
    arr = arr.copy()
    arr[..., :3] *= arr[..., 3:]
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def rgba_to_rgb(x: FloatArray) -> FloatArray:
    """Composite premultiplied RGBA over white: ``clip(1 - clip(a, 0, 1) + rgb, 0, 1)``."""
    rgb = x[:3]
    alpha = np.clip(x[3:4], 0.0, 1.0)
    return np.asarray(np.clip(1.0 - alpha + rgb, 0.0, 1.0), dtype=np.float32)


def pad(x: FloatArray, p: int) -> FloatArray:
    """Zero-pad the two spatial axes by ``p`` on each side."""
    if p < 0:
        raise ValueError("padding must be >= 0")
    return np.pad(x, ((0, 0), (p, p), (p, p))).astype(np.float32)


def make_seed(size: int, n_channels: int = 16, padding: int = 0) -> FloatArray:
    """All zeros except channels ``3:`` set to 1 at the centre cell ``(size // 2, size // 2)``."""
    if n_channels < 4:
        raise ValueError("n_channels must be >= 4 (RGB + alpha)")
    x = np.zeros((n_channels, size, size), dtype=np.float32)
    x[3:, size // 2, size // 2] = 1.0
    return pad(x, padding)


def circle_mask(size: int, rng: np.random.Generator) -> FloatArray:
    """Random disc of shape ``(size, size)`` (1 inside) on a ``[-1, 1]`` grid.

    Centre ~ U(-0.5, 0.5)^2 and radius ~ U(0.1, 0.4). Used for damage: ``x * (1 - mask)``.
    """
    cx, cy = rng.uniform(-0.5, 0.5, size=2)
    r = rng.uniform(0.1, 0.4)
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None]
    inside = ((xs - cx) / r) ** 2 + ((ys - cy) / r) ** 2 < 1.0
    return inside.astype(np.float32)
