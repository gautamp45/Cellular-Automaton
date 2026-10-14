"""Pillow-only rendering helpers (no matplotlib / pygame)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from cellauto._validate import check_int_range

PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "dark": ((12, 12, 16), (240, 240, 240)),
    "light": ((250, 250, 250), (20, 20, 24)),
    "green": ((8, 16, 8), (80, 255, 120)),
}

# 256 x 3 uint8 lookup table for matplotlib's "magma", generated once with
# np.round(matplotlib.colormaps["magma"](np.arange(256))[:, :3] * 255) and pasted here.
_MAGMA_HEX = (
    "00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e090720"
    "0a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c1044"
    "1d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f1163311165331067341069"
    "36106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b"
    "51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c81"
    "6a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e2482802582812581"
    "8326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f"
    "9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367a"
    "b73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070"
    "d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064"
    "e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695c"
    "f56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761"
    "fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571"
    "fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287"
    "fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1"
    "fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf"
)
_LUTS: dict[str, npt.NDArray[np.uint8]] = {
    "magma": np.frombuffer(bytes.fromhex(_MAGMA_HEX), dtype=np.uint8).reshape(256, 3),
    "gray": np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1),
}


def _scaled(arr: npt.NDArray[np.uint8], scale: int) -> Image.Image:
    s = check_int_range("scale", scale, 1, 64)
    img = Image.fromarray(arr)
    if s > 1:
        img = img.resize((img.width * s, img.height * s), Image.Resampling.NEAREST)
    return img


def binary_to_image(grid: npt.ArrayLike, scale: int = 4, palette: str = "dark") -> Image.Image:
    """Render a 2-D 0/1 array as an RGB image, one scale x scale block per cell."""
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; expected one of {sorted(PALETTES)}")
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError(f"grid must be 2-D, got {g.ndim}-D")
    colours = np.asarray(PALETTES[palette], dtype=np.uint8)
    return _scaled(colours[(g > 0).astype(np.intp)], scale)


def scalar_to_image(a: npt.ArrayLike, scale: int = 4, cmap: str = "magma") -> Image.Image:
    """Render a 2-D float array in [0, 1] (clipped) through a 256-entry colour LUT."""
    if cmap not in _LUTS:
        raise ValueError(f"unknown cmap {cmap!r}; expected one of {sorted(_LUTS)}")
    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"array must be 2-D, got {x.ndim}-D")
    idx = np.rint(np.clip(x, 0.0, 1.0) * 255).astype(np.intp)
    return _scaled(_LUTS[cmap][idx], scale)


def rgb_to_image(rgb: npt.ArrayLike, scale: int = 4) -> Image.Image:
    """Render (3, H, W) or (H, W, 3) floats in [0, 1].

    A leading axis of length 3 is read as channels-first, so a (3, N, 3) array is ambiguous
    and is treated as channels-first.
    """
    x = np.asarray(rgb, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"rgb must be 3-D, got {x.ndim}-D")
    if x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    elif x.shape[2] != 3:
        raise ValueError(f"rgb must have a channel axis of length 3, got shape {x.shape}")
    return _scaled(np.rint(np.clip(x, 0.0, 1.0) * 255).astype(np.uint8), scale)


def save_png(img: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def save_gif(frames: Iterable[Image.Image], path: str | Path, fps: int = 20) -> None:
    """Write a looping GIF.

    Pillow merges *identical consecutive* frames into one longer frame, so a static pattern can
    have fewer stored frames than were supplied; playback timing is unchanged.
    """
    rate = check_int_range("fps", fps, 1, 1000)
    imgs = list(frames)
    if not imgs:
        raise ValueError("no frames to write")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=1000 // rate,
        loop=0,
    )
