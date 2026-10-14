"""Render a trained NCA growing from its seed as a GIF (torch-free)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cellauto import render
from cellauto.nca.image import make_seed, rgba_to_rgb
from cellauto.nca.numpy_model import NumpyNCA


def growth_gif(
    model: NumpyNCA,
    out: str | Path,
    *,
    grid_size: int = 28,
    padding: int = 0,
    steps: int = 200,
    seed: int = 0,
    damage_at: Iterable[int] = (),
    scale: int = 8,
    fps: int = 20,
) -> int:
    """Write ``steps + 1`` frames (the seed, then each update); return the frame count.

    The fire mask and the damage mask come from separate RNG streams spawned from ``seed``.
    """
    rng, damage_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    start = make_seed(grid_size, model.config.n_channels, padding)
    states = [start, *model.run(start, steps, rng, tuple(damage_at), damage_rng)]
    frames = (render.rgb_to_image(rgba_to_rgb(s), scale) for s in states)
    render.save_gif(frames, out, fps)
    return len(states)
