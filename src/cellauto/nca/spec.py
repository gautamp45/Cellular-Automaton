"""Architecture constants of the (Growing) Neural CA used by every checkpoint in this repo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellauto._types import FloatArray

# name -> (3x3 integer filter, scale). dx = filter / scale and dy = filter.T / scale.
FILTERS: dict[str, tuple[list[list[int]], float]] = {
    "sobel": ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 8.0),
    "scharr": ([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 16.0),
    "gaussian": ([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 16.0),
    "laplacian": ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 8.0),
    "mean": ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 9.0),
}
ALPHA_CHANNEL = 3
ALIVE_THRESHOLD = 0.1


def perception_kernels(name: str) -> FloatArray:
    """Shape (3, 3, 3): ``[identity, dx = f / s, dy = (f / s).T]``."""
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}; expected one of {sorted(FILTERS)}")
    f, scale = FILTERS[name]
    fx = np.asarray(f, dtype=np.float32) / np.float32(scale)
    identity = np.zeros((3, 3), dtype=np.float32)
    identity[1, 1] = 1.0
    return np.stack([identity, fx, fx.T], axis=0).astype(np.float32)


def is_isotropic(name: str) -> bool:
    """True when ``dx == dy`` (the filter carries no direction): gaussian, laplacian, mean."""
    k = perception_kernels(name)
    return bool(np.array_equal(k[1], k[2]))


@dataclass(frozen=True)
class NCAConfig:
    n_channels: int = 16
    hidden_channels: int = 128
    filter: str = "sobel"
    fire_rate: float = 0.5

    def __post_init__(self) -> None:
        if self.n_channels < 4:
            raise ValueError("n_channels must be >= 4 (RGB + alpha)")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        if self.filter not in FILTERS:
            raise ValueError(f"unknown filter {self.filter!r}; expected one of {sorted(FILTERS)}")
        if not 0.0 < self.fire_rate <= 1.0:
            raise ValueError("fire_rate must be in (0, 1]")
