"""Torch-free reference implementation of the NCA update (inference only).

Layout and conventions (all verified against the PyTorch checkpoints):

* state ``x`` has shape ``(C, H, W)`` float32; channels 0-2 are RGB, channel 3 is alpha;
* perception is a depthwise 3x3 cross-correlation with zero padding, output index ``3 * c + k``
  for ``k`` in (identity, dx, dy);
* update is ``w2 @ relu(w1 @ perception + b1)`` (no bias on the second layer), applied with a
  per-cell Bernoulli(fire_rate) mask;
* a cell lives if the 3x3 max of alpha exceeds 0.1 (out-of-bounds ignored), both before and after
  the update.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from cellauto._types import FloatArray
from cellauto.nca.image import circle_mask
from cellauto.nca.spec import ALIVE_THRESHOLD, ALPHA_CHANNEL, NCAConfig, perception_kernels


class NumpyNCA:
    def __init__(
        self, w1: FloatArray, b1: FloatArray, w2: FloatArray, config: NCAConfig | None = None
    ) -> None:
        self.config = config or NCAConfig()
        c, h = self.config.n_channels, self.config.hidden_channels
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        if self.w1.shape != (h, 3 * c) or self.b1.shape != (h,) or self.w2.shape != (c, h):
            raise ValueError(
                f"weights {self.w1.shape}, {self.b1.shape}, {self.w2.shape} do not match "
                f"config (C={c}, hidden={h})"
            )
        self.kernels = perception_kernels(self.config.filter)

    @classmethod
    def from_npz(cls, path: str | Path, config: NCAConfig | None = None) -> NumpyNCA:
        with np.load(path) as z:
            return cls(z["w1"], z["b1"], z["w2"], config)

    @staticmethod
    def alive(alpha: FloatArray) -> np.ndarray:
        """``(H, W)`` bool: 3x3 max-pool (``-inf`` padding) of alpha above the threshold."""
        p = np.pad(alpha, 1, constant_values=-np.inf)
        h, w = alpha.shape
        pooled = np.max([p[i : i + h, j : j + w] for i in range(3) for j in range(3)], axis=0)
        return np.asarray(pooled > ALIVE_THRESHOLD)

    def perceive(self, x: FloatArray) -> FloatArray:
        """``(C, H, W)`` -> ``(3C, H * W)``, index ``3c + k``, zero padding."""
        c, h, w = x.shape
        p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        shifted = [p[:, i : i + h, j : j + w] for i in range(3) for j in range(3)]
        kern = self.kernels
        per = [
            sum(
                kern[k, i, j] * shifted[3 * i + j]
                for i in range(3)
                for j in range(3)
                if kern[k, i, j] != 0
            )
            for k in range(3)
        ]
        return np.stack(per, axis=1).reshape(3 * c, h * w).astype(np.float32)

    def hidden_activations(self, x: FloatArray) -> FloatArray:
        """``(hidden, H * W)`` ReLU activations for the state ``x``."""
        return np.maximum(self.w1 @ self.perceive(x) + self.b1[:, None], 0.0).astype(np.float32)

    def step(
        self,
        x: FloatArray,
        rng: np.random.Generator | None = None,
        fire_rate: float | None = None,
        return_dx: bool = False,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """One update. With ``fire_rate == 1`` the RNG is not consumed (``rng`` may be None)."""
        c, h, w = x.shape
        fr = self.config.fire_rate if fire_rate is None else fire_rate
        dx = (self.w2 @ self.hidden_activations(x)).reshape(c, h, w)
        if fr < 1.0:
            if rng is None:
                raise ValueError("an rng is required when fire_rate < 1")
            dx = dx * (rng.random((h, w)) <= fr)
        new = x + dx
        living = self.alive(x[ALPHA_CHANNEL]) & self.alive(new[ALPHA_CHANNEL])
        out = (new * living).astype(np.float32)
        if return_dx:
            return out, dx.astype(np.float32)
        return out

    def run(
        self,
        x: FloatArray,
        steps: int,
        rng: np.random.Generator | None = None,
        damage_at: Iterable[int] = (),
        damage_rng: np.random.Generator | None = None,
    ) -> Iterator[FloatArray]:
        """Yield the state after each of ``steps`` updates.

        After update number ``t`` (1-based) for ``t`` in ``damage_at``, the state is multiplied by
        ``1 - circle_mask`` drawn from ``damage_rng`` (a stream separate from ``rng``, so that
        damage is paired across models); the damaged state is what gets yielded for step ``t``.
        """
        schedule = set(damage_at)
        if schedule and damage_rng is None:
            raise ValueError("damage_rng is required when damage_at is given")
        cur = np.asarray(x, dtype=np.float32)
        for t in range(1, steps + 1):
            nxt = self.step(cur, rng)
            assert not isinstance(nxt, tuple)
            cur = nxt
            if t in schedule:
                assert damage_rng is not None
                cur = (cur * (1.0 - circle_mask(cur.shape[-1], damage_rng))).astype(np.float32)
            yield cur
