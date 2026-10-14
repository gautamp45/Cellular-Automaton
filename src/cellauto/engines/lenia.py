"""Lenia (Chan 2019): a continuous-state, continuous-space-ish cellular automaton.

State ``A`` in [0, 1]^(H x W) on a torus. One step is

    U = K * A            (circular convolution, computed with FFT)
    A <- clip(A + G(U) / T, 0, 1)

with a ring-shaped kernel ``K`` (sum 1) and a growth mapping ``G``. Life is the special case with a
3x3 neighbour-count kernel, ``T = 1`` and a step-function growth (see :func:`life_as_lenia`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import numpy.typing as npt

from cellauto._types import FloatArray


@dataclass(frozen=True)
class LeniaParams:
    R: int = 13
    T: float = 10.0
    mu: float = 0.15
    sigma: float = 0.015
    peaks: tuple[float, ...] = (1.0,)  # 'b' in Chan's catalogue
    kernel_core: Literal["exp_bump", "life_box"] = "exp_bump"
    growth: Literal["gaussian", "life_step"] = "gaussian"

    def __post_init__(self) -> None:
        if isinstance(self.R, bool) or not isinstance(self.R, int) or self.R < 1:
            raise ValueError(f"R must be a positive int, got {self.R!r}")
        if not self.T > 0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if not self.sigma > 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        if not self.peaks or any(p < 0 for p in self.peaks):
            raise ValueError("peaks must be a non-empty sequence of non-negative numbers")
        if self.kernel_core not in ("exp_bump", "life_box"):
            raise ValueError(f"unknown kernel_core {self.kernel_core!r}")
        if self.growth not in ("gaussian", "life_step"):
            raise ValueError(f"unknown growth {self.growth!r}")


def _centred_radius(shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    """Distance from the origin cell, which sits at index ``(H // 2, W // 2)``."""
    h, w = shape
    y = np.arange(h)[:, None] - h // 2
    x = np.arange(w)[None, :] - w // 2
    return np.sqrt(x * x + y * y).astype(np.float64)


def _kernel64(params: LeniaParams, shape: tuple[int, int]) -> npt.NDArray[np.float64]:
    h, w = shape
    if params.kernel_core == "life_box":
        if min(h, w) < 3:
            raise ValueError("life_box kernel needs a grid of at least 3x3")
        k = np.zeros(shape, dtype=np.float64)
        k[h // 2 - 1 : h // 2 + 2, w // 2 - 1 : w // 2 + 2] = 1.0
        k[h // 2, w // 2] = 0.0
        return k
    if min(h, w) < 2 * params.R + 1:
        raise ValueError(f"grid {shape} is too small for kernel radius R={params.R}")
    r = _centred_radius(shape) / params.R
    n_peaks = len(params.peaks)
    br = r * n_peaks
    idx = np.minimum(br.astype(np.intp), n_peaks - 1)
    q = br % 1.0
    inside = (q > 0) & (q < 1)
    core = np.zeros_like(q)
    core[inside] = np.exp(4.0 - 1.0 / (q[inside] * (1.0 - q[inside])))
    k = np.where(r < 1, np.asarray(params.peaks, dtype=np.float64)[idx] * core, 0.0)
    total = k.sum()
    if total <= 0:
        raise ValueError("kernel is identically zero (check peaks)")
    return np.asarray(k / total, dtype=np.float64)


def kernel(params: LeniaParams, shape: tuple[int, int]) -> FloatArray:
    """Spatial kernel with its origin at ``(H // 2, W // 2)``.

    ``exp_bump``: ``K(x) = beta[floor(B r)] * exp(4 - 1/(q (1 - q)))``, ``q = B r mod 1``,
    ``r = |x| / R < 1``, normalised to sum 1. ``life_box``: 3x3 ones with a 0 centre (not
    normalised).
    """
    return _kernel64(params, shape).astype(np.float32)


@lru_cache(maxsize=32)
def _kernel_fft_cached(params: LeniaParams, shape: tuple[int, int]) -> npt.NDArray[np.complex128]:
    out = np.fft.fft2(np.fft.ifftshift(_kernel64(params, shape)))
    out.setflags(write=False)
    return out


def kernel_fft(params: LeniaParams, shape: tuple[int, int]) -> npt.NDArray[np.complex128]:
    """FFT of the origin-shifted kernel (float64), cached by ``(params, shape)``; read-only."""
    return _kernel_fft_cached(params, (int(shape[0]), int(shape[1])))


def growth(u: npt.ArrayLike, params: LeniaParams) -> FloatArray:
    """Gaussian: ``2 exp(-(u - mu)^2 / (2 sigma^2)) - 1``. Life step: +1 at 3, 0 at 2, else -1."""
    x = np.asarray(u, dtype=np.float64)
    if params.growth == "gaussian":
        g = 2.0 * np.exp(-((x - params.mu) ** 2) / (2.0 * params.sigma**2)) - 1.0
    else:
        n = np.rint(x)  # integer neighbour counts; FFT round-off is ~1e-12
        g = np.where(n == 3, 1.0, np.where(n == 2, 0.0, -1.0))
    return np.asarray(g, dtype=np.float32)


def step(
    A: npt.ArrayLike, params: LeniaParams, kfft: npt.NDArray[np.complex128] | None = None
) -> FloatArray:
    """One Lenia update. ``kfft`` may be passed to skip the cache lookup."""
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"A must be 2-D, got {a.ndim}-D")
    kf = kernel_fft(params, (a.shape[0], a.shape[1])) if kfft is None else kfft
    u = np.real(np.fft.ifft2(kf * np.fft.fft2(a)))
    nxt = np.clip(a + growth(u, params).astype(np.float64) / params.T, 0.0, 1.0)
    return np.asarray(nxt, dtype=np.float32)


def run(A: npt.ArrayLike, params: LeniaParams, steps: int) -> Iterator[FloatArray]:
    """Yield the state after each of ``steps`` updates (the initial state is not yielded)."""
    if isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a non-negative int")
    cur = np.asarray(A, dtype=np.float32)
    kf = kernel_fft(params, (cur.shape[0], cur.shape[1]))
    for _ in range(steps):
        cur = step(cur, params, kf)
        yield cur


def decode_lenia_rle(text: str) -> FloatArray:
    """Decode Chan's Lenia RLE into floats in [0, 1].

    ``'.'`` = 0; ``'A'..'X'`` = 1..24; two-character ``'pA'..'yO'`` where the prefix ``p..y``
    = 1..10 and value = ``prefix * 24 + (letter - 'A' + 1)``; run counts; ``'$'`` ends a row;
    ``'!'`` ends the pattern. Values are divided by 255.
    """
    rows: list[list[int]] = [[]]
    count = ""
    i = 0
    s = text.strip().rstrip("!")
    while i < len(s):
        ch = s[i]
        if ch.isdigit():
            count += ch
            i += 1
            continue
        k = int(count or 1)
        count = ""
        if ch == "$":
            rows += [[] for _ in range(k)]
            i += 1
            continue
        if ch == ".":
            rows[-1] += [0] * k
            i += 1
            continue
        if "p" <= ch <= "y":
            if i + 1 >= len(s) or not "A" <= s[i + 1] <= "X":
                raise ValueError(f"malformed two-character Lenia RLE value at {i}")
            v = (ord(ch) - ord("p") + 1) * 24 + (ord(s[i + 1]) - ord("A") + 1)
            i += 2
        elif "A" <= ch <= "X":
            v = ord(ch) - ord("A") + 1
            i += 1
        else:
            raise ValueError(f"unknown Lenia RLE token {ch!r}")
        rows[-1] += [v] * k
    width = max(map(len, rows))
    grid = np.array([r + [0] * (width - len(r)) for r in rows], dtype=np.float32)
    return np.asarray(grid / np.float32(255), dtype=np.float32)


def life_as_lenia() -> tuple[LeniaParams, FloatArray]:
    """Life as a special case of Lenia.

    Returns ``(params, kernel3x3)``: the kernel is a 3x3 array of ones with 0 at the centre (NOT
    normalised), ``T = 1`` and growth ``life_step``. With binary ``A`` this equals B3/S23 exactly.
    """
    params = LeniaParams(R=1, T=1.0, mu=0.0, sigma=1.0, peaks=(1.0,),
                         kernel_core="life_box", growth="life_step")  # fmt: skip
    k = np.ones((3, 3), dtype=np.float32)
    k[1, 1] = 0.0
    return params, k
