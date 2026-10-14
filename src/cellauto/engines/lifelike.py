"""Life-like (outer-totalistic, Moore neighbourhood, two-state) 2D cellular automata.

Rules are table-driven: ``lut[state, neighbour_count]`` with a ``(2, 9)`` LUT, so one vectorised
fancy-index step covers every B/S rule. Every rule also has an 18-bit integer code:
bit ``n`` (0..8) = birth on ``n`` neighbours, bit ``9 + n`` = survival on ``n`` neighbours.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from cellauto._types import BinaryArray, Boundary
from cellauto._validate import as_binary, check_boundary, check_int_range

_BS = re.compile(r"B([0-9]*)/?S([0-9]*)")
_SB = re.compile(r"S([0-9]*)/?B([0-9]*)")
_LEGACY = re.compile(r"[0-9]*/[0-9]*")


@dataclass(frozen=True)
class LifeRule:
    birth: frozenset[int]
    survival: frozenset[int]

    def __post_init__(self) -> None:
        for name, counts in (("birth", self.birth), ("survival", self.survival)):
            if any(not 0 <= n <= 8 for n in counts):
                raise ValueError(f"{name} counts must be in 0..8, got {sorted(counts)}")

    @classmethod
    def parse(cls, text: str) -> LifeRule:
        """Parse B/S notation, case-insensitive, optional '/', either order.

        ``'B3/S23'``, ``'b3s23'``, ``'S23/B3'``, ``'B2/S'``, ``'B/S'``. Digits 0-8 only.
        """
        s = "".join(text.split()).upper()
        if _LEGACY.fullmatch(s) and s:
            raise ValueError(f"legacy 'survival/birth' notation {text!r} is not supported; use B/S")
        if m := _BS.fullmatch(s):
            b, sv = m.group(1), m.group(2)
        elif m := _SB.fullmatch(s):
            sv, b = m.group(1), m.group(2)
        else:
            raise ValueError(f"cannot parse rulestring {text!r}; expected e.g. 'B3/S23'")
        if "9" in b + sv:
            raise ValueError(f"neighbour counts must be 0-8 in {text!r}")
        return cls(frozenset(int(c) for c in b), frozenset(int(c) for c in sv))

    @classmethod
    def from_code(cls, code: int) -> LifeRule:
        """Inverse of :attr:`code`."""
        c = check_int_range("code", code, 0, 2**18 - 1)
        return cls(
            frozenset(n for n in range(9) if (c >> n) & 1),
            frozenset(n for n in range(9) if (c >> (9 + n)) & 1),
        )

    @property
    def code(self) -> int:
        return sum(1 << n for n in self.birth) | sum(1 << (9 + n) for n in self.survival)

    @property
    def rulestring(self) -> str:
        """Canonical ``'B3/S23'``."""
        b = "".join(str(n) for n in sorted(self.birth))
        s = "".join(str(n) for n in sorted(self.survival))
        return f"B{b}/S{s}"

    def table(self) -> BinaryArray:
        """Shape (2, 9): ``lut[0, n]`` birth, ``lut[1, n]`` survival."""
        lut = np.zeros((2, 9), dtype=np.uint8)
        lut[0, sorted(self.birth)] = 1
        lut[1, sorted(self.survival)] = 1
        return lut


def _as_rule(rule: LifeRule | str) -> LifeRule:
    return LifeRule.parse(rule) if isinstance(rule, str) else rule


def neighbour_counts(grid: BinaryArray, boundary: Boundary = "wrap") -> BinaryArray:
    """Moore-neighbourhood live counts (uint8) over the last two axes.

    Works on a batch ``(..., H, W)``. ``wrap`` is a torus; ``dead`` treats outside as 0.
    """
    check_boundary(boundary)
    g = np.asarray(grid, dtype=np.uint8)
    if g.ndim < 2:
        raise ValueError("grid must have at least 2 dimensions")
    counts = np.zeros(g.shape, dtype=np.uint8)
    if boundary == "wrap":
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    counts += np.roll(g, (dy, dx), axis=(-2, -1))
    else:
        pad = [(0, 0)] * (g.ndim - 2) + [(1, 1), (1, 1)]
        p = np.pad(g, pad)
        h, w = g.shape[-2:]
        for dy in range(3):
            for dx in range(3):
                if dy != 1 or dx != 1:
                    counts += p[..., dy : dy + h, dx : dx + w]
    return counts


def step(grid: BinaryArray, rule: LifeRule | str, boundary: Boundary = "wrap") -> BinaryArray:
    """One generation: ``rule.table()[grid, neighbour_counts(grid, boundary)]``."""
    g = as_binary(grid, 2)
    lut = _as_rule(rule).table()
    return lut[g, neighbour_counts(g, boundary)]


def step_batch(grids: BinaryArray, tables: BinaryArray, boundary: Boundary = "wrap") -> BinaryArray:
    """Step ``R`` grids ``(R, H, W)`` each under its own ``(2, 9)`` table ``tables[r]``."""
    g = as_binary(grids, 3)
    t = np.asarray(tables, dtype=np.uint8)
    if t.shape != (g.shape[0], 2, 9):
        raise ValueError(f"tables must have shape ({g.shape[0]}, 2, 9), got {t.shape}")
    counts = neighbour_counts(g, boundary)
    return t[np.arange(g.shape[0])[:, None, None], g, counts]


def iterate(
    grid: BinaryArray, rule: LifeRule | str, boundary: Boundary = "wrap"
) -> Iterator[BinaryArray]:
    """Infinite generator of successive states (the first item is the initial grid)."""
    r = _as_rule(rule)
    cur = as_binary(grid, 2)
    while True:
        yield cur
        cur = step(cur, r, boundary)


def run(
    grid: BinaryArray, rule: LifeRule | str, steps: int, boundary: Boundary = "wrap"
) -> BinaryArray:
    """History of shape ``(steps + 1, H, W)``; index 0 is the initial grid."""
    n = check_int_range("steps", steps, 0, 2**31 - 1)
    r = _as_rule(rule)
    first = as_binary(grid, 2)
    out = np.empty((n + 1, *first.shape), dtype=np.uint8)
    out[0] = first
    for t in range(n):
        out[t + 1] = step(out[t], r, boundary)
    return out


def random_grid(
    height: int, width: int, density: float = 0.5, seed: int | None = None
) -> BinaryArray:
    """Bernoulli(density) grid from ``default_rng(seed)``."""
    h = check_int_range("height", height, 1, 2**31 - 1)
    w = check_int_range("width", width, 1, 2**31 - 1)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    return (np.random.default_rng(seed).random((h, w)) < density).astype(np.uint8)
