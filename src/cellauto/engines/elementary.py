"""Elementary (1D, radius 1, two-state) cellular automata.

Wolfram convention: the neighbourhood ``(left, centre, right)`` is read as a 3-bit number
``(l << 2) | (c << 1) | r`` and the rule number's bit at that index is the new state.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cellauto._types import BinaryArray, Boundary
from cellauto._validate import as_binary, check_boundary, check_int_range


def rule_table(rule: int) -> BinaryArray:
    """Shape (8,). Index = (left << 2) | (centre << 1) | right ('111' is bit 7)."""
    r = check_int_range("rule", rule, 0, 255)
    return ((r >> np.arange(8)) & 1).astype(np.uint8)


def neighbourhood_index(row: BinaryArray, boundary: Boundary = "wrap") -> npt.NDArray[np.intp]:
    """``(l << 2) | (c << 1) | r`` for every cell.

    ``wrap`` uses a ring; ``dead`` treats cells outside the row as 0.
    """
    check_boundary(boundary)
    a = as_binary(row, 1).astype(np.intp)
    if boundary == "wrap":
        left, right = np.roll(a, 1), np.roll(a, -1)
    else:
        left = np.concatenate(([0], a[:-1])).astype(np.intp)
        right = np.concatenate((a[1:], [0])).astype(np.intp)
    return (left << 2) | (a << 1) | right


def _step_with_table(row: BinaryArray, table: BinaryArray, boundary: Boundary) -> BinaryArray:
    return table[neighbourhood_index(row, boundary)]


def step(row: BinaryArray, rule: int, boundary: Boundary = "wrap") -> BinaryArray:
    """One generation. Every cell updates, edges included."""
    return _step_with_table(row, rule_table(rule), boundary)


def run(
    initial: BinaryArray, rule: int, generations: int, boundary: Boundary = "wrap"
) -> BinaryArray:
    """Space-time diagram of shape ``(generations + 1, width)``; row 0 is ``initial``."""
    gens = check_int_range("generations", generations, 0, 2**31 - 1)
    table = rule_table(rule)
    first = as_binary(initial, 1)
    check_boundary(boundary)
    out = np.empty((gens + 1, first.shape[0]), dtype=np.uint8)
    out[0] = first
    for t in range(gens):
        out[t + 1] = _step_with_table(out[t], table, boundary)
    return out


def single_cell(width: int) -> BinaryArray:
    """All zeros except a 1 at ``width // 2``."""
    w = check_int_range("width", width, 1, 2**31 - 1)
    row = np.zeros(w, dtype=np.uint8)
    row[w // 2] = 1
    return row


def random_row(width: int, density: float = 0.5, seed: int | None = None) -> BinaryArray:
    """Bernoulli(density) row from ``default_rng(seed)``."""
    w = check_int_range("width", width, 1, 2**31 - 1)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"density must be in [0, 1], got {density}")
    return (np.random.default_rng(seed).random(w) < density).astype(np.uint8)


def from_bits(bits: str) -> BinaryArray:
    """``"0110"`` -> array. ValueError on any other character."""
    if not bits or any(ch not in "01" for ch in bits):
        raise ValueError(f"bits must be a non-empty string of 0/1, got {bits!r}")
    return np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")


def to_bits(row: BinaryArray) -> str:
    return "".join("1" if v else "0" for v in as_binary(row, 1))
