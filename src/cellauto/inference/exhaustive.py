"""Exhaustive rule inference: simulate every one of the 2^18 life-like rules.

For life-like rules brute force is cheap and exact, so no learning is needed; machine learning
earns its place only when the rule space is too large to enumerate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from cellauto._types import BinaryArray, Boundary
from cellauto._validate import as_binary, check_int_range
from cellauto.engines.lifelike import neighbour_counts, tables_from_codes


def consistent_rules(
    x0: BinaryArray,
    xk: BinaryArray,
    k: int,
    boundary: Boundary = "wrap",
    chunk: int = 8192,
) -> npt.NDArray[np.int64]:
    """Codes (sorted) of every rule whose ``k``-step evolution of ``x0`` equals ``xk``."""
    steps = check_int_range("k", k, 1, 1000)
    a, target = as_binary(x0, 2), as_binary(xk, 2)
    if a.shape != target.shape:
        raise ValueError("x0 and xk must have the same shape")
    hits: list[npt.NDArray[np.int64]] = []
    for lo in range(0, 2**18, chunk):
        codes = np.arange(lo, min(lo + chunk, 2**18), dtype=np.int64)
        flat = tables_from_codes(codes).reshape(-1)
        base = (np.arange(len(codes), dtype=np.int64) * 18)[:, None, None]
        g = np.broadcast_to(a, (len(codes), *a.shape)).astype(np.uint8)
        for _ in range(steps):
            g = np.take(flat, base + g.astype(np.int64) * 9 + neighbour_counts(g, boundary))
        hits.append(codes[(g == target).all(axis=(1, 2))])
    return np.concatenate(hits)


def ambiguity(codes: npt.ArrayLike) -> dict[str, Any]:
    """How well do the observations pin the rule down?

    Returns the number of consistent rules, how many of the 18 table bits are identical in all of
    them, and the per-bit agreement ``max(P(bit = 1), P(bit = 0))``.
    """
    c = np.asarray(codes, dtype=np.int64)
    if c.size == 0:
        return {"n_consistent": 0, "bits_fixed": 0, "per_bit_agreement": [0.0] * 18}
    bits = (c[:, None] >> np.arange(18)) & 1
    p1 = bits.mean(axis=0)
    return {
        "n_consistent": int(c.size),
        "bits_fixed": int(((p1 == 0) | (p1 == 1)).sum()),
        "per_bit_agreement": np.maximum(p1, 1 - p1).tolist(),
    }
