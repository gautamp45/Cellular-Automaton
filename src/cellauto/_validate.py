"""Small input validators shared by the engines."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cellauto._types import BinaryArray

_BOUNDARIES = ("wrap", "dead")


def as_binary(arr: npt.ArrayLike, ndim: int) -> BinaryArray:
    """Return a uint8 copy of ``arr``; ValueError unless values are in {0, 1} and ndim matches."""
    a = np.asarray(arr)
    if a.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got {a.ndim}-D")
    if a.size and not np.isin(a, (0, 1)).all():
        raise ValueError("array values must be in {0, 1}")
    return a.astype(np.uint8, copy=True)


def check_boundary(b: str) -> None:
    if b not in _BOUNDARIES:
        raise ValueError(f"boundary must be one of {_BOUNDARIES}, got {b!r}")


def check_int_range(name: str, v: int, lo: int, hi: int) -> int:
    """Validate ``lo <= v <= hi`` for a real int (bool is rejected)."""
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{name} must be an int, got {type(v).__name__}")
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}")
    return int(v)
