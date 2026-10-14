"""Small statistics helpers shared by the analyses (numpy only)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def shannon_entropy(counts: npt.ArrayLike) -> float:
    """Shannon entropy in bits of a vector of non-negative counts (0 log 0 = 0)."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-(p * np.log2(p)).sum())


def rankdata(a: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Ranks starting at 1, with ties given their average rank."""
    x = np.asarray(a)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    sorted_x = x[order]
    # boundaries of runs of equal values
    starts = np.flatnonzero(np.r_[True, sorted_x[1:] != sorted_x[:-1]])
    ends = np.r_[starts[1:], x.shape[0]]
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = (s + 1 + e) / 2.0
    return ranks


def spearman(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Spearman rank correlation (Pearson on average ranks); NaN if either side is constant."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
