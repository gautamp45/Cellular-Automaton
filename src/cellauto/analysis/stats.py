"""Small statistics helpers shared by the analyses (numpy only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Interval:
    mean: float  # the point estimate (the statistic applied to the full sample)
    ci_low: float
    ci_high: float


def bootstrap_ci(
    x: npt.ArrayLike,
    stat: Callable[..., float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap confidence interval of ``stat`` (default: the mean).

    ``stat`` should accept ``axis=1`` for speed; otherwise it is called once per resample.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("x must be a non-empty 1-D array")
    rng = np.random.default_rng(seed)
    samples = a[rng.integers(0, a.size, size=(n_boot, a.size))]
    try:
        stats = np.asarray(stat(samples, axis=1), dtype=np.float64)
    except TypeError:
        stats = np.array([stat(row) for row in samples], dtype=np.float64)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return Interval(float(stat(a)), float(lo), float(hi))


def paired_permutation_test(
    a: npt.ArrayLike, b: npt.ArrayLike, n_perm: int = 100_000, seed: int = 0
) -> float:
    """Two-sided sign-flip permutation test on the mean of ``a - b``.

    Under the null the sign of each paired difference is exchangeable. Returns the Monte-Carlo
    p-value with the +1 correction, so it is never exactly 0.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    done = 0
    while done < n_perm:
        m = min(20_000, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(m, d.size))
        extreme += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).sum())
        done += m
    return (extreme + 1) / (n_perm + 1)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("need 0 <= successes <= n and n > 0")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def cohens_d_paired(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Paired Cohen's d (mean over sd of the differences); NaN if the differences are constant."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def holm_adjust(pvalues: npt.ArrayLike) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone), in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, float((p.size - rank) * p[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()
