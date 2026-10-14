"""Mathematics of life-like rules: duality, Langton's lambda, mean-field theory."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Literal

import numpy as np
import numpy.typing as npt

from cellauto.engines.lifelike import LifeRule, random_grid, step_batch

_BINOM = np.array([comb(8, n) for n in range(9)], dtype=np.float64)


# --- 1. black/white duality ---------------------------------------------------------------------


def dual(rule: LifeRule) -> LifeRule:
    """The rule obtained by inverting every cell.

    ``B' = {8 - n : n not in S}`` and ``S' = {8 - n : n not in B}``.
    """
    return LifeRule(
        birth=frozenset(8 - n for n in range(9) if n not in rule.survival),
        survival=frozenset(8 - n for n in range(9) if n not in rule.birth),
    )


def is_self_dual(rule: LifeRule) -> bool:
    return dual(rule) == rule


def dual_codes(codes: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Vectorised :func:`dual` on 18-bit rule codes."""
    c = np.asarray(codes, dtype=np.int64)
    n = np.arange(9)
    birth = (c[:, None] >> n) & 1
    surv = (c[:, None] >> (9 + n)) & 1
    new_birth = 1 - surv[:, ::-1]  # n in B'  iff  8 - n not in S
    new_surv = 1 - birth[:, ::-1]
    return np.asarray((new_birth << n).sum(axis=1) | ((new_surv << n).sum(axis=1) << 9))


# --- 2. Langton's lambda ------------------------------------------------------------------------


def lambda_(rule: LifeRule) -> float:
    """Probability that a uniformly random 3x3 neighbourhood maps to 1."""
    births = sum(comb(8, n) for n in rule.birth)
    survivals = sum(comb(8, n) for n in rule.survival)
    return (births + survivals) / 512.0


def lambda_codes(codes: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    c = np.asarray(codes, dtype=np.int64)
    n = np.arange(9)
    bits = np.concatenate([(c[:, None] >> n) & 1, (c[:, None] >> (9 + n)) & 1], axis=1)
    return np.asarray(bits @ np.tile(_BINOM, 2) / 512.0)


# --- 3. mean-field theory -----------------------------------------------------------------------


def mean_field_map(rho: float | npt.ArrayLike, rule: LifeRule) -> float | npt.NDArray[np.float64]:
    """``f(rho)``: next density if cells are independent with density ``rho``.

    The live-neighbour count is then Binomial(8, rho).
    """
    r = np.asarray(rho, dtype=np.float64)
    p = _BINOM * r[..., None] ** np.arange(9) * (1.0 - r[..., None]) ** (8 - np.arange(9))
    s = p[..., sorted(rule.survival)].sum(axis=-1)
    b = p[..., sorted(rule.birth)].sum(axis=-1)
    out = r * s + (1.0 - r) * b
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class FixedPoint:
    rho: float
    derivative: float
    stability: Literal["stable", "unstable", "marginal"]


def _g(rho: float | npt.NDArray[np.float64], rule: LifeRule) -> npt.NDArray[np.float64]:
    return np.asarray(mean_field_map(rho, rule)) - np.asarray(rho)


def fixed_points(rule: LifeRule) -> list[FixedPoint]:
    """Roots of ``f(rho) - rho`` on [0, 1] with their stability.

    Sign changes on a 20,001-point grid are refined by 60 bisection steps; exact zeros on the grid
    are kept; roots closer than 1e-3 are merged. ``|f'| < 1 - 1e-3`` is stable, ``> 1 + 1e-3`` is
    unstable and anything between is marginal (near-degenerate roots, e.g. Coral).
    """
    grid = np.linspace(0.0, 1.0, 20001)
    g = _g(grid, rule)
    roots: list[float] = [float(grid[i]) for i in np.flatnonzero(g == 0.0)]
    for i in np.flatnonzero(g[:-1] * g[1:] < 0):
        lo, hi = float(grid[i]), float(grid[i + 1])
        glo = float(g[i])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            gm = float(_g(np.float64(mid), rule))
            if (gm < 0) == (glo < 0):
                lo, glo = mid, gm
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    merged: list[float] = []
    for r in sorted(roots):
        if not merged or r - merged[-1] > 1e-3:
            merged.append(r)
    h = 1e-6
    points = []
    for r in merged:
        lo, hi = max(r - h, 0.0), min(r + h, 1.0)
        slope = (float(mean_field_map(hi, rule)) - float(mean_field_map(lo, rule))) / (hi - lo)
        if abs(slope) < 1 - 1e-3:
            stab: Literal["stable", "unstable", "marginal"] = "stable"
        elif abs(slope) > 1 + 1e-3:
            stab = "unstable"
        else:
            stab = "marginal"
        points.append(FixedPoint(rho=r, derivative=slope, stability=stab))
    return points


def iterate_mean_field(rule: LifeRule, rho0: float = 0.5, steps: int = 1000) -> float:
    rho = float(rho0)
    for _ in range(steps):
        rho = float(mean_field_map(rho, rule))
    return rho


def mean_field_codes(
    codes: npt.NDArray[np.int64], rho0: float = 0.5, steps: int = 1000
) -> npt.NDArray[np.float64]:
    """Vectorised :func:`iterate_mean_field` over many rule codes."""
    c = np.asarray(codes, dtype=np.int64)
    n = np.arange(9)
    birth = ((c[:, None] >> n) & 1).astype(np.float64)
    surv = ((c[:, None] >> (9 + n)) & 1).astype(np.float64)
    rho = np.full(c.shape[0], float(rho0))
    for _ in range(steps):
        p = _BINOM * rho[:, None] ** n * (1.0 - rho[:, None]) ** (8 - n)
        rho = rho * (p * surv).sum(axis=1) + (1.0 - rho) * (p * birth).sum(axis=1)
    return rho


def simulated_density(
    rule: LifeRule, size: int = 127, steps: int = 400, seeds: int = 3, window: int = 50
) -> float:
    """Mean density over the last ``window`` steps of ``steps`` steps, averaged over ``seeds`` runs.

    A ``size x size`` torus from a random start with density 0.5. The size is deliberately not a
    power of two: Replicator (B1357/S1357) is linear over GF(2) and dies on such tori.
    """
    grids = np.stack([random_grid(size, size, 0.5, seed) for seed in range(seeds)])
    tables = np.broadcast_to(rule.table(), (seeds, 2, 9))
    tail = []
    for t in range(1, steps + 1):
        grids = step_batch(grids, tables)
        if t > steps - window:
            tail.append(grids.mean(axis=(1, 2)))
    return float(np.mean(tail))


def mean_field_report(rule: LifeRule) -> dict[str, object]:
    """Fixed points, the simulated density and the gap ``|simulated - mean-field|`` for a rule."""
    sim = simulated_density(rule)
    mf = iterate_mean_field(rule)
    return {
        "fixed_points": [
            {"rho": fp.rho, "derivative": fp.derivative, "stability": fp.stability}
            for fp in fixed_points(rule)
        ],
        "simulated_density": sim,
        "correlation_gap": abs(sim - mf),
    }
