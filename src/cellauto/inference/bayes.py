"""Exact Bayesian estimation of a life-like rule from one-step observations.

Each lookup-table entry ``L[s, n]`` is an independent Bernoulli parameter. If the observed output
bit is flipped with probability ``eps``, an entry seen with ``k1`` ones and ``k0`` zeros has
posterior log-odds ``(k1 - k0) * log((1 - eps) / eps)`` (uniform prior), so ``p = sigmoid(llr)``.
With ``eps = 0`` any contradiction is impossible and raises. Entries never observed keep
``p = 1/2``: they are unidentifiable from the data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cellauto._types import BinaryArray, Boundary
from cellauto._validate import as_binary
from cellauto.engines.lifelike import LifeRule, neighbour_counts


class InconsistentObservation(ValueError):
    """Noise-free observations that no life-like rule can explain."""


@dataclass(frozen=True)
class RulePosterior:
    p_one: npt.NDArray[np.float64]  # (2, 9): P(L[s, n] = 1 | data)
    counts: npt.NDArray[np.int64]  # (2, 9): observations per entry
    ones: npt.NDArray[np.int64]  # (2, 9): how many of them were 1

    def map_rule(self) -> LifeRule:
        """The most probable rule; unobserved entries (p = 1/2) are taken to be 0."""
        table = self.p_one > 0.5
        return LifeRule(
            frozenset(int(n) for n in np.flatnonzero(table[0])),
            frozenset(int(n) for n in np.flatnonzero(table[1])),
        )

    def unidentified(self) -> list[tuple[int, int]]:
        """``(state, count)`` entries with no observations."""
        return [(int(s), int(n)) for s, n in np.argwhere(self.counts == 0)]


def infer_rule(
    pairs: Iterable[tuple[BinaryArray, BinaryArray]],
    eps: float = 0.0,
    boundary: Boundary = "wrap",
) -> RulePosterior:
    """Posterior over the rule given ``(x0, x1)`` grid pairs, ``x1`` one step after ``x0``."""
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must be in [0, 0.5)")
    total = np.zeros(18, dtype=np.int64)
    ones = np.zeros(18, dtype=np.int64)
    for x0, x1 in pairs:
        a, b = as_binary(x0, 2), as_binary(x1, 2)
        if a.shape != b.shape:
            raise ValueError("x0 and x1 must have the same shape")
        idx = (a.astype(np.int64) * 9 + neighbour_counts(a, boundary)).ravel()
        total += np.bincount(idx, minlength=18)
        ones += np.bincount(idx, weights=b.ravel(), minlength=18).astype(np.int64)
    k1, k0 = ones.reshape(2, 9), (total - ones).reshape(2, 9)
    if eps == 0.0:
        if ((k1 > 0) & (k0 > 0)).any():
            bad = np.argwhere((k1 > 0) & (k0 > 0))[0]
            raise InconsistentObservation(
                f"entry (state={bad[0]}, count={bad[1]}) was seen both alive and dead"
            )
        p = np.where(k1 > 0, 1.0, np.where(k0 > 0, 0.0, 0.5))
    else:
        llr = (k1 - k0) * np.log((1.0 - eps) / eps)
        p = 0.5 * (1.0 + np.tanh(0.5 * llr))  # a sigmoid that cannot overflow
    return RulePosterior(p_one=p, counts=(k1 + k0).astype(np.int64), ones=k1.astype(np.int64))
