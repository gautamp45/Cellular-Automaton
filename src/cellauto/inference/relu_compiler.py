"""Compile any life-like rule into an exact ReLU network.

Claim (a theorem, checked by tests): every life-like rule is computed exactly by
``Conv3x3(1 -> 1, all-ones kernel with centre weight 1/2) -> ReLU units -> linear readout``.

Construction. ``v = n + s/2`` (``n`` live neighbours, ``s`` the cell state) takes the 18 distinct
values ``v_k = k / 2``. Even ``k``: ``s = 0, n = k / 2`` (a birth entry); odd ``k``: ``s = 1,
n = (k - 1) / 2`` (a survival entry). With ``y_k = L[s, n]`` the piecewise-linear interpolant is

    g(v) = y_0 + sum_{j=0}^{16} c_j * ReLU(v - j/2),   m_j = (y_{j+1} - y_j) / 0.5,
    c_0 = m_0,   c_j = m_j - m_{j-1}.

Hidden units are the nonzero ``c_j``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cellauto._types import BinaryArray, Boundary
from cellauto.engines.lifelike import LifeRule, neighbour_counts

CONSTRUCTION = (
    "v = n + s/2 (n live neighbours, s the cell state) takes the 18 values k/2; the table entry "
    "y_k is L[0, k/2] for even k and L[1, (k-1)/2] for odd k. The piecewise-linear interpolant is "
    "g(v) = y_0 + sum_j c_j ReLU(v - j/2) with c_0 = m_0, c_j = m_j - m_(j-1) and slopes "
    "m_j = (y_(j+1) - y_j) / 0.5; hidden units are the nonzero c_j."
)


@dataclass(frozen=True)
class ReluNet:
    centre_weight: float
    biases: npt.NDArray[np.float64]  # hidden unit i computes ReLU(v + biases[i])
    out_weights: npt.NDArray[np.float64]
    out_bias: float

    @property
    def hidden_units(self) -> int:
        return int(self.biases.shape[0])


def _y_values(rule: LifeRule) -> npt.NDArray[np.float64]:
    table = rule.table()
    return np.array([table[k % 2, k // 2] for k in range(18)], dtype=np.float64)


def compile_rule(rule: LifeRule) -> ReluNet:
    y = _y_values(rule)
    slopes = np.diff(y) / 0.5  # m_0 .. m_16
    c = np.concatenate([[slopes[0]], np.diff(slopes)])  # c_0 .. c_16
    keep = np.flatnonzero(c != 0)
    return ReluNet(
        centre_weight=0.5,
        biases=-(keep / 2.0),
        out_weights=c[keep],
        out_bias=float(y[0]),
    )


def apply(net: ReluNet, grid: BinaryArray, boundary: Boundary = "wrap") -> npt.NDArray[np.float64]:
    """Run the network on a 0/1 grid; the result is exactly 0.0 or 1.0 everywhere."""
    g = np.asarray(grid, dtype=np.float64)
    v = neighbour_counts(grid, boundary).astype(np.float64) + net.centre_weight * g
    hidden = np.maximum(v[..., None] + net.biases, 0.0)
    return np.asarray(net.out_bias + hidden @ net.out_weights, dtype=np.float64)


def hidden_unit_counts(codes: npt.ArrayLike | None = None) -> npt.NDArray[np.int64]:
    """Hidden units needed per rule code (default: all 2^18 rules), vectorised."""
    c = np.arange(2**18, dtype=np.int64) if codes is None else np.asarray(codes, dtype=np.int64)
    k = np.arange(18)
    shift = np.where(k % 2 == 0, k // 2, 9 + k // 2)  # birth bit for even k, survival bit for odd k
    y = ((c[:, None] >> shift) & 1).astype(np.int64)
    diff = np.diff(y, axis=1)  # proportional to the slopes m_j
    changes = np.concatenate([diff[:, :1], np.diff(diff, axis=1)], axis=1)
    return np.asarray((changes != 0).sum(axis=1), dtype=np.int64)


def hidden_unit_histogram() -> list[int]:
    """Number of rules (out of 2^18) needing 0, 1, ..., 17 hidden units."""
    return np.bincount(hidden_unit_counts(), minlength=18).tolist()
