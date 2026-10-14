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
