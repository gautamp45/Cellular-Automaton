"""Mathematics of the 256 elementary cellular automata.

Symmetry (a Klein four-group acting on rule space), Langton's lambda, surjectivity and
injectivity from the de Bruijn graph, preimage counting by transfer matrices, Garden-of-Eden
counts, and exact GF(2) results for additive rules.
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt

from cellauto._types import BinaryArray
from cellauto._validate import as_binary, check_int_range
from cellauto.engines.elementary import rule_table

# --- 1. symmetry --------------------------------------------------------------------------------


def reflect(rule: int) -> int:
    """Swap left and right: output bit ``i`` takes the value of the mirrored neighbourhood."""
    t = rule_table(rule)
    out = 0
    for i in range(8):
        left, centre, right = (i >> 2) & 1, (i >> 1) & 1, i & 1
        out |= int(t[(right << 2) | (centre << 1) | left]) << i
    return out


def complement(rule: int) -> int:
    """Swap the roles of 0 and 1: output bit ``i`` is ``1 - table[7 - i]``."""
    t = rule_table(rule)
    out = 0
    for i in range(8):
        out |= (1 - int(t[7 - i])) << i
    return out


def orbit(rule: int) -> list[int]:
    """Sorted ``{r, reflect(r), complement(r), reflect(complement(r))}``."""
    return sorted({rule, reflect(rule), complement(rule), reflect(complement(rule))})


def equivalence_classes() -> dict[int, list[int]]:
    """Orbits of the Klein four-group, keyed by their smallest rule (88 classes)."""
    classes: dict[int, list[int]] = {}
    for n in range(256):
        o = orbit(n)
        classes[o[0]] = o
    return classes


# --- 2. Langton's lambda ------------------------------------------------------------------------


def lambda_(rule: int) -> float:
    """Fraction of the 8 neighbourhoods that map to 1."""
    return float(rule_table(rule).sum()) / 8.0


# --- 3. surjectivity, injectivity, preimages ----------------------------------------------------


def is_surjective(rule: int) -> bool:
    """Surjective on the infinite line (Hedlund; Amoroso-Patt subset construction).

    States are pairs (a, b); reading output bit ``y``, (a, b) -> (b, c) iff ``f(a, b, c) = y``.
    Starting from the set of all four states, the rule is surjective iff the empty set is
    unreachable.
    """
    t = rule_table(rule)
    start = frozenset(range(4))
    seen = {start}
    stack = [start]
    while stack:
        s = stack.pop()
        for y in (0, 1):
            nxt = frozenset(
                ((ab << 1) & 3) | c for ab in s for c in (0, 1) if t[(ab << 1) | c] == y
            )
            if not nxt:
                return False
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return True


def surjective_rules() -> list[int]:
    return [n for n in range(256) if is_surjective(n)]


def is_injective(rule: int) -> bool:
    """Injective on the infinite line (hence reversible), via the pair graph (Amoroso-Patt).

    Nodes are pairs of de Bruijn states ``((a, b), (a', b'))``; there is an edge to
    ``((b, c), (b', c'))`` iff ``f(a, b, c) = f(a', b', c')``. Two distinct bi-infinite
    configurations with the same image correspond to a bi-infinite path through an off-diagonal
    node, so ``rule`` is injective iff no off-diagonal node survives repeatedly pruning nodes that
    have no incoming or no outgoing edge.
    """
    t = rule_table(rule)
    edges: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for u, v in itertools.product(range(4), repeat=2):
        for c, d in itertools.product((0, 1), repeat=2):
            if t[(u << 1) | c] == t[(v << 1) | d]:
                edges.setdefault((u, v), set()).add((((u << 1) & 3) | c, ((v << 1) & 3) | d))
    alive = {(u, v) for u in range(4) for v in range(4)}
    changed = True
    while changed:
        changed = False
        has_in = {w for n in alive for w in edges.get(n, ()) if w in alive}
        for n in list(alive):
            has_out = any(w in alive for w in edges.get(n, ()))
            if not has_out or n not in has_in:
                alive.discard(n)
                changed = True
    return not any(u != v for u, v in alive)


def is_injective_on_rings(rule: int, max_n: int = 10) -> bool:
    """Exhaustively check injectivity on cyclic configurations of every length 1..``max_n``."""
    t = rule_table(rule)
    for n in range(1, max_n + 1):
        codes = np.arange(2**n, dtype=np.int64)
        bits = ((codes[:, None] >> np.arange(n)) & 1).astype(np.intp)
        idx = (np.roll(bits, 1, axis=1) << 2) | (bits << 1) | np.roll(bits, -1, axis=1)
        images = (t[idx].astype(np.int64) << np.arange(n)).sum(axis=1)
        if np.unique(images).size != codes.size:
            return False
    return True


def reversible_rules(max_n: int = 10) -> list[int]:
    """Surjective rules that are injective on all rings of length 1..``max_n``.

    Only surjective rules can be reversible, so the (cheap) ring test is applied to those. The
    exact infinite-line criterion is :func:`is_injective`; the two agree for the 256 rules.
    """
    return [n for n in surjective_rules() if is_injective_on_rings(n, max_n)]


def transfer_matrix(rule: int, y: int) -> npt.NDArray[np.int64]:
    """4x4 matrix ``M_y[(a, b), (b, c)] = 1`` iff ``f(a, b, c) = y``."""
    if y not in (0, 1):
        raise ValueError("y must be 0 or 1")
    t = rule_table(rule)
    m = np.zeros((4, 4), dtype=np.int64)
    for a, b, c in itertools.product((0, 1), repeat=3):
        if t[(a << 2) | (b << 1) | c] == y:
            m[(a << 1) | b, (b << 1) | c] = 1
    return m


def _as_config(config: str | BinaryArray | list[int]) -> npt.NDArray[np.uint8]:
    if isinstance(config, str):
        if not config or any(ch not in "01" for ch in config):
            raise ValueError(f"config must be a non-empty 0/1 string, got {config!r}")
        return np.frombuffer(config.encode("ascii"), dtype=np.uint8) - ord("0")
    return as_binary(config, 1)


def preimage_count(rule: int, config: str | BinaryArray | list[int]) -> int:
    """Number of ring configurations that map to ``config`` in one step: ``trace(M_y1 ... M_yN)``.

    Exact for any length (Python integers are used beyond 60 cells to avoid overflow).
    """
    y = _as_config(config)
    mats = (transfer_matrix(rule, 0), transfer_matrix(rule, 1))
    if len(y) > 60:
        prod = [[int(i == j) for j in range(4)] for i in range(4)]
        for bit in y:
            m = mats[int(bit)]
            prod = [
                [sum(prod[i][k] * int(m[k, j]) for k in range(4)) for j in range(4)]
                for i in range(4)
            ]
        return sum(prod[i][i] for i in range(4))
    p = np.eye(4, dtype=np.int64)
    for bit in y:
        p = p @ mats[int(bit)]
    return int(np.trace(p))


def garden_of_eden_count(rule: int, n: int) -> int:
    """Ring configurations of length ``n`` with no preimage (``2^n`` minus the image size)."""
    n = check_int_range("n", n, 1, 24)
    t = rule_table(rule)
    codes = np.arange(2**n, dtype=np.int64)
    bits = ((codes[:, None] >> np.arange(n)) & 1).astype(np.intp)
    idx = (np.roll(bits, 1, axis=1) << 2) | (bits << 1) | np.roll(bits, -1, axis=1)
    images = (t[idx].astype(np.int64) << np.arange(n)).sum(axis=1)
    return int(2**n - np.unique(images).size)


# --- 4. additive rules over GF(2) ---------------------------------------------------------------


def additive_coefficients(rule: int) -> tuple[int, int, int]:
    """``(a, b, d)`` with ``f(l, c, r) = a*l xor b*c xor d*r``; ValueError if not additive."""
    t = rule_table(rule)
    a, b, d = int(t[4]), int(t[2]), int(t[1])
    for i in range(8):
        left, centre, right = (i >> 2) & 1, (i >> 1) & 1, i & 1
        if int(t[i]) != ((a & left) ^ (b & centre) ^ (d & right)):
            raise ValueError(f"rule {rule} is not additive (XOR of its neighbours)")
    return a, b, d


def gf2_transition_matrix(rule: int, n: int) -> npt.NDArray[np.uint8]:
    """The linear map of an additive rule on a ring of ``n`` cells, as an ``n x n`` 0/1 matrix."""
    a, b, d = additive_coefficients(rule)
    n = check_int_range("n", n, 1, 4096)
    m = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        m[i, (i - 1) % n] ^= a
        m[i, i] ^= b
        m[i, (i + 1) % n] ^= d
    return m


def gf2_rank(matrix: npt.ArrayLike) -> int:
    """Rank over GF(2) by Gaussian elimination (rows packed as Python ints)."""
    m = np.asarray(matrix, dtype=np.uint8) & 1
    rows = [int("".join(map(str, r)), 2) for r in m]
    rank = 0
    while rows:
        pivot = rows.pop()
        if pivot:
            rank += 1
            low = pivot & -pivot
            rows = [r ^ pivot if r & low else r for r in rows]
    return rank


def gf2_matrix_power(matrix: npt.ArrayLike, power: int) -> npt.NDArray[np.uint8]:
    """``matrix ** power`` over GF(2) by repeated squaring."""
    base = np.asarray(matrix, dtype=np.int64) & 1
    result = np.eye(base.shape[0], dtype=np.int64)
    p = check_int_range("power", power, 0, 2**31 - 1)
    while p:
        if p & 1:
            result = (result @ base) & 1
        base = (base @ base) & 1
        p >>= 1
    return result.astype(np.uint8)


def rule90_closed_form(t: int, width: int) -> BinaryArray:
    """Row ``t`` of rule 90 from a single centre cell, with no simulation.

    Cell ``x`` (relative to the centre) is 1 iff ``|x| <= t``, ``t + x`` is even, and with
    ``k = (t + x) / 2`` we have ``k & (t - k) == 0`` (Lucas' theorem: C(t, k) is odd iff the bits
    of k are a subset of the bits of t). Valid while the pattern has not wrapped, i.e. for
    ``t <= width // 2``.
    """
    t = check_int_range("t", t, 0, 2**31 - 1)
    width = check_int_range("width", width, 1, 2**31 - 1)
    if t > width // 2:
        raise ValueError(f"closed form is only valid for t <= width // 2 = {width // 2}")
    row = np.zeros(width, dtype=np.uint8)
    centre = width // 2
    for x in range(-t, t + 1):
        if (t + x) % 2 == 0:
            k = (t + x) // 2
            if k & (t - k) == 0:
                row[centre + x] = 1
    return row


__all__ = [
    "additive_coefficients",
    "complement",
    "equivalence_classes",
    "garden_of_eden_count",
    "gf2_matrix_power",
    "gf2_rank",
    "gf2_transition_matrix",
    "is_injective",
    "is_injective_on_rings",
    "is_surjective",
    "lambda_",
    "orbit",
    "preimage_count",
    "reflect",
    "reversible_rules",
    "rule90_closed_form",
    "surjective_rules",
    "transfer_matrix",
]
