import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cellauto.engines.lifelike import (
    LifeRule,
    neighbour_counts,
    random_grid,
    run,
    step,
    step_batch,
)


def test_parse_variants() -> None:
    assert LifeRule.parse("b3s23").rulestring == "B3/S23"
    assert LifeRule.parse("S23/B3") == LifeRule.parse("B3/S23")
    assert LifeRule.parse("B2/S").survival == frozenset()
    assert LifeRule.parse("B/S") == LifeRule(frozenset(), frozenset())


@pytest.mark.parametrize("bad", ["B9/S23", "23/3", "hello", "", "B3"])
def test_parse_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        LifeRule.parse(bad)


def test_table_life() -> None:
    assert LifeRule.parse("B3/S23").table().tolist() == [
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
    ]


@given(st.integers(0, 2**18 - 1))
def test_code_roundtrip(code: int) -> None:
    r = LifeRule.from_code(code)
    assert r.code == code
    assert LifeRule.parse(r.rulestring) == r


def test_counts_regression_b1() -> None:
    g = np.zeros((5, 5), dtype=np.uint8)
    g[0, 0] = g[0, 1] = g[1, 0] = 1
    assert neighbour_counts(g, "dead")[0, 1] == 2
    assert (neighbour_counts(g, "dead") <= 8).all()
    one = np.zeros((5, 5), dtype=np.uint8)
    one[0, 0] = 1
    assert neighbour_counts(one, "wrap")[4, 4] == 1


def test_step_batch_matches_step() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 2**18, 16)
    rules = [LifeRule.from_code(int(c)) for c in codes]
    grids = (rng.random((16, 9, 11)) < 0.4).astype(np.uint8)
    tables = np.stack([r.table() for r in rules])
    for boundary in ("wrap", "dead"):
        got = step_batch(grids, tables, boundary)  # type: ignore[arg-type]
        for i, r in enumerate(rules):
            assert np.array_equal(got[i], step(grids[i], r, boundary))  # type: ignore[arg-type]


def _slow(grid: list[list[int]], rule: LifeRule, boundary: str) -> list[list[int]]:
    h, w = len(grid), len(grid[0])
    out = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            n = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == dj == 0:
                        continue
                    a, b = i + di, j + dj
                    if boundary == "wrap":
                        n += grid[a % h][b % w]
                    elif 0 <= a < h and 0 <= b < w:
                        n += grid[a][b]
            alive = grid[i][j]
            out[i][j] = int(n in (rule.survival if alive else rule.birth))
    return out


@given(
    st.integers(0, 2**18 - 1),
    st.integers(1, 12),
    st.integers(1, 12),
    st.integers(0, 2**32 - 1),
    st.sampled_from(["wrap", "dead"]),
)
def test_step_matches_slow(code: int, h: int, w: int, seed: int, boundary: str) -> None:
    rule = LifeRule.from_code(code)
    g = random_grid(h, w, 0.5, seed)
    assert step(g, rule, boundary).tolist() == _slow(g.tolist(), rule, boundary)  # type: ignore[arg-type]


def test_run_shape_and_errors() -> None:
    g = random_grid(6, 7, seed=1)
    assert run(g, "B3/S23", 4).shape == (5, 6, 7)
    assert np.array_equal(run(g, "B3/S23", 0)[0], g)
    with pytest.raises(ValueError):
        run(g, "B3/S23", -1)
    with pytest.raises(ValueError):
        step(g, "B3/S23", "torus")  # type: ignore[arg-type]
