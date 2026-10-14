import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cellauto.engines.elementary import (
    from_bits,
    random_row,
    rule_table,
    run,
    single_cell,
    step,
    to_bits,
)


def test_rule_tables() -> None:
    assert rule_table(30).tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert rule_table(110).tolist() == [0, 1, 1, 1, 0, 1, 1, 0]
    assert rule_table(90).tolist() == [0, 1, 0, 1, 1, 0, 1, 0]


@pytest.mark.parametrize("bad", [256, -1, True, 1.5])
def test_rule_table_rejects(bad: object) -> None:
    with pytest.raises(ValueError):
        rule_table(bad)  # type: ignore[arg-type]


def test_rule30_centre_column_a051023() -> None:
    h = run(single_cell(201), 30, 40)
    assert h[:25, 100].tolist() == [
        1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
    ]  # fmt: skip


def test_rule30_early_rows() -> None:
    h = run(single_cell(201), 30, 2)
    assert h[1, 97:104].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert h[2, 96:105].tolist() == [0, 0, 1, 1, 0, 0, 1, 0, 0]


def test_rule110_one_step() -> None:
    assert run(single_cell(11), 110, 1)[1].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_rule90_sierpinski_gould() -> None:
    h = run(single_cell(129), 90, 32)
    for t in range(32):
        assert h[t].sum() == 2 ** bin(t).count("1")


def test_edges_regression_b3() -> None:
    row = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    assert step(row, 90, "wrap").tolist() == [0, 1, 0, 0, 1]
    assert step(row, 90, "dead").tolist() == [0, 1, 0, 0, 0]


def test_rule90_gf2_nilpotency_on_power_of_two_ring() -> None:
    assert run(random_row(256, seed=3), 90, 128, "wrap")[-1].sum() == 0
    assert run(random_row(255, seed=3), 90, 128, "wrap")[-1].sum() != 0


def test_run_shape_and_errors() -> None:
    row = single_cell(7)
    assert run(row, 30, 0).shape == (1, 7)
    assert run(row, 30, 5).shape == (6, 7)
    with pytest.raises(ValueError):
        run(row, 30, -1)
    with pytest.raises(ValueError):
        run(row, 30, 1, "torus")  # type: ignore[arg-type]


def test_bits_roundtrip_and_random_determinism() -> None:
    assert to_bits(from_bits("0110")) == "0110"
    with pytest.raises(ValueError):
        from_bits("01x")
    assert np.array_equal(random_row(100, seed=1), random_row(100, seed=1))
    assert not np.array_equal(random_row(100, seed=1), random_row(100, seed=2))


def _slow_step(row: list[int], rule: int) -> list[int]:
    n = len(row)
    return [
        (rule >> ((row[(i - 1) % n] << 2) | (row[i] << 1) | row[(i + 1) % n])) & 1 for i in range(n)
    ]


def _reflect(rule: int) -> int:
    out = 0
    for idx in range(8):
        lt, c, r = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        mirrored = (r << 2) | (c << 1) | lt
        out |= ((rule >> mirrored) & 1) << idx
    return out


@given(st.integers(0, 255), st.lists(st.integers(0, 1), min_size=1, max_size=64))
def test_matches_slow_reference(rule: int, cells: list[int]) -> None:
    row = np.array(cells, dtype=np.uint8)
    assert step(row, rule, "wrap").tolist() == _slow_step(cells, rule)


@given(st.integers(0, 255), st.lists(st.integers(0, 1), min_size=1, max_size=64))
def test_mirror_symmetry(rule: int, cells: list[int]) -> None:
    row = np.array(cells, dtype=np.uint8)
    for boundary in ("wrap", "dead"):
        lhs = step(row[::-1].copy(), _reflect(rule), boundary)  # type: ignore[arg-type]
        assert np.array_equal(lhs, step(row, rule, boundary)[::-1])  # type: ignore[arg-type]
