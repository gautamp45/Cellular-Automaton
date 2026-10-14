import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cellauto.analysis import elementary as eca
from cellauto.engines import elementary as eng

SURJECTIVE = [15, 30, 45, 51, 60, 75, 85, 86, 89, 90, 101, 102, 105, 106, 120, 135, 149, 150,
              153, 154, 165, 166, 169, 170, 180, 195, 204, 210, 225, 240]  # fmt: skip


@given(st.integers(0, 255))
def test_klein_four_group(rule: int) -> None:
    assert eca.reflect(eca.reflect(rule)) == rule
    assert eca.complement(eca.complement(rule)) == rule
    assert eca.reflect(eca.complement(rule)) == eca.complement(eca.reflect(rule))
    assert rule in eca.orbit(rule) and len(eca.orbit(rule)) in (1, 2, 4)


def test_classes_and_known_orbits() -> None:
    classes = eca.equivalence_classes()
    assert len(classes) == 88
    assert sum(len(v) for v in classes.values()) == 256
    assert eca.orbit(30) == [30, 86, 135, 149]
    assert eca.orbit(110) == [110, 124, 137, 193]


@given(
    st.integers(0, 255),
    st.lists(st.integers(0, 1), min_size=1, max_size=40),
    st.sampled_from(["wrap", "dead"]),
)
def test_simulation_identities(rule: int, cells: list[int], boundary: str) -> None:
    row = np.array(cells, dtype=np.uint8)
    step = lambda r, n: eng.step(r, n, boundary)  # type: ignore[arg-type]  # noqa: E731
    assert np.array_equal(step(row[::-1].copy(), eca.reflect(rule)), step(row, rule)[::-1])
    if boundary == "wrap":  # complement symmetry needs a boundary that is itself symmetric
        assert np.array_equal(step(1 - row, eca.complement(rule)), 1 - step(row, rule))


def test_lambda() -> None:
    assert eca.lambda_(30) == 0.5
    assert eca.lambda_(110) == 0.625
    assert eca.lambda_(0) == 0.0 and eca.lambda_(255) == 1.0


def test_surjective_rules() -> None:
    assert eca.surjective_rules() == SURJECTIVE
    assert len(SURJECTIVE) == 30


def test_reversible_rules_ring_test_and_exact_test_agree() -> None:
    expected = [15, 51, 85, 170, 204, 240]
    assert eca.reversible_rules() == expected
    assert [n for n in range(256) if eca.is_injective(n)] == expected


def _brute_force_counts(rule: int, n: int) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for bits in itertools.product((0, 1), repeat=n):
        y = tuple(eng.step(np.array(bits, dtype=np.uint8), rule).tolist())
        counts[y] = counts.get(y, 0) + 1
    return counts


@pytest.mark.parametrize("rule", [0, 30, 90, 110, 150, 184])
def test_preimage_count_matches_brute_force(rule: int) -> None:
    counts = _brute_force_counts(rule, 10)
    for seed in range(20):
        cfg = np.random.default_rng(seed).integers(0, 2, 10)
        assert eca.preimage_count(rule, cfg.astype(np.uint8)) == counts.get(tuple(cfg), 0)
    assert eca.preimage_count(rule, "0" * 10) == counts.get((0,) * 10, 0)


def test_preimage_count_big_rings_and_errors() -> None:
    assert eca.preimage_count(204, "01" * 50) == 1  # identity rule: exactly one preimage
    assert eca.preimage_count(204, "01" * 50) == eca.preimage_count(204, "01" * 20)
    with pytest.raises(ValueError):
        eca.preimage_count(30, "01x")


def test_garden_of_eden_counts() -> None:
    assert eca.garden_of_eden_count(30, 12) == 280
    assert eca.garden_of_eden_count(110, 12) == 1971
    assert eca.garden_of_eden_count(90, 12) == 3072
    for rule in (30, 110):
        configs = ("".join(bits) for bits in itertools.product("01", repeat=8))
        assert eca.garden_of_eden_count(rule, 8) == sum(
            eca.preimage_count(rule, cfg) == 0 for cfg in configs
        )


def test_rule90_closed_form_matches_simulation() -> None:
    width, steps = 257, 128
    history = eng.run(eng.single_cell(width), 90, steps)
    for t in range(steps + 1):
        assert np.array_equal(eca.rule90_closed_form(t, width), history[t]), t
    with pytest.raises(ValueError):
        eca.rule90_closed_form(129, 257)


def test_gf2_rank_and_nilpotency() -> None:
    assert eca.gf2_rank(eca.gf2_transition_matrix(90, 12)) == 10
    assert eca.gf2_rank(np.eye(5, dtype=np.uint8)) == 5
    a256 = eca.gf2_transition_matrix(90, 256)
    assert eca.gf2_matrix_power(a256, 127).any()  # one power short of nilpotent
    assert not eca.gf2_matrix_power(a256, 128).any()  # (S + S^-1)^(2^7) = 0 on a ring of 2^8
    assert eca.gf2_matrix_power(eca.gf2_transition_matrix(90, 255), 128).any()
    for rule in (60, 102, 150):
        assert eca.gf2_transition_matrix(rule, 7).shape == (7, 7)
    with pytest.raises(ValueError):
        eca.gf2_transition_matrix(30, 8)


def test_gf2_matrix_matches_simulation() -> None:
    row = eng.random_row(31, seed=2)
    for rule in (60, 90, 102, 150):
        a = eca.gf2_transition_matrix(rule, 31).astype(np.int64)
        assert np.array_equal((a @ row) & 1, eng.step(row, rule))
