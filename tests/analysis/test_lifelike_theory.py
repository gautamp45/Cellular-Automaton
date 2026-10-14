import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from cellauto.analysis import lifelike as ll
from cellauto.analysis.stats import rankdata, spearman
from cellauto.catalog import lifelike_rule
from cellauto.engines.lifelike import LifeRule, random_grid, step

LIFE = LifeRule.parse("B3/S23")


def test_dual_vectors() -> None:
    assert ll.dual(LIFE) == LifeRule.parse("B0123478/S01234678")
    assert ll.dual(LifeRule.parse("B0123478/S01234678")) == LIFE
    assert ll.is_self_dual(LifeRule.parse("B3678/S34678"))  # Day & Night
    assert not ll.is_self_dual(LIFE)


@given(st.integers(0, 2**18 - 1))
def test_dual_is_an_involution(code: int) -> None:
    r = LifeRule.from_code(code)
    assert ll.dual(ll.dual(r)) == r


@given(st.integers(0, 2**18 - 1), st.integers(0, 2**32 - 1))
def test_duality_simulation_identity_on_the_torus(code: int, seed: int) -> None:
    """Inverting every cell maps a rule to its dual. (A dead border is not invariant under
    inversion, so the identity is exact only for wrap-around grids.)"""
    rule = LifeRule.from_code(code)
    g = random_grid(9, 11, 0.5, seed)
    assert np.array_equal(1 - step(g, rule), step(1 - g, ll.dual(rule)))


def test_counts_over_all_codes() -> None:
    codes = np.arange(2**18, dtype=np.int64)
    d = ll.dual_codes(codes)
    assert np.array_equal(ll.dual_codes(d), codes)  # involution
    assert int((d == codes).sum()) == 512  # self-dual rules: 2^9
    assert (2**18 + 512) // 2 == 131328  # Burnside: number of classes
    # the B0-free set is not closed under duality: 0 in B' iff 8 not in S
    b0_free = (codes & 1) == 0
    assert not np.all((d[b0_free] & 1) == 0)
    assert np.array_equal((d & 1) == 1, ((codes >> (9 + 8)) & 1) == 0)
    for c in (6152, 2**18 - 1, 0, 12345):
        assert int(ll.dual_codes(np.array([c]))[0]) == ll.dual(LifeRule.from_code(c)).code


def test_lambda_values() -> None:
    assert ll.lambda_(LIFE) == pytest.approx(0.2734, abs=5e-5)
    assert ll.lambda_(lifelike_rule("highlife")) == pytest.approx(0.3281, abs=5e-5)
    assert ll.lambda_(lifelike_rule("seeds")) == pytest.approx(0.0547, abs=5e-5)
    assert ll.lambda_(lifelike_rule("day_and_night")) == pytest.approx(0.5)
    codes = np.array([r.code for r in (LIFE, lifelike_rule("seeds"))])
    assert np.allclose(
        ll.lambda_codes(codes), [ll.lambda_(LIFE), ll.lambda_(lifelike_rule("seeds"))]
    )


def _fp(rule_id: str) -> list[tuple[float, str]]:
    return [(round(p.rho, 4), p.stability) for p in ll.fixed_points(lifelike_rule(rule_id))]


def test_fixed_points() -> None:
    assert _fp("life") == [(0.0, "stable"), (0.1925, "unstable"), (0.3702, "stable")]
    assert _fp("highlife") == [(0.0, "stable"), (0.1915, "unstable"), (0.3886, "stable")]
    assert _fp("seeds") == [(0.0, "stable"), (0.0518, "unstable"), (0.2368, "stable")]
    assert _fp("day_and_night") == [
        (0.0, "stable"), (0.2845, "unstable"), (0.5, "stable"), (0.7155, "unstable"),
        (1.0, "stable"),
    ]  # fmt: skip


def test_coral_has_a_marginal_root_near_one() -> None:
    pts = ll.fixed_points(lifelike_rule("coral"))
    assert any(p.stability == "marginal" and p.rho > 0.9 for p in pts)
    assert all(b.rho - a.rho > 1e-3 for a, b in itertools.pairwise(pts))  # deduplicated


def test_self_dual_fixed_points_are_symmetric() -> None:
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 50:
        bits = int(rng.integers(0, 2**9))
        survival = frozenset(n for n in range(9) if (bits >> n) & 1)
        rule = LifeRule(frozenset(8 - n for n in range(9) if n not in survival), survival)
        assert ll.is_self_dual(rule)
        roots = sorted(round(p.rho, 3) for p in ll.fixed_points(rule))
        assert roots == sorted(round(1 - r, 3) for r in roots), rule.rulestring
        checked += 1


def test_mean_field_iteration_and_batch_agree() -> None:
    rules = [LIFE, lifelike_rule("seeds"), lifelike_rule("coral"), lifelike_rule("day_and_night")]
    batch = ll.mean_field_codes(np.array([r.code for r in rules]))
    for rule, value in zip(rules, batch, strict=True):
        assert value == pytest.approx(ll.iterate_mean_field(rule), abs=1e-12)
    assert batch[0] == pytest.approx(0.370, abs=1e-3)  # Life
    assert batch[1] == pytest.approx(0.237, abs=1e-3)  # Seeds
    assert batch[3] == pytest.approx(0.5, abs=1e-9)  # Day & Night


@pytest.mark.slow
def test_mean_field_versus_simulation() -> None:
    """Mean-field works for gas-like rules, fails where correlations dominate (127², 400 steps)."""
    sim = {name: ll.simulated_density(lifelike_rule(name)) for name in ("life", "seeds", "coral")}
    assert sim["life"] == pytest.approx(0.074, abs=0.03)  # mean-field predicts 0.370
    assert sim["seeds"] == pytest.approx(0.208, abs=0.03)  # mean-field predicts 0.237
    assert sim["coral"] > 0.7  # mean-field predicts 0 (the cluster-growth failure)


def test_replicator_survives_on_a_non_power_of_two_torus() -> None:
    assert ll.simulated_density(lifelike_rule("replicator"), size=61, steps=150, seeds=1) > 0.05


def test_stats_helpers() -> None:
    assert rankdata([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
