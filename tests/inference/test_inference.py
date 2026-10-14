import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cellauto.catalog import lifelike_rules
from cellauto.engines.lifelike import LifeRule, random_grid, step
from cellauto.inference import exhaustive, relu_compiler
from cellauto.inference.bayes import InconsistentObservation, infer_rule

LIFE = LifeRule.parse("B3/S23")

# ---- Bayesian estimator ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "rulestring", ["B3/S23", "B36/S23", "B3678/S34678", "B2/S", "B35678/S5678"]
)
def test_exact_recovery_from_one_random_pair(rulestring: str) -> None:
    rule = LifeRule.parse(rulestring)
    x0 = random_grid(64, 64, 0.5, seed=0)
    post = infer_rule([(x0, step(x0, rule))])
    assert post.unidentified() == []
    assert post.map_rule() == rule
    assert set(np.unique(post.p_one)) <= {0.0, 1.0}


@pytest.mark.parametrize("rulestring", ["B3/S23", "B36/S23", "B3678/S34678", "B2/S"])
def test_noise_robustness(rulestring: str) -> None:
    rule = LifeRule.parse(rulestring)
    rng = np.random.default_rng(1)
    x0 = random_grid(64, 64, 0.5, seed=0)
    x1 = step(x0, rule)
    noisy = x1 ^ (rng.random(x1.shape) < 0.05).astype(np.uint8)
    post = infer_rule([(x0, noisy)], eps=0.05)
    assert post.map_rule() == rule  # 0 bit errors with 5% output noise
    assert ((post.p_one > 0.5) != rule.table().astype(bool))[post.counts > 0].sum() == 0


def test_glider_leaves_entries_unidentified_but_observed_ones_are_correct() -> None:
    x0 = np.zeros((16, 16), dtype=np.uint8)
    x0[1, 2] = x0[2, 3] = x0[3, 1] = x0[3, 2] = x0[3, 3] = 1
    post = infer_rule([(x0, step(x0, LIFE))])
    assert len(post.unidentified()) > 0
    table = LIFE.table()
    observed = post.counts > 0
    assert ((post.p_one > 0.5)[observed] == table.astype(bool)[observed]).all()
    assert (post.p_one[~observed] == 0.5).all()


def test_contradictions_and_validation() -> None:
    x0 = random_grid(16, 16, 0.5, seed=2)
    wrong = 1 - step(x0, LIFE)
    with pytest.raises(InconsistentObservation):
        infer_rule([(x0, wrong[:, :]), (x0, step(x0, LIFE))])
    infer_rule([(x0, wrong)], eps=0.1)  # fine under noise
    with pytest.raises(ValueError):
        infer_rule([(x0, x0)], eps=0.7)
    with pytest.raises(ValueError):
        infer_rule([(x0, x0[:8])])


def test_posterior_confidence_grows_with_evidence() -> None:
    x0 = random_grid(32, 32, 0.5, seed=5)
    x1 = step(x0, LIFE)
    rng = np.random.default_rng(0)
    pairs = [(x0, x1 ^ (rng.random(x1.shape) < 0.2).astype(np.uint8)) for _ in range(1)]
    p1 = infer_rule(pairs, eps=0.2).p_one
    p2 = infer_rule(pairs * 3, eps=0.2).p_one
    seen = infer_rule(pairs, eps=0.2).counts > 0
    assert (np.abs(p2 - 0.5)[seen] >= np.abs(p1 - 0.5)[seen] - 1e-12).all()


# ---- exhaustive search ----------------------------------------------------------------------


def _reference_grid() -> np.ndarray:
    """The 24x24, density-0.4 grid of the verified claim: it is the 6th draw of the reference
    script's ``default_rng(0)`` stream (five 64x64 rule demos, each with a signal and a noise draw,
    come first)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        rng.random((64, 64))
        rng.random((64, 64))
    return (rng.random((24, 24)) < 0.4).astype(np.uint8)


def test_exhaustive_search_pins_down_life() -> None:
    x0 = _reference_grid()
    x2 = step(step(x0, LIFE), LIFE)
    codes = exhaustive.consistent_rules(x0, x2, 2)
    assert codes.tolist() == [LIFE.code]  # exactly one consistent rule
    info = exhaustive.ambiguity(codes)
    assert info["n_consistent"] == 1 and info["bits_fixed"] == 18


def test_identifiability_depends_on_the_data() -> None:
    """Other 24x24 grids exercise the rare entries (8 live neighbours) less: the true rule is always
    consistent, but a few extra rules may be too."""
    for seed in (1, 2, 3):
        x0 = random_grid(24, 24, 0.4, seed=seed)
        codes = exhaustive.consistent_rules(x0, step(step(x0, LIFE), LIFE), 2)
        assert LIFE.code in codes and 1 <= len(codes) <= 16
        assert exhaustive.ambiguity(codes)["bits_fixed"] >= 14


def test_exhaustive_search_reports_ambiguity() -> None:
    x0 = np.zeros((8, 8), dtype=np.uint8)
    x0[3, 3] = 1  # one live cell: most table entries are never exercised
    codes = exhaustive.consistent_rules(x0, step(x0, LIFE), 1)
    info = exhaustive.ambiguity(codes)
    assert info["n_consistent"] > 1000 and LIFE.code in codes
    assert 0 < info["bits_fixed"] < 18
    assert exhaustive.ambiguity([])["n_consistent"] == 0


def test_exhaustive_dead_boundary() -> None:
    x0 = random_grid(12, 12, 0.5, seed=4)
    codes = exhaustive.consistent_rules(x0, step(x0, LIFE, "dead"), 1, "dead")
    assert LIFE.code in codes


# ---- ReLU compiler --------------------------------------------------------------------------


def test_life_needs_four_hidden_units() -> None:
    net = relu_compiler.compile_rule(LIFE)
    assert net.centre_weight == 0.5 and net.hidden_units == 4
    assert (-net.biases).tolist() == [2.0, 2.5, 3.5, 4.0]  # kinks at v = 2, 2.5, 3.5, 4
    assert net.out_weights.tolist() == [2.0, -2.0, -2.0, 2.0]
    assert net.out_bias == 0.0


def test_compiled_nets_are_exact_for_the_catalog() -> None:
    rng = np.random.default_rng(0)
    for entry in lifelike_rules():
        rule = LifeRule.parse(entry["rulestring"])
        net = relu_compiler.compile_rule(rule)
        for boundary in ("wrap", "dead"):
            g = (rng.random((20, 23)) < 0.5).astype(np.uint8)
            out = relu_compiler.apply(net, g, boundary)  # type: ignore[arg-type]
            assert np.array_equal(out, step(g, rule, boundary)), entry["id"]  # type: ignore[arg-type]


@settings(max_examples=200, deadline=None)
@given(st.integers(0, 2**18 - 1), st.integers(0, 2**32 - 1))
def test_compiled_nets_are_exact_for_random_rules(code: int, seed: int) -> None:
    rule = LifeRule.from_code(code)
    g = random_grid(10, 12, 0.5, seed)
    out = relu_compiler.apply(relu_compiler.compile_rule(rule), g)
    assert np.array_equal(out, step(g, rule))


def test_hidden_unit_statistics_over_all_rules() -> None:
    counts = relu_compiler.hidden_unit_counts()
    assert counts.min() == 0 and counts.max() == 17
    assert counts.mean() == 12.5
    assert relu_compiler.hidden_unit_histogram() == [
        2, 2, 34, 62, 296, 652, 1786, 3742, 7622, 13586, 22046, 31810, 40580, 44864,
        41782, 31186, 16924, 5168,
    ]  # fmt: skip


def test_vectorised_counts_match_compiled_nets() -> None:
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 2**18, 300)
    vec = relu_compiler.hidden_unit_counts(codes)
    for code, n in zip(codes, vec, strict=True):
        assert relu_compiler.compile_rule(LifeRule.from_code(int(code))).hidden_units == n
