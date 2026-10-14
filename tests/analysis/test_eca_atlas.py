import numpy as np
import pytest

from cellauto.analysis import elementary as eca
from cellauto.analysis.eca_atlas import (
    DEFAULT,
    EMBED_FEATURES,
    FEATURES,
    QUICK,
    Protocol,
    behaviour_features,
    build_atlas,
    run_atlas,
)
from cellauto.contract.models import EcaAtlas

# (damage, compression, density), verified with 6 trials at width 251
REFERENCE = {
    30: (0.448, 1.003, 0.500),
    110: (0.186, 0.641, 0.570),
    54: (0.195, 0.646, 0.479),
    90: (0.136, 1.003, 0.502),
    184: (0.044, 0.092, 0.523),
}


@pytest.fixture(scope="module")
def all_features() -> dict[str, np.ndarray]:
    return behaviour_features()


def test_reference_values(all_features: dict[str, np.ndarray]) -> None:
    for rule, (dmg, comp, dens) in REFERENCE.items():
        assert all_features["damage"][rule] == pytest.approx(dmg, abs=0.03), rule
        assert all_features["compression"][rule] == pytest.approx(comp, abs=0.03), rule
        assert all_features["density"][rule] == pytest.approx(dens, abs=0.03), rule


def test_rule_zero_is_trivial(all_features: dict[str, np.ndarray]) -> None:
    f = {k: v[0] for k, v in all_features.items()}
    assert f["damage"] == 0 and f["density"] == 0 and f["activity"] == 0
    assert f["block_entropy"] == 0 and f["compression"] < 0.01


def test_known_qualitative_behaviour(all_features: dict[str, np.ndarray]) -> None:
    assert all_features["period"][184] >= 1  # traffic settles into a travelling pattern
    assert all_features["period"][30] == 0  # rule 30 never repeats
    assert all_features["block_entropy"][30] > 0.95 > all_features["block_entropy"][184]


def test_features_are_deterministic_and_chunk_independent() -> None:
    rules = [3, 30, 110, 150]
    a = behaviour_features(rules, DEFAULT, chunk=64)
    b = behaviour_features(rules, DEFAULT, chunk=1)
    for name in FEATURES:
        assert np.array_equal(a[name], b[name]), name
    full = behaviour_features(None, QUICK)
    again = behaviour_features(None, QUICK)
    assert all(np.array_equal(full[n], again[n]) for n in FEATURES)
    assert len(full["damage"]) == 256


def _orbit_spreads(feats: dict[str, np.ndarray]) -> list[float]:
    return [
        float(feats[name][rules].max() - feats[name][rules].min())
        for rules in eca.equivalence_classes().values()
        for name in ("damage", "compression", "activity")
    ]


def test_symmetry_orbits_are_consistent_at_the_atlas_budget(
    all_features: dict[str, np.ndarray],
) -> None:
    """With only 6 trials, rules whose dynamics depend on the initial row (e.g. the 110 orbit) are
    noisy, so the ±0.05 property holds for the large majority of orbits, not all."""
    spreads = np.array(_orbit_spreads(all_features))
    assert (spreads <= 0.05).mean() >= 0.9
    assert np.median(spreads) < 0.02
    for rule in range(256):
        c = eca.complement(rule)
        # 0.03 holds with enough trials (slow test below). With the 6 shared initial rows (mean
        # density 0.532) shift-like rules inherit that sample's bias, e.g. rule 170 gives 0.064.
        assert abs(all_features["density"][c] - (1 - all_features["density"][rule])) <= 0.08, rule


@pytest.mark.slow
def test_symmetry_orbits_are_consistent_with_enough_trials() -> None:
    """The exact property from the plan: within an orbit damage, compression and activity agree
    within 0.05 (sampling noise shrinks like 1/sqrt(trials): worst 0.22 at 6, 0.04 at 96)."""
    feats = behaviour_features(None, Protocol(trials=96))
    assert max(_orbit_spreads(feats)) <= 0.05
    for rule in range(256):
        assert abs(feats["density"][eca.complement(rule)] - (1 - feats["density"][rule])) <= 0.03


def test_atlas_structure(all_features: dict[str, np.ndarray]) -> None:
    atlas = build_atlas(all_features, DEFAULT)
    model = EcaAtlas.model_validate(atlas)
    assert [r.rule for r in model.rows] == list(range(256))
    assert model.chosen_k in (3, 4, 5, 6)
    assert [s.k for s in model.k_selection] == [3, 4, 5, 6]
    assert all(-1 <= s.silhouette <= 1 for s in model.k_selection)
    assert {c.k for c in model.clusterings} >= {model.chosen_k, 4}
    assert model.embedding.features == list(EMBED_FEATURES)
    assert 0.3 < sum(model.embedding.explained_variance) <= 1.0
    for clustering in model.clusterings:
        assert sum(c.size for c in clustering.summary) == 256
        counts = np.array(clustering.contingency.counts)
        assert counts.sum() == 18  # the 18 catalog rules with a Wolfram class, and no others
        assert len(clustering.contingency.assignments) == 18
    # cluster 0 is the most compressible
    first = next(c for c in model.clusterings if c.k == model.chosen_k).summary
    comps = [c.centroid["compression"] for c in first]
    assert comps == sorted(comps)


def test_trivial_rules_share_a_cluster_apart_from_chaotic(
    all_features: dict[str, np.ndarray],
) -> None:
    atlas = EcaAtlas.model_validate(build_atlas(all_features, DEFAULT))
    cl = {r.rule: r.cluster_k4 for r in atlas.rows}
    assert len({cl[r] for r in (0, 8, 32, 40)}) == 1  # class I together
    assert len({cl[r] for r in (30, 45, 90, 150)}) == 1  # class III chaos together
    assert cl[0] != cl[30]


def test_run_atlas_quick_and_determinism() -> None:
    a = run_atlas(quick=True)
    b = run_atlas(quick=True)
    assert a["atlas"] == b["atlas"]
    assert a["meta"]["quick"] is True and a["meta"]["runtime_s"] > 0
    assert a["atlas"]["protocol"]["width"] == QUICK.width
    EcaAtlas.model_validate(a["atlas"])
