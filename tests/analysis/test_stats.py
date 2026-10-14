import numpy as np
import pytest

from cellauto.analysis.stats import (
    bootstrap_ci,
    cohens_d_paired,
    holm_adjust,
    paired_permutation_test,
    wilson_interval,
)


def test_bootstrap_ci_of_standard_normal() -> None:
    x = np.random.default_rng(0).standard_normal(1000)
    ci = bootstrap_ci(x)
    assert ci.ci_low < 0 < ci.ci_high
    assert ci.ci_high - ci.ci_low == pytest.approx(0.12, abs=0.03)
    assert ci.mean == pytest.approx(x.mean())
    assert bootstrap_ci(x) == ci  # seeded


def test_bootstrap_ci_other_statistics() -> None:
    x = np.random.default_rng(1).exponential(2.0, 300)
    med = bootstrap_ci(x, np.median, n_boot=2000)
    assert med.ci_low <= med.mean <= med.ci_high
    custom = bootstrap_ci(x, lambda v: float(np.max(v)), n_boot=500)  # no axis argument
    assert custom.mean == pytest.approx(x.max())
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_permutation_test() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal(50)
    shifted = base + 1.0 + 0.1 * rng.standard_normal(50)
    assert paired_permutation_test(shifted, base, n_perm=20_000) < 0.01
    same_a, same_b = rng.standard_normal(50), rng.standard_normal(50)
    assert paired_permutation_test(same_a, same_b, n_perm=20_000) > 0.05
    assert paired_permutation_test(shifted, base, n_perm=2000, seed=3) == paired_permutation_test(
        shifted, base, n_perm=2000, seed=3
    )
    assert paired_permutation_test(base, base, n_perm=1000) == 1.0  # no difference at all


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and lo == pytest.approx(1 - 0.2775, abs=1e-3)
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_cohens_d_and_holm() -> None:
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 1.0, 2.0, 2.0])
    assert cohens_d_paired(a, b) == pytest.approx(np.mean(a - b) / np.std(a - b, ddof=1))
    assert np.isnan(cohens_d_paired(a, a + 1))
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([0.5, 0.9]) == pytest.approx([1.0, 1.0])
