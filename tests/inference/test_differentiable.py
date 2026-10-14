import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cellauto.engines.lifelike import LifeRule, random_grid, step  # noqa: E402
from cellauto.inference import differentiable as D  # noqa: E402

pytestmark = pytest.mark.torch

LIFE = LifeRule.parse("B3/S23")


def _hard_theta(rule: LifeRule) -> "torch.Tensor":
    return torch.where(torch.as_tensor(rule.table()) > 0, 30.0, -30.0)[None]


@pytest.mark.parametrize("boundary", ["wrap", "dead"])
def test_soft_step_equals_the_exact_ca_on_binary_inputs(boundary: str) -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        rule = LifeRule.from_code(int(rng.integers(0, 2**18)))
        g = random_grid(14, 17, 0.5, seed=int(rng.integers(0, 10_000)))
        out = D.soft_step(
            torch.as_tensor(g, dtype=torch.float32)[None], _hard_theta(rule), boundary
        )
        assert np.allclose(out[0].numpy(), step(g, rule, boundary), atol=1e-6)  # type: ignore[arg-type]


def test_soft_step_is_differentiable_and_stays_in_unit_interval() -> None:
    x = torch.rand(2, 10, 10)
    theta = torch.randn(2, 2, 9, requires_grad=True)
    y = D.rollout(x, theta, 3)
    assert y.min() >= 0 and y.max() <= 1
    y.sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert (theta.grad != 0).any()


def _reproduces(codes: np.ndarray, x0: np.ndarray, clean: np.ndarray, k: int) -> list[bool]:
    result = []
    for c in codes:
        out = x0
        for _ in range(k):
            out = step(out, LifeRule.from_code(int(c)))
        result.append(bool(np.array_equal(out, clean)))
    return result


def test_k1_is_convex_so_every_restart_succeeds() -> None:
    x0 = random_grid(32, 32, 0.4, seed=1000)
    clean = step(x0, LIFE)
    fit = D.fit_rule(x0, clean, 1, restarts=10, steps=500, seed=0)
    assert all(_reproduces(fit.codes, x0, clean, 1))
    assert (fit.codes == LIFE.code).all()  # all 18 entries are observed on a 32x32 soup


def test_k2_is_non_convex_not_every_restart_succeeds() -> None:
    x0 = random_grid(32, 32, 0.4, seed=1000)
    clean = step(step(x0, LIFE), LIFE)
    fit = D.fit_rule(x0, clean, 2, restarts=10, steps=500, seed=0)
    ok = _reproduces(fit.codes, x0, clean, 2)
    assert 0 < sum(ok) < len(ok)  # some restarts find Life, others get stuck


def test_fit_is_seeded() -> None:
    x0 = random_grid(16, 16, 0.4, seed=1)
    clean = step(x0, LIFE)
    a = D.fit_rule(x0, clean, 1, restarts=3, steps=30, seed=5)
    b = D.fit_rule(x0, clean, 1, restarts=3, steps=30, seed=5)
    assert np.array_equal(a.theta, b.theta)
    assert not np.array_equal(a.theta, D.fit_rule(x0, clean, 1, restarts=3, steps=30, seed=6).theta)


@pytest.mark.slow
def test_quick_study_structure_and_determinism() -> None:
    from pathlib import Path

    from cellauto.contract.models import InferenceStudy

    results = Path(__file__).resolve().parents[2] / "results"
    first = D.run_study(results, D.QUICK, workers=1)
    again = D.run_study(results, D.QUICK, workers=1)
    study = InferenceStudy.model_validate(first)
    assert [(c.k, c.eps) for c in study.cells] == [(1, 0.0), (1, 0.05), (2, 0.0), (2, 0.05)]
    for c in study.cells:
        assert 0 <= c.success_ci_low <= c.success_rate <= c.success_ci_high <= 1
        assert c.n_trials == c.n_instances * D.QUICK.restarts
    strip = lambda cells: [  # noqa: E731
        {k: v for k, v in c.items() if "seconds" not in k} for c in cells
    ]
    assert strip(first["cells"]) == strip(again["cells"])  # wall-clock aside, deterministic
    assert {r["cluster"] for r in first["rules"]} == {0, 1, 2, 3}
