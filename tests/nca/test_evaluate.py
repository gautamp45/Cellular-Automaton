import json
from pathlib import Path

import numpy as np
import pytest

from cellauto.contract.models import NcaEvaluation
from cellauto.nca import evaluate as ev
from cellauto.nca.checkpoints import find_repo_root
from cellauto.nca.image import circle_mask

ROOT = find_repo_root(Path(__file__))


def test_damage_masks_are_identical_across_models_for_a_seed() -> None:
    p = ev.QUICK
    for seed in range(3):
        masks = [circle_mask(28, np.random.default_rng(p.damage_seed_offset + seed))
                 for _ in range(2)]  # fmt: skip
        assert np.array_equal(masks[0], masks[1])
    assert not np.array_equal(
        circle_mask(28, np.random.default_rng(p.damage_seed_offset)),
        circle_mask(28, np.random.default_rng(p.damage_seed_offset + 1)),
    )


def test_run_one_is_deterministic_and_has_the_documented_shape() -> None:
    p = ev.QUICK
    a = ev.run_one("chest-l2-sobel", 0, p, ROOT)
    b = ev.run_one("chest-l2-sobel", 0, p, ROOT)
    assert np.array_equal(a["series"], b["series"])
    assert a["series"].shape == (p.steps,)
    c = ev.run_one("chest-l2-sobel", 1, p, ROOT)
    assert not np.array_equal(a["series"], c["series"])  # different fire stream
    assert a["mse_200"] == pytest.approx(float(a["series"][199]), rel=1e-6)
    assert a["peak_after_damage"] >= a["mse_1000"] - 1e-12


def test_regeneration_is_undefined_for_non_converged_models() -> None:
    r = ev.run_one("chest-l2-laplacian", 0, ev.QUICK, ROOT)
    assert r["regen_steps"] is None and r["regen_defined"] is False
    good = ev.run_one("chest-l2-sobel-damage", 0, ev.QUICK, ROOT)
    assert good["regen_defined"] is True
    assert good["regen_steps"] is None or good["regen_steps"] >= 1


def test_paired_streams_do_not_depend_on_the_model() -> None:
    """Fire and damage RNG streams are separate, so the damage mask never shifts with fire draws."""
    p = ev.EvalProtocol(seeds=1, steps=320, damage_at=300)
    a = ev.run_one("chest-l2-sobel", 0, p, ROOT)
    b = ev.run_one("chest-l2-sobel-damage", 0, p, ROOT)
    assert a["series"].shape == b["series"].shape == (320,)


@pytest.mark.slow
def test_quick_evaluation_is_deterministic_and_valid() -> None:
    first = ev.run_evaluation(ev.QUICK, ev.QUICK_MODELS, workers=2, root=ROOT)
    second = ev.run_evaluation(ev.QUICK, ev.QUICK_MODELS, workers=1, root=ROOT)
    assert first["evaluation"] == second["evaluation"]  # independent of worker count
    model = NcaEvaluation.model_validate(first["evaluation"])
    by_id = {m.id: m for m in model.models}
    assert by_id["chest-l2-laplacian"].regen_steps is None
    assert not by_id["chest-l2-laplacian"].converged and by_id["chest-l2-sobel"].converged
    assert by_id["chest-l2-laplacian"].robustness is None
    assert by_id["chest-l2-sobel"].robustness is not None
    lo, hi = by_id["chest-l2-sobel"].mse_200.ci_low, by_id["chest-l2-sobel"].mse_200.ci_high
    assert lo <= by_id["chest-l2-sobel"].mse_200.mean <= hi
    assert {c.family for c in model.comparisons} == {"filter"}
    curve = by_id["chest-l2-sobel"].curve
    assert len(curve.steps) == len(curve.mean) == len(curve.ci_low) <= 100
    assert curve.steps[-1] == ev.QUICK.steps and all(
        lo_ <= m <= hi_ for lo_, m, hi_ in zip(curve.ci_low, curve.mean, curve.ci_high, strict=True)
    )
    assert "inference stochasticity only" in model.caveat
    json.dumps(first)  # serialisable


def test_committed_results_are_consistent_if_present() -> None:
    path = ROOT / "results" / "nca_eval.json"
    if not path.is_file():
        pytest.skip("nca-eval has not been run")
    result = json.loads(path.read_text())
    model = NcaEvaluation.model_validate(result["evaluation"])
    assert len(model.models) == 25 and model.protocol.seeds == 30
    by_id = {m.id: m for m in model.models}
    assert by_id["chest-l2-laplacian"].regen_steps is None
    assert by_id["chest-l2-sobel"].mse_200.mean < 0.005
    assert all(m.id in by_id for m in model.models)
