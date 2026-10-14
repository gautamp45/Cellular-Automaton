"""Multi-seed NCA evaluation with regeneration metrics and paired statistical tests.

Protocol (fixed; see ``content/nca.md``). For every model and every seed ``s``:

* separate RNG streams, ``fire_rng = default_rng(s)`` and ``damage_rng = default_rng(10_000 + s)``,
  so the damage mask for seed ``s`` is identical across models and comparisons are paired;
* grow from the seed for ``steps`` updates at fire rate 0.5, recording MSE(RGBA vs target) after
  every update;
* after update ``damage_at`` (once its MSE is recorded) multiply the state by
  ``1 - circle_mask(damage_rng)``.

The reported confidence intervals cover **inference stochasticity only**: every configuration was
trained once, so training-seed variance is not measured.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cellauto.analysis.eca_atlas import run_info
from cellauto.analysis.stats import (
    Interval,
    bootstrap_ci,
    cohens_d_paired,
    holm_adjust,
    paired_permutation_test,
)
from cellauto.nca.checkpoints import find_repo_root, load_manifest, load_model
from cellauto.nca.image import circle_mask, load_target, make_seed
from cellauto.nca.numpy_model import NumpyNCA
from cellauto.nca.spec import NCAConfig

CAVEAT = (
    "These confidence intervals cover inference stochasticity only (the random fire mask and the "
    "damage position). Each configuration was trained once, so training-seed variance is "
    "unmeasured and a filter or loss 'effect' may partly be training luck."
)
REGEN_RULE = (
    "first update count k >= 1 after the damage with MSE <= max(1.1 * mse_pre, mse_pre + 1e-4), "
    "where mse_pre is the MSE just before damage; defined only if mse_pre <= converged_threshold"
)


@dataclass(frozen=True)
class EvalProtocol:
    seeds: int = 30
    steps: int = 1000
    damage_at: int = 300
    fire_rate: float = 0.5
    damage_seed_offset: int = 10_000
    robustness_rates: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    robustness_seeds: int = 10
    converged_threshold: float = 0.005
    curve_points: int = 100
    n_boot: int = 10_000
    curve_boot: int = 2_000
    n_perm: int = 100_000


DEFAULT = EvalProtocol()
QUICK = EvalProtocol(seeds=3, steps=400, damage_at=200, robustness_seeds=2, n_boot=500,
                     curve_boot=200, n_perm=2_000)  # fmt: skip


QUICK_MODELS = ("chest-l2-sobel", "chest-l2-laplacian")


def _model_for_rate(model: NumpyNCA, fire_rate: float) -> NumpyNCA:
    cfg: NCAConfig = replace(model.config, fire_rate=fire_rate)
    return NumpyNCA(model.w1, model.b1, model.w2, cfg)


def run_one(model_id: str, seed: int, p: EvalProtocol, root: Path | None = None) -> dict[str, Any]:
    """One model, one seed: the MSE series and the per-run metrics."""
    base = find_repo_root(root)
    entry = next(m for m in load_manifest(base) if m["id"] == model_id)
    model = _model_for_rate(load_model(model_id, base), p.fire_rate)
    target = load_target(base / entry["target_image"], entry["grid_size"])
    if entry["padding"]:
        target = np.pad(target, ((0, 0), (entry["padding"],) * 2, (entry["padding"],) * 2))
    fire_rng = np.random.default_rng(seed)
    damage_rng = np.random.default_rng(p.damage_seed_offset + seed)
    x = make_seed(entry["grid_size"], entry["n_channels"], entry["padding"])
    mse = np.empty(p.steps)
    for t in range(1, p.steps + 1):
        nxt = model.step(x, fire_rng)
        assert isinstance(nxt, np.ndarray)
        x = nxt
        mse[t - 1] = float(((x[:4] - target) ** 2).mean())
        if t == p.damage_at:
            x = (x * (1.0 - circle_mask(x.shape[-1], damage_rng))).astype(np.float32)

    mse_200 = float(mse[min(199, p.steps - 1)])
    mse_end = float(mse[-1])
    pre = float(mse[p.damage_at - 1])
    regen: int | None = None
    if pre <= p.converged_threshold:
        threshold = max(1.1 * pre, pre + 1e-4)
        post = mse[p.damage_at :]
        hits = np.flatnonzero(post <= threshold)
        regen = int(hits[0]) + 1 if hits.size else None
    return {
        "series": mse.astype(np.float32),
        "mse_200": mse_200,
        "mse_1000": mse_end,
        "stability": mse_end / mse_200 if mse_200 > 0 else float("nan"),
        "peak_after_damage": float(np.max(mse[p.damage_at :])),
        "psnr_200": float(10.0 * np.log10(1.0 / mse_200)) if mse_200 > 0 else float("inf"),
        "regen_steps": regen,
        "regen_defined": pre <= p.converged_threshold,
    }


def run_robustness(model_id: str, rate: float, seed: int, root: Path | None = None) -> float:
    """MSE after 200 updates at ``rate`` (no damage), fire stream ``default_rng(seed)``."""
    base = find_repo_root(root)
    entry = next(m for m in load_manifest(base) if m["id"] == model_id)
    model = _model_for_rate(load_model(model_id, base), rate)
    target = load_target(base / entry["target_image"], entry["grid_size"])
    rng = np.random.default_rng(seed)
    x = make_seed(entry["grid_size"], entry["n_channels"], entry["padding"])
    for _ in range(200):
        nxt = model.step(x, rng)
        assert isinstance(nxt, np.ndarray)
        x = nxt
    return float(((x[:4] - target) ** 2).mean())


def _job(args: tuple[Any, ...]) -> tuple[Any, ...]:
    kind, *rest = args
    if kind == "run":
        model_id, seed, p, root = rest
        return (kind, model_id, seed, run_one(model_id, seed, p, root))
    model_id, rate, seed, root = rest
    return (kind, model_id, rate, seed, run_robustness(model_id, rate, seed, root))


def _interval(x: npt.ArrayLike, p: EvalProtocol) -> dict[str, float]:
    iv: Interval = bootstrap_ci(np.asarray(x, dtype=np.float64), n_boot=p.n_boot)
    return {"mean": iv.mean, "ci_low": iv.ci_low, "ci_high": iv.ci_high}


def _curve(series: npt.NDArray[np.float32], p: EvalProtocol) -> dict[str, Any]:
    n_seeds, steps = series.shape
    idx = np.unique(np.linspace(0, steps - 1, p.curve_points).round().astype(int))
    rng = np.random.default_rng(0)
    resample = rng.integers(0, n_seeds, size=(p.curve_boot, n_seeds))
    means = series[:, idx].astype(np.float64)[resample].mean(axis=1)  # (boot, points)
    lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
    return {
        "steps": [int(i) + 1 for i in idx],
        "mean": series[:, idx].mean(axis=0).tolist(),
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
    }


def _round(obj: Any) -> Any:
    """Round floats to 8 significant digits so the results file is compact and stable."""
    if isinstance(obj, float):
        return float(f"{obj:.8g}")
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def _compare(
    runs: dict[str, list[dict[str, Any]]],
    family: str,
    target: str,
    a: str,
    b: str,
    metric: str,
    p: EvalProtocol,
) -> dict[str, Any] | None:
    xs, ys = [], []
    for ra, rb in zip(runs[a], runs[b], strict=True):
        va, vb = ra[metric], rb[metric]
        if va is not None and vb is not None:
            xs.append(float(va))
            ys.append(float(vb))
    if len(xs) < 2:
        return None
    d = cohens_d_paired(ys, xs)
    return {
        "family": family, "target": target, "a": a, "b": b, "metric": metric,
        "n_pairs": len(xs), "mean_diff": float(np.mean(np.array(ys) - np.array(xs))),
        "cohens_d": None if np.isnan(d) else d,
        "p_value": paired_permutation_test(ys, xs, n_perm=p.n_perm),
        "p_holm": 1.0,
    }  # fmt: skip


def _holm_family(rows: list[dict[str, Any]]) -> None:
    for row, adj in zip(rows, holm_adjust([r["p_value"] for r in rows]), strict=True):
        row["p_holm"] = adj


def run_evaluation(
    p: EvalProtocol = DEFAULT,
    models: Sequence[str] | None = None,
    workers: int | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Run the protocol and return ``{"evaluation": NcaEvaluation-shaped, "meta": {...}}``."""
    t0 = time.perf_counter()
    base = find_repo_root(root)
    manifest = {m["id"]: m for m in load_manifest(base)}
    ids = sorted(manifest) if models is None else list(models)

    jobs: list[tuple[Any, ...]] = [("run", i, s, p, base) for i in ids for s in range(p.seeds)]
    robust = [
        ("rob", i, rate, s, base)
        for i in ids
        for rate in p.robustness_rates
        for s in range(p.robustness_seeds)
    ]
    # Robustness is only meaningful for converged models, which is known after the main runs.
    if workers == 1:
        results = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs, chunksize=4))
    runs: dict[str, list[dict[str, Any]]] = {i: [] for i in ids}
    for _, model_id, _, res in sorted(results, key=lambda r: (r[1], r[2])):
        runs[model_id].append(res)

    converged = {i: float(np.mean([r["mse_200"] for r in runs[i]])) <= p.converged_threshold
                 for i in ids}  # fmt: skip
    robust = [j for j in robust if converged[j[1]]]
    if workers == 1:
        rob_results = [_job(j) for j in robust]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rob_results = list(pool.map(_job, robust, chunksize=4))
    rob: dict[tuple[str, float], list[float]] = {}
    for _, model_id, rate, _, mse in sorted(rob_results, key=lambda r: (r[1], r[2], r[3])):
        rob.setdefault((model_id, rate), []).append(mse)

    model_rows = []
    for i in ids:
        rs = runs[i]
        regen = [r["regen_steps"] for r in rs if r["regen_steps"] is not None]
        model_rows.append(
            {
                "id": i,
                "n_seeds": p.seeds,
                "converged": converged[i],
                "mse_200": _interval([r["mse_200"] for r in rs], p),
                "mse_1000": _interval([r["mse_1000"] for r in rs], p),
                "stability": _interval([r["stability"] for r in rs], p),
                "peak_after_damage": _interval([r["peak_after_damage"] for r in rs], p),
                "psnr_200": _interval([r["psnr_200"] for r in rs], p),
                "regen_steps": _interval(regen, p) if regen else None,
                "regen_defined": sum(r["regen_defined"] for r in rs),
                "regen_recovered": len(regen),
                "robustness": (
                    [{"fire_rate": rate, "mse_200": _interval(rob[(i, rate)], p)}
                     for rate in p.robustness_rates] if converged[i] else None
                ),
                "curve": _curve(np.stack([r["series"] for r in rs]), p),
            }
        )  # fmt: skip

    comparisons: list[dict[str, Any]] = []

    def add(family: str, specs: list[tuple[str, str, str, str]]) -> None:
        rows = [c for t, a, b, m in specs if a in runs and b in runs
                for c in [_compare(runs, family, t, a, b, m, p)] if c is not None]  # fmt: skip
        _holm_family(rows)
        comparisons.extend(rows)

    targets = sorted({m["target"] for m in manifest.values()})
    filters = ("scharr", "gaussian", "laplacian", "mean")
    add(
        "filter", [(t, f"{t}-l2-sobel", f"{t}-l2-{f}", "mse_200") for t in targets for f in filters]
    )
    losses = ("l1", "manhattan", "hinge")
    add(
        "loss", [(t, f"{t}-l2-sobel", f"{t}-{x}-sobel", "mse_200") for t in targets for x in losses]
    )
    add("damage_training", [("chest", "chest-l2-sobel", "chest-l2-sobel-damage", m)
                            for m in ("regen_steps", "peak_after_damage")])  # fmt: skip

    evaluation = {
        "format_version": 1,
        "protocol": {
            "seeds": p.seeds, "steps": p.steps, "damage_at": p.damage_at,
            "fire_rate": p.fire_rate, "damage_seed_offset": p.damage_seed_offset,
            "robustness_rates": list(p.robustness_rates),
            "robustness_seeds": p.robustness_seeds,
            "converged_threshold": p.converged_threshold, "regen_rule": REGEN_RULE,
        },
        "models": model_rows,
        "comparisons": comparisons,
        "caveat": CAVEAT,
    }  # fmt: skip
    return {
        "evaluation": _round(evaluation),
        "meta": {**run_info(), "runtime_s": round(time.perf_counter() - t0, 2),
                 "workers": workers, "quick": p == QUICK, "models": ids},
    }  # fmt: skip
