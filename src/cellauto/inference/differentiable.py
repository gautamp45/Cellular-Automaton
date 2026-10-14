"""Rule learning by gradient descent through a differentiable (relaxed) life-like CA.

The lookup table is relaxed to logits ``theta`` (2 x 9) with ``L = sigmoid(theta)``. For a soft
state ``x`` in [0, 1] let ``n`` be the (real-valued) Moore sum, ``l_s(n)`` the linear
interpolation of ``L[s, :]`` between integer counts, and

    x' = (1 - x) * l_0(n) + x * l_1(n),

which equals the exact CA on binary inputs. The loss is the binary cross-entropy between the soft
``k``-step rollout and the observed grid. For ``k = 1`` and binary ``x0`` the loss is a sum over
table entries of logistic losses in ``theta[s, n]`` (convex); for ``k >= 2`` the composition makes
it non-convex.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from cellauto.analysis.stats import wilson_interval
from cellauto.engines.lifelike import LifeRule, random_grid, step
from cellauto.inference.exhaustive import ambiguity, consistent_rules


def moore_sum(x: torch.Tensor, boundary: str = "wrap") -> torch.Tensor:
    """Sum of the 8 neighbours over the last two axes."""
    if boundary == "wrap":
        total = torch.zeros_like(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    total = total + torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        return total
    if boundary != "dead":
        raise ValueError("boundary must be 'wrap' or 'dead'")
    h, w = x.shape[-2:]
    p = F.pad(x, (1, 1, 1, 1))
    total = torch.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                total = total + p[..., dy : dy + h, dx : dx + w]
    return total


def soft_step(x: torch.Tensor, theta: torch.Tensor, boundary: str = "wrap") -> torch.Tensor:
    """One relaxed step. ``x``: (R, H, W) in [0, 1]; ``theta``: (R, 2, 9) logits."""
    table = torch.sigmoid(theta)
    n = moore_sum(x, boundary)
    lo = n.floor().clamp(0, 7)
    w = n - lo
    idx = lo.long().reshape(x.shape[0], -1)
    outs = []
    for s in (0, 1):
        a = torch.gather(table[:, s, :], 1, idx).reshape(x.shape)
        b = torch.gather(table[:, s, :], 1, (idx + 1).clamp(max=8)).reshape(x.shape)
        outs.append(a * (1 - w) + b * w)
    return (1 - x) * outs[0] + x * outs[1]


def rollout(x0: torch.Tensor, theta: torch.Tensor, k: int, boundary: str = "wrap") -> torch.Tensor:
    x = x0
    for _ in range(k):
        x = soft_step(x, theta, boundary)
    return x


@dataclass(frozen=True)
class FitResult:
    theta: np.ndarray  # (restarts, 2, 9) final logits
    losses: np.ndarray  # (restarts,) final BCE per restart
    codes: np.ndarray  # (restarts,) MAP rule code (theta > 0) per restart
    seconds: float


def theta_to_code(theta: np.ndarray) -> np.ndarray:
    bits = (theta.reshape(theta.shape[0], 18) > 0).astype(np.int64)
    return (bits << np.arange(18)).sum(axis=1)


def fit_rule(
    x0: np.ndarray,
    xk: np.ndarray,
    k: int,
    boundary: str = "wrap",
    restarts: int = 10,
    steps: int = 500,
    lr: float = 0.1,
    seed: int = 0,
    init_scale: float = 1.0,
) -> FitResult:
    """Adam on the relaxed table from ``restarts`` random initialisations (run as a batch)."""
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    theta = (torch.randn(restarts, 2, 9, generator=gen) * init_scale).requires_grad_(True)
    a = torch.as_tensor(x0, dtype=torch.float32)[None].expand(restarts, -1, -1)
    target = torch.as_tensor(xk, dtype=torch.float32)[None].expand(restarts, -1, -1)
    opt = torch.optim.Adam([theta], lr=lr)
    per_restart = torch.zeros(restarts)
    for _ in range(steps):
        opt.zero_grad()
        pred = rollout(a, theta, k, boundary).clamp(1e-6, 1 - 1e-6)
        per_restart = F.binary_cross_entropy(pred, target, reduction="none").mean(dim=(1, 2))
        per_restart.sum().backward()  # independent restarts: each gets its own gradient
        opt.step()
    final = theta.detach().numpy()
    return FitResult(final, per_restart.detach().numpy(), theta_to_code(final),
                     time.perf_counter() - t0)  # fmt: skip


# --- the study ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyProtocol:
    n_per_cluster: int = 5
    grid: int = 32
    density: float = 0.4
    ks: tuple[int, ...] = (1, 2, 3, 4)
    eps_values: tuple[float, ...] = (0.0, 0.02, 0.05)
    restarts: int = 10
    steps: int = 500
    lr: float = 0.1
    seed: int = 0


DEFAULT = StudyProtocol()
QUICK = StudyProtocol(n_per_cluster=1, ks=(1, 2), eps_values=(0.0, 0.05), restarts=2, steps=150)


def pick_rules(results_dir: Path, p: StudyProtocol) -> list[dict[str, Any]]:
    """``n_per_cluster`` rules from each atlas cluster, drawn with a fixed seed."""
    from cellauto.export.binary import read_cabin

    table = read_cabin(results_dir / "lifelike_atlas" / "atlas.header.json")
    clusters = table["cluster"].astype(int)
    rng = np.random.default_rng(p.seed)
    picks = []
    for c in sorted(set(clusters.tolist())):
        rows = np.flatnonzero(clusters == c)
        take = rng.choice(rows, size=min(p.n_per_cluster, rows.size), replace=False)
        picks += [{"row": int(r), "code": 2 * int(r), "cluster": int(c)} for r in sorted(take)]
    return picks


def _instance(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    """One (rule, k): a shared exhaustive search, then one fit per noise level."""
    torch.set_num_threads(1)
    rule_idx, rule, k, p = args
    true = LifeRule.from_code(rule["code"])
    x0 = random_grid(p.grid, p.grid, p.density, seed=1000 + rule_idx)
    clean = x0
    for _ in range(k):
        clean = step(clean, true)

    t0 = time.perf_counter()
    consistent = consistent_rules(x0, clean, k)
    exhaustive_s = time.perf_counter() - t0
    info = ambiguity(consistent)
    p1 = ((consistent[:, None] >> np.arange(18)) & 1).mean(axis=0)
    identifiable = (p1 == 0) | (p1 == 1)
    true_bits = (true.code >> np.arange(18)) & 1

    trials = []
    for eps in p.eps_values:
        rng = np.random.default_rng([p.seed, rule_idx, k, round(eps * 1000)])
        observed = clean ^ (rng.random(clean.shape) < eps).astype(np.uint8) if eps else clean
        fit = fit_rule(
            x0, observed, k, restarts=p.restarts, steps=p.steps, lr=p.lr,
            seed=int(rng.integers(0, 2**31 - 1)),
        )  # fmt: skip
        successes, bit_errors = [], []
        for code in fit.codes:
            out = x0
            for _ in range(k):
                out = step(out, LifeRule.from_code(int(code)))
            successes.append(bool(np.array_equal(out, clean)))
            bits = (int(code) >> np.arange(18)) & 1
            wrong = (bits != true_bits)[identifiable]
            bit_errors.append(float(wrong.mean()) if identifiable.any() else 0.0)
        trials.append(
            {
                "rule_idx": rule_idx, "cluster": rule["cluster"], "k": k, "eps": eps,
                "successes": successes, "bit_errors": bit_errors,
                "identifiable_bits": int(identifiable.sum()),
                "n_consistent": info["n_consistent"],
                "fit_seconds": fit.seconds, "exhaustive_seconds": exhaustive_s,
            }
        )  # fmt: skip
    return trials


def _cell(trials: list[dict[str, Any]]) -> dict[str, Any]:
    flat = [s for t in trials for s in t["successes"]]
    n, ok = len(flat), int(sum(flat))
    lo, hi = wilson_interval(ok, n)
    best = [any(t["successes"]) for t in trials]
    blo, bhi = wilson_interval(int(sum(best)), len(best))
    return {
        "n_instances": len(trials), "n_trials": n,
        "success_rate": ok / n, "success_ci_low": lo, "success_ci_high": hi,
        "best_of_restarts_rate": float(np.mean(best)),
        "best_ci_low": blo, "best_ci_high": bhi,
        "bit_error_rate": float(np.mean([e for t in trials for e in t["bit_errors"]])),
        "fit_seconds_mean": float(np.mean([t["fit_seconds"] for t in trials])),
        "exhaustive_seconds_mean": float(np.mean([t["exhaustive_seconds"] for t in trials])),
        "identifiable_bits_mean": float(np.mean([t["identifiable_bits"] for t in trials])),
    }  # fmt: skip


def run_study(
    results_dir: Path, p: StudyProtocol = DEFAULT, workers: int | None = None
) -> dict[str, Any]:
    """Run the study and return an ``InferenceStudy``-shaped dict."""
    rules = pick_rules(results_dir, p)
    jobs = [(i, r, k, p) for i, r in enumerate(rules) for k in p.ks]
    if workers == 1:
        grouped = [_instance(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            grouped = list(pool.map(_instance, jobs, chunksize=1))
    trials = [t for group in grouped for t in group]
    cells = []
    for k in p.ks:
        for eps in p.eps_values:
            group = [t for t in trials if t["k"] == k and t["eps"] == eps]
            cells.append({"k": k, "eps": eps, **_cell(group)})
    by_cluster = []
    for c in sorted({r["cluster"] for r in rules}):
        for k in p.ks:
            sel = [t for t in trials if t["cluster"] == c and t["k"] == k and t["eps"] == 0.0]
            if sel:
                cell = _cell(sel)
                by_cluster.append(
                    {
                        "cluster": c,
                        "k": k,
                        "n_instances": len(sel),
                        "success_rate": cell["success_rate"],
                        "identifiable_bits_mean": cell["identifiable_bits_mean"],
                    }
                )
    return {
        "format_version": 1,
        "protocol": {"n_rules": len(rules), "grid": p.grid, "density": p.density,
                     "ks": list(p.ks), "eps_values": list(p.eps_values),
                     "restarts": p.restarts, "steps": p.steps, "lr": p.lr, "seed": p.seed,
                     "optimizer": "Adam", "init": "theta ~ N(0, 1)"},
        "rules": [{"code": r["code"], "row": r["row"], "cluster": r["cluster"],
                   "rulestring": LifeRule.from_code(r["code"]).rulestring} for r in rules],
        "cells": cells,
        "by_cluster": by_cluster,
    }  # fmt: skip


__all__ = ["DEFAULT", "QUICK", "FitResult", "StudyProtocol", "fit_rule", "pick_rules", "rollout",
           "run_study", "soft_step", "theta_to_code"]  # fmt: skip
