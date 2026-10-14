# P12: Rigorous NCA evaluation

**Tier:** Should · **Prereqs:** P05, P09 (`stats.py`) · **File:** `src/cellauto/nca/evaluate.py`, `src/cellauto/analysis/stats.py`

The 2023 GIFs and the single-seed MSE table are anecdotes. This phase turns them into an evaluation with uncertainty, which is the difference between "I trained some models" and "I can evaluate models".

## 1. `analysis/stats.py` (numpy only)
```python
def bootstrap_ci(x, stat=np.mean, n_boot=10_000, alpha=0.05, seed=0) -> Interval   # percentile bootstrap
def paired_permutation_test(a, b, n_perm=100_000, seed=0) -> float  # two-sided sign-flip test on a−b
def wilson_interval(successes, n, z=1.96) -> tuple[float, float]
def cohens_d_paired(a, b) -> float
```
Tests: the bootstrap CI of N(0,1) samples (n = 1000) contains 0 and has width ≈ 0.12 ± 0.03; the permutation test gives p < 0.01 for a constant shift of 1σ with n = 50, and p > 0.05 for identical distributions (seeded); Wilson(0, 10) = (0, ~0.278).

## 2. Protocol (fixed, documented in `content/nca.md`)

For each of the 25 models and each seed s ∈ 0..29:
- **Separate RNG streams** (a lesson from the pilot): `fire_rng = default_rng(s)` and `damage_rng = default_rng(10_000 + s)`. Damage masks are then identical across models for the same s, so model comparisons are **paired**.
- Run 1000 steps from the seed at fire rate 0.5. Record MSE(RGBA vs target) at every step.
- **Damage:** at t = 300, multiply the state by `1 − circle_mask(28, damage_rng)`.
- Metrics:
  - `mse_200`, `mse_1000`
  - `stability = mse_1000 / mse_200` (> 1 means drift)
  - `peak_after_damage` = max MSE over t ∈ [300, 1000)
  - `regen_steps` = first t − 300 with MSE ≤ max(1.1·mse_299, mse_299 + 1e-4). **Only defined when mse_299 ≤ 0.005.** For non-converged models it is `None`. The pilot showed that a relative threshold on a bad model reports a meaningless "recovery in 1 step".
  - `psnr_200` = 10·log10(1 / mse_200)
- **Fire-rate robustness** (models with converged mse_200 only): `mse_200` at fire rates {0.25, 0.5, 0.75, 1.0}, 10 seeds.

Verified pilot (5 seeds, shared RNG): `chest-l2-sobel-damage` regen 35.4 ± 8.2 steps vs `chest-l2-sobel` 59.6 ± 16.5; mse_200 0.0003 for both; laplacian mse_200 0.0326. Runtime ≈ 1 s per model-seed-1000 steps, so the full protocol takes ≈ 15–20 min.

## 3. Analyses to report (`results/nca_eval.json`, exported as `nca/evaluation.json`, model `NcaEvaluation`)
1. **Ablation table:** per model, mean and 95% bootstrap CI for every metric.
2. **Filter effect:** within each target, compare sobel vs {scharr, gaussian, laplacian, mean} on mse_200 (L2 models), paired by seed. Report effect size and permutation p-value with Holm correction across comparisons.
3. **Loss effect:** sobel models, L2 vs {L1, Manhattan, Hinge}.
4. **Damage training effect:** `chest-l2-sobel-damage` vs `chest-l2-sobel` on `regen_steps` and `peak_after_damage`, paired.
5. **Curves for the frontend:** the mean and 95% band of MSE(t) for every model, downsampled to 100 points (in the JSON).
6. Fill `NcaModel.metrics` in `nca/models.json`.

**Mandatory caveat in the content and REPORT:** these CIs cover **inference stochasticity only**. Each configuration was *trained once*, so training-seed variance is unmeasured, and a filter "effect" may partly be training luck. P14 addresses this. Stating it shows you know the difference.

## 4. Tests
- `--quick` (2 models × 3 seeds × 400 steps) runs in < 30 s and is deterministic.
- Damage masks are identical across models for the same seed (assert equal arrays).
- `regen_steps is None` for `chest-l2-laplacian`.

CLI: `cellauto analyze nca-eval [--quick]`.

**Commit:** `feat(nca): multi-seed evaluation protocol with regeneration metrics and statistical tests`
