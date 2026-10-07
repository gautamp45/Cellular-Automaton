# P10: Life-like theory and the full rule-space atlas

**Tier:** Should · **Prereqs:** P07, P09 (reuses `embed.py`, `stats.py`) · **Files:** `analysis/lifelike.py`, `analysis/atlas.py`

## 1. Black/white duality (an involution on rule space)

Inverting every cell maps rule (B, S) to its dual:
B' = {8 − n : n ∉ S}, S' = {8 − n : n ∉ B}.
```python
def dual(rule: LifeRule) -> LifeRule
def is_self_dual(rule) -> bool
```
Verified:
- `dual(Life) == B0123478/S01234678` (the commented-out rule in the original `main_run.py`).
- Day & Night is self-dual.
- `1 − step(g, r) == step(1 − g, dual(r))` holds in simulation.
- Duality is an involution. There are exactly **512 self-dual rules** (= 2⁹, since S determines B), so there are **(2¹⁸ + 2⁹)/2 = 131,328 equivalence classes** (Burnside).
- The B0-free subset is **not** closed under duality: 0 ∈ B' iff 8 ∉ S.

Tests: the simulation identity under hypothesis; the involution and the self-dual count by vectorised brute force over all 2¹⁸ codes (< 1 s).

## 2. Langton's λ for life-like rules
λ = Σ_{n∈B} ½·C(8,n)/256 + Σ_{n∈S} ½·C(8,n)/256 (the probability that a uniformly random 3×3 neighbourhood maps to 1).
Verified: Life 0.2734, HighLife 0.3281, Seeds 0.0547, Day & Night 0.5000.

## 3. Mean-field theory

Assuming cells are independent with density ρ, n ~ Binomial(8, ρ), so
f(ρ) = ρ·Σ_{n∈S} C(8,n)ρⁿ(1−ρ)^{8−n} + (1−ρ)·Σ_{n∈B} C(8,n)ρⁿ(1−ρ)^{8−n}.
```python
def mean_field_map(rho, rule) -> float
def fixed_points(rule) -> list[FixedPoint]
    """Roots of f(ρ) − ρ on [0,1]: sign changes on a 20,001-point grid + bisection (60 iters) + exact zeros.
    Deduplicate roots within 1e-3. derivative by central difference (one-sided at 0/1).
    stability: |f'| < 1 − 1e-3 → stable; |f'| > 1 + 1e-3 → unstable; else marginal."""
def iterate_mean_field(rule, rho0=0.5, steps=1000) -> float
```
Verified fixed points (ρ, stability):
- Life: 0 (stable), **0.1925 (unstable)**, **0.3702 (stable)**
- HighLife: 0 s, 0.1915 u, 0.3886 s
- Seeds: 0 s, 0.0518 u, 0.2368 s
- Day & Night: 0 s, 0.2845 u, 0.5 s, 0.7155 u, 1 s (symmetric about ½, as self-duality predicts)
- Coral: near-degenerate roots close to 1 with f' ≈ 1. This is why dedup and "marginal" exist.

**Property test:** for every self-dual rule, the fixed-point set is symmetric under ρ → 1 − ρ (sample 50 self-dual codes).

**Mean-field vs reality** (`correlation_gap`). Simulate on a **127×127** torus (not a power of 2: Replicator B1357/S1357 is linear over GF(2) and dies on 128² for the same nilpotency reason as rule 90), random ρ₀ = 0.5, 400 steps, mean density of the last 50 steps, 3 seeds. Verified at 128² and t = 400:

| rule | MF ρ* from 0.5 | simulated |
|---|---|---|
| Life | 0.370 | 0.074 |
| HighLife | 0.389 | 0.038 |
| Seeds | 0.237 | 0.208 |
| Day & Night | 0.500 | 0.586 |
| Maze | 0.523 | 0.548 |
| Coral | 0.000 | 0.809 |
| Diamoeba | 0.000 | 1.000 |

The content page should explain the result. Mean-field works for "gas-like" rules (Seeds) and fails where spatial correlations dominate: Life's sparse ash, and Coral/Diamoeba's growing clusters. That gap is the motivation for everything that follows.

Fill `LifeLikeRule.{dual, self_dual, lambda, mean_field}` for the catalog.

## 4. The complete rule-space atlas (all 131,072 B0-free rules)

`analysis/atlas.py`, built on `lifelike.step_batch`.

**Why B0-free:** B0 rules strobe (empty space is born every step), and Golly-style emulation would complicate the features. State this in the content.

**Protocol (fixed):** grid **61×61** torus (prime side), ρ₀ = 0.5, T = 200, a batch of 512 rules at a time, `seed = 0` for the initial grid (**the same initial grid for every rule**, which is a controlled comparison), plus a perturbed copy (centre cell flipped) for damage spreading.

| column | dtype | definition |
|---|---|---|
| (row index) | – | row = code >> 1 (only even codes are B0-free). No id column needed. |
| `density` | uint16-q | mean density over t ∈ [150, 200) |
| `density_std` | uint16-q | std of density over the window |
| `activity` | uint16-q | mean fraction of cells changed per step over the window |
| `damage` | uint16-q | Hamming fraction between original and perturbed copies at t = 200 |
| `compression` | uint16-q | zlib size / raw size of `packbits` of the final grid |
| `lambda` | uint16-q | analytic λ |
| `mf_rho` | uint16-q | mean-field iterate from 0.5 |
| `pca_x`,`pca_y` | uint16-q | 2-D PCA of the standardised first five features |
| `cluster` | uint8 | k-means label (k by silhouette on a 10k subsample, k ∈ 3..8) |
| `is_catalog` | uint8 | 1 if the rule is in `lifelike_rules.json` |

**Engineering:**
- Process in chunks of 4096 rules. Write each chunk to `results/lifelike_atlas/chunk_<i>.npz` and skip chunks that exist, so a crash resumes.
- `--quick` runs a fixed random sample of 4096 rules (seed 0) for CI.
- Runtime verified: 512 rules × 64² × 200 steps = 4.7 s, so the full run (×2 for the damage copy) is ≈ 40 min on one CPU. Record it in meta.
- Export: `lifelike/atlas.header.json` (`BinaryTableHeader`) + `lifelike/atlas.bin` (≈ 2.9 MB) and a small `lifelike/atlas_summary.json` (`LifelikeAtlasMeta`: cluster sizes, centroids in feature space, silhouette scores, protocol, the edge-of-chaos table below, and the catalog rules' row indices).

**Edge-of-chaos analysis** (honest hypothesis test): bin λ into 20 quantile bins. Per bin, report the median and IQR of activity, damage and compression, plus Spearman correlations. Langton (1990) proposed complex behaviour peaks at intermediate λ. Mitchell, Hraber & Crutchfield (1993) challenged the generality of such claims. Report what the 131k rules show, including if it's a null result.

## 5. Tests
- `dual`, `lambda`, `fixed_points` vectors above.
- The atlas on a 64-rule subset is deterministic (run twice and get identical arrays). Life's row has density within 0.03–0.12 (verified 0.074 at 128²; 61² will differ slightly).
- The cabin export round-trips. Row ↔ code mapping: `LifeRule.from_code(2*row)`.

**Commit:** `feat(analysis): life-like duality, mean-field theory, complete rule-space atlas`
