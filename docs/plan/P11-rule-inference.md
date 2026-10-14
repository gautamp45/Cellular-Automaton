# P11: Inverse problems, recovering the rule from observations

**Tier:** Should · **Prereqs:** P10 · **Files:** `inference/{bayes,exhaustive,relu_compiler,differentiable}.py`

This replaces the deleted Rule Predictor with a principled treatment. Its arc is the ML-maturity signal of the repo: **exact inference where it's tractable, gradient-based learning where it isn't, and evidence for which is which.**

## 1. Exact Bayesian estimator (k = 1): `inference/bayes.py`

Observation: x₀ and a noisy x₁, where each output bit is flipped with probability ε. Each LUT entry L[s,n] is an independent Bernoulli parameter. For an entry with k₁ observed 1s and k₀ observed 0s, the posterior log-odds is
`llr = (k₁ − k₀)·log((1−ε)/ε)` (ε > 0), and `p = σ(llr)`. For ε = 0, any contradiction raises `InconsistentObservation`. Entries never observed have p = ½ and are **unidentifiable**.

```python
@dataclass(frozen=True)
class RulePosterior:
    p_one: FloatArray        # (2,9)
    counts: np.ndarray       # (2,9) observations per entry
    def map_rule(self) -> LifeRule
    def unidentified(self) -> list[tuple[int,int]]
def infer_rule(pairs: Iterable[tuple[BinaryArray, BinaryArray]], eps: float = 0.0, boundary="wrap") -> RulePosterior
```
Implement the counts with `np.add.at(k1, (x0, counts), x1)`. Verified: from one 64×64 random pair at ρ = 0.5, the rule is recovered exactly with 0 unobserved entries for B3/S23, B36/S23, B3678/S34678, B2/S and B35678/S5678. With 5% output noise there are 0 bit errors.

**Identifiability test:** a lone glider on an empty 16×16 grid leaves entries unobserved (high counts never occur). Assert `len(unidentified()) > 0` and that every *observed* entry matches Life.
**Browser:** this is ~30 lines of TS and drives the "draw, run, infer" demo with per-bit confidence bars. Add `#bayes-inference` to ALGORITHMS.md and `fixtures/inference.json` (3 cases with counts and p_one).

## 2. Exhaustive search (k ≥ 1): `inference/exhaustive.py`

There are only 2¹⁸ = 262,144 rules, so simulate them all.
```python
def consistent_rules(x0, xk, k, boundary="wrap", chunk=8192) -> np.ndarray   # codes whose k-step output == xk
def ambiguity(codes) -> dict   # n_consistent, bits fixed across the set, per-bit agreement
```
Verified: 24×24, ρ₀ = 0.4, k = 2, true rule Life: **2.7 s** for all 262,144 rules, **exactly 1** consistent rule, all 18 bits fixed.
**Point to make in the content:** for life-like rules, brute force is cheaper and exact, so ML is unnecessary. ML earns its place when the rule space is too big to enumerate. For example, isotropic non-totalistic rules (Hensel notation) have 2¹⁰² rules.

## 3. Rule → ReLU network compiler (exact; a theorem plus code): `inference/relu_compiler.py`

**Claim (verified):** every life-like rule is computed *exactly* by
Conv3×3(1→1, all-ones kernel with centre weight ½) → ReLU units → linear readout.

**Construction.** v = n + s/2 takes the 18 distinct values 0, 0.5, …, 8.5, where v_k = k/2. Even k means s = 0, n = k/2 (birth entry). Odd k means s = 1, n = (k−1)/2 (survival entry). Let y_k = L[s,n]. The piecewise-linear interpolant is
g(v) = y₀ + Σ_{j=0}^{16} c_j·ReLU(v − j/2), with slopes m_j = (y_{j+1} − y_j)/0.5, c₀ = m₀, c_j = m_j − m_{j−1}.
Hidden units = number of nonzero c_j.

```python
@dataclass(frozen=True)
class ReluNet: centre_weight: float; biases: FloatArray; out_weights: FloatArray; out_bias: float
def compile_rule(rule: LifeRule) -> ReluNet
def apply(net, grid, boundary="wrap") -> FloatArray       # exact 0/1 output
```
Verified:
- **Life needs 4 hidden units**: kinks at v = 2, 2.5, 3.5, 4 with c = (+2, −2, −2, +2), i.e. 2·[ReLU(v−2) − ReLU(v−2.5) − ReLU(v−3.5) + ReLU(v−4)].
- The compiled nets are exact on 300 random rules × random grids.
- Over all 2¹⁸ rules, hidden units range from **0 to 17, mean exactly 12.5**. Histogram (index = units): `[2,2,34,62,296,652,1786,3742,7622,13586,22046,31810,40580,44864,41782,31186,16924,5168]`.

Tests: those exact values; `apply(compile_rule(r), g) == lifelike.step(g, r)` for every catalog rule plus 200 random codes (hypothesis).
Export `inference/relu_examples.json` (`ReluExamples`) with compiled nets for the catalog rules plus the histogram, and `#relu-compiler` in ALGORITHMS.md. The frontend can show "this rule *is* this 4-neuron network" and animate the hidden units. This compiler also gives P13 its known-optimal architecture.

## 4. Differentiable CA, rule learning by gradient descent: `inference/differentiable.py` (torch)

Relax the LUT to logits θ ∈ ℝ^{2×9}, L = σ(θ). Soft step for x ∈ [0,1]:
n = Moore-sum(x) ∈ [0,8], interpolate ℓ_s(n) linearly between integer n for s ∈ {0,1}, and set
x' = (1 − x)·ℓ₀(n) + x·ℓ₁(n). This equals the exact CA on binary inputs.
Loss: BCE between the soft k-step rollout from x₀ and the observed x_k. Use Adam (lr 0.1), 500 steps, 10 random restarts.

**Theory to verify empirically:** for k = 1 with binary x₀, the loss is a sum over entries of logistic losses in θ[s,n], so it is **convex** and should succeed from every init. For k ≥ 2 the composition makes it non-convex.

**Study** (`cellauto analyze inference-study`; `--quick` = 3 rules, 2 restarts):
- 20 target rules (stratified by atlas cluster, fixed seed) × k ∈ {1,2,3,4} × ε ∈ {0, 0.02, 0.05} × 10 restarts, on 32×32 grids, ρ₀ = 0.4.
- Metrics: success rate (MAP rule reproduces the clean x_k, or matches the true rule up to bits exhaustive search marks as unidentifiable), bit error rate, wall-clock vs exhaustive search.
- Report with Wilson 95% CIs. Expected shape: ~100% at k = 1, then declining with k. **Report whatever you get.**

Export `inference/study.json` (`InferenceStudy`).

**Commits:** `feat(inference): Bayesian rule posterior, exhaustive search, exact ReLU compiler` and `feat(inference): differentiable CA rule learning study`
