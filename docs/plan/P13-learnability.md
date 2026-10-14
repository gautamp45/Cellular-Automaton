# P13: How hard is it for a neural network to learn Life?

**Tier:** Could · **Prereqs:** P11 (ReLU compiler) · **Extra:** `[train]` · **Files:** `learnability/{models,experiment}.py`, `configs/experiments/learnability.yaml`

**Background:** Springer & Kenyon, *It's Hard for Neural Networks To Learn the Game of Life*, arXiv:2009.01398 (2020). They found that small conv nets trained to predict n steps of Life rarely converge, and that reliable convergence needs far more parameters than the minimal network. They linked this to the lottery-ticket hypothesis. **Read §2–3 of the paper before implementing**, and cite it as the inspiration. This is *our* protocol on *our* architecture, not an exact replication, and the content must say so.

**Our twist (a selling point):** P11's compiler gives a provably exact network for Life in this architecture with **4 hidden units**. So we know the minimal-width solution exists and can ask whether gradient descent finds it.

## Architecture (one step, n = 1)
`Conv3×3(1→h, circular padding, bias) → ReLU → Conv1×1(h→1, bias)`, output logits, loss BCE-with-logits.
The compiled Life net is an instance with h = 4: every hidden channel's 3×3 kernel is ones with centre ½; the biases are −2, −2.5, −3.5, −4; the readout weights are α·(2, −2, −2, 2) and the readout bias is −α/2. The raw readout out ∈ {0,1} then becomes the logit α·(out − ½), so it has the correct sign at threshold 0. Use α = 10.

## Experiment (`cellauto experiment learnability`; `--quick` = h ∈ {4, 16}, 3 seeds, 200 steps)
- Width h ∈ {4, 5, 6, 8, 12, 16, 32, 64}, 30 seeds each.
- Data: fresh random 32×32 boards each batch, density ~ U(0.05, 0.95) per board; batch 64; Adam lr 1e-2; 3000 steps (on CPU ≈ seconds to a minute per run; record it).
- **Success** = 100% cell accuracy on 1000 held-out boards (threshold logit 0), with fixed held-out seed 12345.
- Report P(success | h) with Wilson 95% CIs, the median steps-to-success, and loss curves (median + IQR, 100 points).
- **Init control 1:** h = 4 initialised at the compiled solution plus Gaussian noise σ ∈ {0, 0.1, 0.3, 1.0}. Where does the basin end?
- **Init control 2:** h = 32 with 4 channels initialised from the compiled solution and the rest random (a "planted ticket").
- **Could:** n = 2 steps (two stacked blocks with shared weights) to probe the paper's multi-step finding.

## Output
`results/learnability.json` → exported as `learnability/results.json` (`LearnabilityResults`): per-h success, CI, curves; init-control tables; meta. Content page `learnability.md`: the question, the construction, and the result, honestly, whatever it is.

## Tests (`torch` marker)
- The compiled h = 4 net, loaded into the torch module, gets 100% accuracy on 1000 boards.
- `--quick` runs and writes a valid `LearnabilityResults`.

**Commit:** `feat(learnability): width vs trainability experiment for learning Life, with exact-solution controls`
