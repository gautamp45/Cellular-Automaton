# P14: NCA retraining (GPU recommended)

**Tier:** Could · **Prereqs:** P08, P12 · **Extra:** `[train]`

**Why:** (1) The current targets are opaque 28×28 squares, so "growth" is filling a square. (2) Every configuration was trained once, so the P12 filter/loss conclusions are confounded with training luck.

**Cost estimate:** one iteration ≈ batch 8 × ~80 steps × 72² cells × ~8.2K MACs × 3 (forward + backward) ≈ 8×10¹⁰ FLOP. That's roughly 1–3 s on a laptop CPU and much faster on a GPU, so 8000 iterations ≈ 4–6 h CPU per model. **Use a free Colab/Kaggle GPU.** Add `scripts/colab_train.ipynb`, a 5-cell notebook that clones the repo, runs `pip install -e .[train]`, runs `cellauto nca train --config ...`, and downloads `models/nca/*.npz`. It's a thin wrapper; all logic stays in the package.

## Experiments (each a YAML in `configs/nca/`; each run writes `runs/<id>/`)
1. **Shaped targets:** add `assets/nca/targets/` RGBA targets with real transparency:
   - 2–3 Noto Emoji PNGs (Apache-2.0; e.g. 🦎 U+1F98E, 🦋 U+1F98B), downloaded once to `assets/nca/targets/emoji_<code>.png`, resized to 40 px, padding 16. This mirrors Distill so the results are comparable.
   - One MedMNIST image with a circular alpha mask, to keep the medical thread.
   - If the network is blocked, ask the user to provide the files. Don't synthesise fake "emoji".
2. **Distill's three regimes** on one target: growing (no pool), persistent (pool), regenerating (pool + damage). Evaluate each with the P12 protocol (regen only applies where it's meaningful).
3. **Training-seed variance:** rerun the filter ablation (sobel, scharr, laplacian) × 3 training seeds on one shaped target. Now the CIs include training variance, and you can re-check the filter effect properly.
4. **Optimisation ablation:** `grad_norm` on/off × `lr_schedule` on/off (2×2, 3 seeds).
5. **Could, isotropy:** laplacian-only perception with a symmetry-breaking seed. Mordvintsev, Randazzo & Fouts, *Growing Isotropic NCA*, arXiv:2205.01681 (2022), show isotropic NCAs need structured seeds to break symmetry. Read the paper and implement the simplest variant (e.g. a two-cell seed with different hidden states). This ties the repo's original isotropic-filter failure to the literature.

## Output
New checkpoints go through the manifest (ids like `lizard-l2-sobel-regen-s0`). P12's evaluation runs on them, and the export adds them to `nca/models.json` automatically. **No frontend change is needed**, because the model picker is data-driven.

Add `docs/model_cards/nca.md` entries for every new model.

**Commit:** one per experiment, e.g. `exp(nca): regenerating lizard, 3 training seeds`
