# Cellular-Automaton: Revamp Plan (v2)

This directory is the complete, executable plan to turn this repo into a polished **research-grade library + static data backend** for a frontend that showcases elementary (1D), life-like (2D), continuous (Lenia), and neural cellular automata.

It is written for an AI coding agent (or a human) to execute **one phase file at a time**, without further design work. Every number quoted as "verified" was produced by running reference code during planning (Oct 2026). If a test built on a verified value fails, **the implementation is wrong, not the value.**

---

## 1. The story this repo tells

The repo is organised around a single narrative, which the frontend follows page by page:

> **From hand-written rules to learned rules.**
> Elementary CA (8-bit rules, exact math) → Life-like CA (18-bit rules, 2D, emergent structure) → Lenia (continuous state, space and time) → Neural CA (the update rule *is* a neural network, learned by gradient descent). Then we **invert** the arrow: given observed dynamics, recover the rule (Bayesian estimation, exhaustive search, gradient descent), and ask how hard it is for neural networks to *learn* a CA at all.

Each layer shows a different skill set:

| Layer | Math showcased | ML showcased | Engineering showcased |
|---|---|---|---|
| Elementary | group actions (88 symmetry classes), de Bruijn graphs and surjectivity, transfer matrices (preimage counting), GF(2) linear algebra (Lucas' theorem, nilpotency) | unsupervised behaviour classification (features, PCA, k-means) vs Wolfram classes | vectorised numpy, property-based tests |
| Life-like | B/S duality (an involution on rule space), mean-field theory with fixed-point stability, Langton's λ | rule-space atlas of **all 131,072** non-B0 rules; edge-of-chaos hypothesis test | batched simulation, binary columnar export |
| Lenia | FFT convolution, smooth kernels, Life as a special case of Lenia | — | float parity across languages |
| Neural CA | perception as fixed convolution, stochastic updates | rigorous ablation (filters × losses, 30 seeds, paired tests, bootstrap CIs), regeneration metrics, retraining | torch-free inference, checkpoint conversion |
| Inverse problems | Bayesian posterior over rule bits, identifiability | differentiable CA (convex at k=1, non-convex beyond), exact ReLU compiler for every life-like rule, learnability experiment (Springer & Kenyon 2020) | exhaustive search over 2¹⁸ rules in ~3 s |

---

## 2. Architecture principles (apply in every phase)

1. **Python is the source of truth and the build step. The browser is the runtime.** There is no server. Python produces versioned, schema-validated static data (catalogs, model weights, precomputed analyses, golden fixtures, explainer text) into `web/public/data/`. The frontend only reads it.
2. **Frontend-complete contract.** The frontend must never need a backend change to add a page or a control. That holds because:
   - every engine publishes its **parameter spec** (name, type, range, default, description) in `engines.json`, so UI controls are generated from data;
   - every algorithm the browser re-implements has an **exact spec** in `docs/ALGORITHMS.md` plus **golden fixtures**;
   - every number or chart the frontend shows comes from an exported dataset with a JSON Schema;
   - explanatory prose lives in `content/*.md` and is exported, not hard-coded in React.
3. **Functional core:** engines are pure `state in → state out` numpy functions. No globals, no UI, no work at import time.
4. **Table-driven rules:** elementary rules are an 8-entry LUT, and life-like rules a `(2, 9)` LUT indexed `lut[state, count]`. This maps 1:1 to a TypeScript loop or a GPU shader uniform.
5. **Dependencies are tiered.** Core: `numpy`, `pillow`, `pydantic`. `[train]`: `torch`, `pyyaml`, `tqdm`. `[dev]`: `pytest`, `hypothesis`, `ruff`, `mypy`. Everything in the browser-facing path runs without torch.
6. **Deterministic outputs:** seeded RNG everywhere, sorted JSON keys, no timestamps, and a sha256 for every exported file. CI re-runs the export and fails on any diff.
7. **Science hygiene:** every stochastic result is reported with its seed count, a mean ± 95% bootstrap CI, and a paired test where two models are compared. Negative results are reported, not hidden.

---

## 3. Execution order

Execute phases **in numeric order**. Each phase file lists prerequisites, exact steps, a "Done when" checklist, and commit message(s). Tiers tell you what to cut if time runs out: **Must** is required for the frontend, **Should** is the core showcase value, and **Could** is optional stretch work.

| Phase | File | Tier | Needs torch | Est. effort |
|---|---|---|---|---|
| 00 | [P00-hygiene.md](P00-hygiene.md): junk removal, archive crypto, delete Rule Predictor | Must | – | S |
| 01 | [P01-scaffold.md](P01-scaffold.md): pyproject, tooling, CI | Must | – | S |
| 02 | [P02-elementary.md](P02-elementary.md): 1D engine | Must | – | S |
| 03 | [P03-lifelike.md](P03-lifelike.md): 2D engine, rulestrings, RLE patterns, catalogs | Must | – | M |
| 04 | [P04-lenia.md](P04-lenia.md): continuous CA engine + Orbium | Should | – | M |
| 05 | [P05-nca-inference.md](P05-nca-inference.md): torch-free NCA, checkpoint conversion, manifest | Must | – | M |
| 06 | [P06-render-cli.md](P06-render-cli.md): Pillow renderer + `cellauto` CLI | Must | – | S |
| 07 | [P07-contract-export.md](P07-contract-export.md): Pydantic contract, JSON Schema, export, fixtures, ALGORITHMS.md | Must | – | L |
| 08 | [P08-nca-training.md](P08-nca-training.md): port torch training with fixes | Must | yes | M |
| 09 | [P09-eca-theory.md](P09-eca-theory.md): symmetry, surjectivity, preimages, GF(2), behaviour atlas | Should | – | M |
| 10 | [P10-lifelike-theory.md](P10-lifelike-theory.md): duality, mean-field, λ, full rule-space atlas | Should | – | L |
| 11 | [P11-rule-inference.md](P11-rule-inference.md): Bayesian, exhaustive, differentiable inference + ReLU compiler | Should | partly | L |
| 12 | [P12-nca-evaluation.md](P12-nca-evaluation.md): rigorous multi-seed evaluation, regeneration, stats | Should | – | M |
| 13 | [P13-learnability.md](P13-learnability.md): can a CNN learn Life? (Springer & Kenyon 2020) | Could | yes | M |
| 14 | [P14-nca-retraining.md](P14-nca-retraining.md): alpha-masked targets, symmetry study (GPU recommended) | Could | yes | L |
| 15 | [P15-docs-report.md](P15-docs-report.md): README, technical report, model cards, license | Must | – | M |

**When to start the frontend:** for literally zero backend changes, start after **P12**, when every dataset in the §5 matrix exists. Starting after **P07** is also safe: P09–P14 only *add* datasets and fill fields already declared as optional, and the frontend discovers them via `index.json`. The pages for those datasets just wait for their phase.

---

## 4. Target repository layout (end state)

```
.
├── pyproject.toml   README.md   LICENSE   CITATION.cff   .gitignore   .gitattributes
├── .github/workflows/ci.yml
├── src/cellauto/
│   ├── __init__.py  __main__.py  py.typed  _types.py  _validate.py
│   ├── engines/      elementary.py  lifelike.py  lenia.py
│   ├── catalog/      __init__.py  rules.py  patterns.py  rle.py  data/*.json
│   ├── nca/          spec.py  numpy_model.py  checkpoints.py  image.py  evaluate.py
│   │   └── training/ model.py  losses.py  train.py                 # [train]
│   ├── analysis/     elementary.py  lifelike.py  atlas.py  stats.py  embed.py
│   ├── inference/    bayes.py  exhaustive.py  relu_compiler.py  differentiable.py (torch)
│   ├── learnability/ models.py  experiment.py                     # [train]
│   ├── contract/     models.py  schema.py                          # Pydantic v2 = the API
│   ├── export/       web.py  fixtures.py  binary.py  content.py
│   ├── render.py
│   └── cli.py
├── configs/          nca/*.yaml  experiments/*.yaml
├── content/          *.md   (explainer prose with YAML front-matter, exported to the frontend)
├── models/nca/       manifest.json  <id>.npz  <id>.pt
├── assets/nca/       targets/*.png  animations/<id>.gif
├── results/          (raw experiment outputs: csv/json, committed when small)
├── scripts/          convert_checkpoints.py  run_all_experiments.sh
├── tests/            mirrors src/ layout; tests/fixtures_py/ for python-only fixtures
├── web/public/data/  GENERATED by `cellauto export-web`; committed
├── experiments/ca_crypto/   archived legacy code (excluded from lint/tests/package)
└── docs/
    ├── plan/         this plan
    ├── ALGORITHMS.md   exact spec of every browser-side algorithm
    ├── DATA_CONTRACT.md  human-readable companion to the JSON Schemas
    ├── ARCHITECTURE.md   REPORT.md   ROADMAP.md
    └── model_cards/  nca.md
```

---

## 5. Frontend-completeness matrix (definition of done for the backend)

The backend is "frontend-complete" when every row has its data file, its algorithm spec (if the browser computes it live), and its fixture.

| Frontend feature | Data file(s) (in `web/public/data/`) | Live algorithm spec | Fixture | Phase |
|---|---|---|---|---|
| 1D explorer: rule picker, bit toggles, boundary, init modes | `engines.json`, `elementary/rules.json` | ALGORITHMS §ECA | `fixtures/elementary.json` | 02, 07 |
| 1D rule atlas: 256 rules, symmetry class, λ, surjective/reversible, behaviour metrics, cluster | `elementary/atlas.json` | – (precomputed) | – | 09 |
| 1D math demos: preimage counter, Rule 90 closed form | `elementary/atlas.json` | ALGORITHMS §TransferMatrix, §Lucas | `fixtures/elementary_math.json` | 09 |
| 2D playground: presets, B/S editor, patterns, draw, wrap/dead | `engines.json`, `lifelike/rules.json`, `lifelike/patterns.json` | ALGORITHMS §LifeLike, §RLE | `fixtures/lifelike.json`, `fixtures/rle.json` | 03, 07 |
| 2D duality toggle ("show the dual rule") | `lifelike/rules.json` (`dual` field) | ALGORITHMS §Duality | `fixtures/lifelike_math.json` | 10 |
| 2D mean-field panel: map f(ρ), fixed points, cobweb plot | `lifelike/rules.json` (`mean_field` field) | ALGORITHMS §MeanField | `fixtures/lifelike_math.json` | 10 |
| Rule-space atlas: 131,072-point scatter, colour by cluster, click a point to load its rule | `lifelike/atlas.header.json` + `lifelike/atlas.bin` | ALGORITHMS §RuleCode | – | 10 |
| Lenia: Orbium + presets, R/T/μ/σ sliders | `engines.json`, `lenia/presets.json` | ALGORITHMS §Lenia | `fixtures/lenia.json` | 04, 07 |
| "Life is a Lenia" toggle | `lenia/presets.json` (`life_as_lenia`) | ALGORITHMS §Lenia | `fixtures/lenia.json` | 04 |
| NCA lab: model picker, fire rate, click to damage | `nca/models.json`, `nca/weights/<id>.json`, `nca/targets/*.png` | ALGORITHMS §NCA | `fixtures/nca.json` | 05, 07 |
| NCA ablation: filter × loss grid with CIs, regeneration curves | `nca/evaluation.json`, `nca/animations/*.gif` | – | – | 12 |
| Rule inference: draw/run, infer the rule live with per-bit confidence | `engines.json` | ALGORITHMS §BayesInference | `fixtures/inference.json` | 11 |
| Inference study: success vs k, exhaustive vs gradient | `inference/study.json` | – | – | 11 |
| ReLU-compiler demo: rule ↔ network, step through hidden units | `inference/relu_examples.json` | ALGORITHMS §ReLUCompiler | `fixtures/inference.json` | 11 |
| Learnability: success probability vs width, loss curves | `learnability/results.json` | – | – | 13 |
| All explainer text | `content/<slug>.md` + `content/index.json` | – | – | 07, 15 |
| Discovery: what datasets exist, their versions and hashes | `index.json` | – | – | 07 |

---

## 6. Decision points (defaults chosen; change before execution if needed)

| # | Decision | Default |
|---|---|---|
| D1 | Package name | `cellauto` |
| D2 | `Rule Predictor/` | Delete. Its idea is rebuilt properly in P11. |
| D3 | `CA_Cryptography System/` | Archive as-is in `experiments/ca_crypto/` (credit collaborator Manan Shah). |
| D4 | Google Distill Colab notebook | Delete; link it in README. |
| D5 | License | MIT. Confirm with collaborator. |
| D6 | NCA GIFs (~15 MB) | Keep, moved to `assets/nca/animations/`. |
| D7 | Runtime | 100% client-side; no FastAPI. If a backend is ever wanted for portfolio reasons, give it real server work (e.g. a training queue), never the render loop. |
| D8 | Contract tech | Pydantic v2 models → JSON Schema → TS types generated by the frontend (`json-schema-to-typescript`). |
| D9 | Big tabular data | Custom documented binary columnar format (`header.json` + `.bin`), not Arrow. Zero JS dependencies, ~20 lines to parse. |

---

## 7. Rules for the executing agent

1. Work on the current feature branch. One phase = one or two commits, with the messages given in the phase file.
2. Before every commit: `ruff check . && ruff format --check . && mypy src && pytest -q`. All must pass. Mark slow tests `@pytest.mark.slow`. They are excluded by default (`-m "not slow"` in addopts) and run in CI's nightly or manual job.
3. Use `git mv` for moves so history is preserved.
4. **Never change a verified test vector to make a test pass.** Find the bug.
5. **Never change NCA maths** beyond the listed fixes. All 25 checkpoints must keep loading and producing the same results.
6. No new dependencies beyond those listed in P01.
7. If the repo state contradicts a phase file, stop and report rather than improvise.
8. Long computations (atlas, evaluation, experiments) go behind CLI commands with `--quick` variants for CI. The full outputs are committed under `results/` and exported.
9. Record runtime and machine info for every full experiment run in its results JSON (`meta.runtime_s`, `meta.python`, `meta.numpy`, `meta.torch`, `meta.cpu`, `meta.git_sha`). These go into results files, not web exports, which must stay deterministic.

---

## 8. Audit summary (why this plan exists)

The full audit is in [AUDIT.md](AUDIT.md). In short:
- **Three copies** of the CA logic, with spaces in directory names, no tests, no packaging, no license, and committed `__pycache__`/`.DS_Store`.
- **Verified bugs:** Game of Life edge neighbour count can be **−1**; the 1D CA never updates edge cells; Rule Predictor data is actually B23/S23 and its CSV truncates grids to `...`; the crypto RNG ignores its seed and can produce rule 256 or 0 generations; NCA `Image.ANTIALIAS` crashes on Pillow ≥ 10; the NCA ignores `fire_rate` and `--config`; the NCA kernel is not a buffer.
- **NCA facts:** all 25 checkpoints are one architecture (16 ch, 48→128→16, ~8.2K params). Inference is reproducible in ~60 lines of numpy. Sobel/Scharr reach MSE ≈ 1e-4; isotropic filters are 20–100× worse; Hinge plateaus. All targets have alpha = 255 everywhere, so "growth" fills a square.

## 9. References used by the plan

- Wolfram, *A New Kind of Science* (2002); Wolfram, "Statistical mechanics of cellular automata", Rev. Mod. Phys. 55 (1983).
- Hedlund, "Endomorphisms and automorphisms of the shift dynamical system" (1969); Amoroso & Patt, "Decision procedures for surjectivity and injectivity of parallel maps for tessellation structures" (1972).
- Langton, "Computation at the edge of chaos" (1990); Mitchell, Hraber & Crutchfield, "Revisiting the edge of chaos" (1993).
- Cook, "Universality in elementary cellular automata", Complex Systems 15 (2004).
- Chan, "Lenia: Biology of Artificial Life", Complex Systems 28(3) (2019); Lenia reference implementation and `animals.json` catalogue: https://github.com/Chakazul/Lenia (MIT).
- Mordvintsev, Randazzo, Niklasson & Levin, "Growing Neural Cellular Automata", Distill (2020).
- Mordvintsev, Randazzo & Fouts, "Growing Isotropic Neural Cellular Automata", arXiv:2205.01681 (2022).
- Springer & Kenyon, "It's Hard for Neural Networks To Learn the Game of Life", arXiv:2009.01398 (2020).
- Yang et al., "MedMNIST v2" (2023), the source of the three target images.
- LifeWiki (conwaylife.com) for rule names and pattern RLEs.
