# cellauto

Cellular automata, from hand-written rules to learned rules: **elementary** (1D), **life-like** (2D),
**Lenia** (continuous) and **neural CA**, with reference implementations, mathematical analysis, rule
inference, and a static, schema-validated **data backend** for a client-side web frontend.

> Status: the Python package and data backend are built through phase P12 of
> [`docs/plan/`](docs/plan/README.md) (progress in [`docs/plan/STATUS.md`](docs/plan/STATUS.md)). The
> frontend has not been started; the learnability study (P13), NCA retraining (P14) and the final
> report phase (P15) are still open.

## The idea

Elementary CA (8-bit rules, exact maths) → life-like CA (18-bit rules, emergent structure) → Lenia
(continuous state, space and time) → neural CA (the update rule *is* a neural network learned by
gradient descent). Then the arrow is reversed: given observed dynamics, recover the rule.

| Layer | What is here |
|---|---|
| Elementary | vectorised engine; 88 symmetry classes; surjectivity / reversibility; preimage counting by transfer matrices; Lucas closed form for rule 90; behaviour atlas of all 256 rules |
| Life-like | table-driven engine, RLE, 27 named rules, 12 patterns; duality, mean-field theory; **atlas of all 131,072 B0-free rules** |
| Lenia | FFT engine, Orbium presets, Life as a special case |
| Neural CA | torch-free inference for 25 trained models; torch training; multi-seed evaluation (30 seeds) with bootstrap CIs and paired tests: direction-aware filters converge, symmetric filters and the Hinge loss do not, and training with damage regenerates ~2.4× faster |
| Inverse problems | exact Bayesian posterior, exhaustive search over all 2^18 rules, exact ReLU compiler, differentiable-CA study |

## Install and use

```bash
python3 -m pip install -e ".[dev]"           # numpy, pillow, pydantic (+ pytest, ruff, mypy, hypothesis)
python3 -m pip install -e ".[dev,train]"     # adds torch, pyyaml, tqdm for training and the inference study

cellauto elementary --rule 30 --out rule30.png
cellauto life --preset highlife --pattern glider --steps 120 --out life.gif
cellauto lenia --preset O2u --steps 300 --out orbium.gif
cellauto nca run --model chest-l2-sobel-damage --damage-at 100 --out chest.gif
cellauto list {elementary|lifelike|patterns|lenia|models}
cellauto export-web [--check]                # regenerate / verify web/public/data
cellauto analyze {eca-atlas|lifelike-atlas|inference-study|nca-eval} [--quick]
cellauto nca train --config configs/nca/chest-l2-sobel-damage.yaml
```

Checks before every commit: `ruff check . && ruff format --check . && mypy src && pytest -q`
(`pytest -q -m slow` for the long tests, `pytest -q -m torch` with torch installed).

## Layout

```
src/cellauto/   engines/ catalog/ nca/ analysis/ inference/ contract/ export/ render.py cli.py
models/nca/     25 trained checkpoints (.pt, torch-free .npz) and manifest.json
assets/nca/     target images and growth animations
results/        committed outputs of the analyses (atlases, inference study, NCA evaluation)
web/public/data GENERATED static data backend for the frontend (committed; `export-web --check` in CI)
content/        explainer pages exported to the frontend
docs/           ALGORITHMS.md (browser implementation spec), DATA_CONTRACT.md, plan/
experiments/    archived legacy code (a CA stream-cipher experiment), not maintained
```

The contract between Python and the frontend is [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md); every
algorithm the browser must re-implement is specified, with golden fixtures, in
[`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Credits

Lenia: Bert Wang-Chak Chan, [Chakazul/Lenia](https://github.com/Chakazul/Lenia) (MIT), source of the
Orbium presets. Neural CA: Mordvintsev, Randazzo, Niklasson and Levin, *Growing Neural Cellular
Automata*, Distill 2020. Targets are 28×28 MedMNIST images. The archived cipher experiment includes work
by Manan Shah. Upstream references are listed in [`docs/plan/README.md`](docs/plan/README.md).

License: MIT is declared in `pyproject.toml`; the licence file and the collaborator's confirmation are
still open items (see `docs/plan/STATUS.md`).
