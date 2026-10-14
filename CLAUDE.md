# CLAUDE.md

Guidance for Claude Code sessions in this repo. Keep it short and current; update it when the repo's shape changes (P01, P07 and P15 should each revisit it).

## What this repo is

`cellauto` is a Python package with reference implementations and analyses of cellular automata: elementary (1D), life-like (2D), Lenia (continuous) and Neural CA, plus rule-inference studies. It exports a **static, schema-validated data backend** (`web/public/data/`) for a client-side frontend. There is no server.

**Current state:** built through phase **P12** of the written plan (all legacy scripts are gone; `experiments/ca_crypto/` is the archived remainder). Open phases: P13 (learnability, torch), P14 (NCA retraining, GPU), P15 (README/report/model cards/licence). Progress lives in `docs/plan/STATUS.md`. The frontend can start now (see HANDOFF §6); backend changes from here are additive only.

## Start every session with

1. `git status && git log --oneline -5`
2. Read `docs/plan/STATUS.md` (what's done and what's next) and, on your first session, `docs/plan/HANDOFF.md` (branches, environment, pitfalls).
3. If `docs/plan/` or `src/cellauto/` is missing on your branch, merge `origin/main` into it (HANDOFF §2); `main` has everything.
4. Read the phase file you're executing: `docs/plan/P<NN>-*.md`.

## Non-negotiable rules

- Follow the phase file. If it contradicts the repo, stop and ask; don't improvise.
- **Never edit a "verified" value to make a test pass.** Find the bug. Evidence is in `docs/plan/reference/` (run its scripts from the repo root).
- Never change NCA maths beyond the fixes listed in P05/P08. All 25 checkpoints must keep loading and reproducing.
- No dependencies beyond P01's list (core: numpy, pillow, pydantic; `[train]`: torch, pyyaml, tqdm; `[dev]`: pytest, hypothesis, ruff, mypy; P15 may add `[docs]`: matplotlib).
- Exports to `web/public/data/` must be deterministic: sorted keys, no timestamps. After P07, contract changes are **additive only** (see `docs/DATA_CONTRACT.md`).
- Use `git mv` for moves. One phase per commit (or two, as the phase file says), using its commit message.
- Update `docs/plan/STATUS.md` (and any doc whose subject changed) in the same commit that completes or pauses a phase.
- Long analyses go behind `cellauto analyze … [--quick]`; their results live in `results/` and are exported by `export-web`. Re-run `cellauto export-web` and commit `web/public/data/` whenever data, contract or content change.
- Don't create PRs unless asked. Push only to your session's designated branch.

## Commands (from P01 on)

```bash
python3 -m pip install -e ".[dev]"     # the SessionStart hook does this automatically in cloud sessions
ruff check . && ruff format --check . && mypy src && pytest -q    # must pass before every commit
pytest -q -m slow                       # long tests (atlas, NCA convergence); CI runs them weekly
cellauto export-web --check             # from P07: fails if committed web data is stale
cellauto analyze {eca-atlas|lifelike-atlas|inference-study|nca-eval} [--quick] [--workers N]
                                        # long analyses -> results/*.json (committed); then cellauto export-web
```
Runtimes, the new-dataset recipe and the numbers to quote are in `docs/plan/STATUS.md`.
CI runs numpy 2.2.x on Python 3.10/3.12 (the sandbox has newer numpy): also check `PYTHONPATH=/tmp/np22 python3 -m mypy src` after `pip install --target /tmp/np22 numpy==2.2.6` (HANDOFF §5).
If `pytest` can't import `cellauto`, you're hitting the uv-tool shim in `/root/.local/bin`. Use `python3 -m pytest` (HANDOFF §3).

## Environment quick facts (cloud sandbox)

- `download.pytorch.org` is blocked, but torch **can** be installed from PyPI in an isolated directory (it drags in ~5 GB of NVIDIA wheels, ~25 min on the sandbox proxy, CPU-only use is fine):
  `pip install --target /tmp/torchenv torch pyyaml tqdm` then `PYTHONPATH=/tmp/torchenv python3 -m pytest -q -m torch`. Without it, torch tests `importorskip` and run in CI. Long training runs belong on a laptop or Colab/Kaggle.
- PyPI and `raw.githubusercontent.com` are reachable. 4 CPUs / 15 GB RAM: use `--workers` and `OPENBLAS_NUM_THREADS=1` for long runs; foreground `sleep` is blocked (use background jobs).
- GitHub operations go through the MCP GitHub tools, not `gh`.

## Layout (see docs/plan/README.md §4)

`src/cellauto/{engines,catalog,nca,analysis,inference,contract,export}` (+ `learnability/` in P13) · `models/nca/` · `assets/nca/` · `content/` · `results/` · `web/public/data/` (generated, committed) · `experiments/` (archived legacy, not linted) · `docs/{plan,ALGORITHMS.md,DATA_CONTRACT.md,REPORT.md}`
