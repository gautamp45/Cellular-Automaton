# CLAUDE.md

Guidance for Claude Code sessions in this repo. Keep it short and current; update it when the repo's shape changes (P01, P07 and P15 should each revisit it).

## What this repo is

`cellauto` is a Python package (being built) with reference implementations and analyses of cellular automata: elementary (1D), life-like (2D), Lenia (continuous) and Neural CA. It exports a **static, schema-validated data backend** (`web/public/data/`) for a future client-side frontend. There is no server.

**Current state:** mid-refactor, driven by a written plan. Progress lives in `docs/plan/STATUS.md`. Until P00–P08 are done, the top-level legacy scripts (`OD_CA.py`, `GameofLife.py`, `TD CA/`, `neural-cellular-automata/`, …) still exist and are being replaced phase by phase. Don't fix or extend legacy code; replace it as the plan says.

## Start every session with

1. `git status && git log --oneline -5`
2. Read `docs/plan/STATUS.md` (what's done and what's next) and, on your first session, `docs/plan/HANDOFF.md` (branches, environment, pitfalls).
3. If `docs/plan/` is missing on your branch, follow HANDOFF §2 to merge `origin/claude/zen-clarke-bey6eo`.
4. Read the phase file you're executing: `docs/plan/P<NN>-*.md`.

## Non-negotiable rules

- Follow the phase file. If it contradicts the repo, stop and ask; don't improvise.
- **Never edit a "verified" value to make a test pass.** Find the bug. Evidence is in `docs/plan/reference/` (run its scripts from the repo root).
- Never change NCA maths beyond the fixes listed in P05/P08. All 25 checkpoints must keep loading and reproducing.
- No dependencies beyond P01's list (core: numpy, pillow, pydantic; `[train]`: torch, pyyaml, tqdm; `[dev]`: pytest, hypothesis, ruff, mypy; P15 may add `[docs]`: matplotlib).
- Exports to `web/public/data/` must be deterministic: sorted keys, no timestamps. After P07, contract changes are **additive only** (see `docs/DATA_CONTRACT.md`).
- Use `git mv` for moves. One phase per commit (or two, as the phase file says), using its commit message.
- Update `docs/plan/STATUS.md` in the same commit that completes or pauses a phase.
- Don't create PRs unless asked. Push only to your session's designated branch.

## Commands (from P01 on)

```bash
python3 -m pip install -e ".[dev]"     # the SessionStart hook does this automatically in cloud sessions
ruff check . && ruff format --check . && mypy src && pytest -q    # must pass before every commit
pytest -q -m slow                       # long tests (atlas, NCA convergence); CI runs them weekly
cellauto export-web --check             # from P07: fails if committed web data is stale
```
If `pytest` can't import `cellauto`, you're hitting the uv-tool shim in `/root/.local/bin`. Use `python3 -m pytest` (HANDOFF §3).

## Environment quick facts (cloud sandbox)

- Torch **cannot** be installed here (`download.pytorch.org` is blocked, and the PyPI CUDA wheel won't import). Torch tests `importorskip` and run in CI. Training runs locally or on Colab/Kaggle.
- PyPI and `raw.githubusercontent.com` are reachable.
- GitHub operations go through the MCP GitHub tools, not `gh`.

## Layout (target; see docs/plan/README.md §4)

`src/cellauto/{engines,catalog,nca,analysis,inference,learnability,contract,export}` · `models/nca/` · `assets/nca/` · `content/` · `results/` · `web/public/data/` (generated, committed) · `experiments/` (archived legacy, not linted) · `docs/{plan,ALGORITHMS.md,DATA_CONTRACT.md,REPORT.md}`
