# Handoff: picking up the build-out in a fresh session

This is for whoever (human or AI) continues this work in a new session with no memory of earlier ones. Read it once, fully. After that, `CLAUDE.md` plus `STATUS.md` are enough for each session.

---

## 1. Where things stand

- **Goal:** turn this repo into the `cellauto` package plus a static data backend (`web/public/data/`) for a frontend showcasing 1D, 2D (Life + variants), Lenia and Neural CA, with math/ML analysis. The frontend comes later and must not need backend changes.
- **Plan:** `docs/plan/README.md` (overview, principles, phase table, frontend-completeness matrix, rules) plus one file per phase, `P00`–`P15`. The audit of the original code is in `AUDIT.md`.
- **Progress:** `docs/plan/STATUS.md`. As of the planning session, **no production code has changed**. The repo is still the original scripts, and the next phase is **P00**.
- **Evidence:** `docs/plan/reference/` has the scripts that produced every "verified" number.

## 2. Branches (read before your first git command)

The plan was written on branch `claude/zen-clarke-bey6eo`. A fresh cloud session is usually given a **new** branch created from `main`. If `main` doesn't contain `docs/plan/`, the new session won't see the plan.

**Preferred:** before the next session, merge `claude/zen-clarke-bey6eo` into `main` (open a PR on GitHub and merge it). Every later session then starts from a branch that already has the plan. After each session, merge that session's branch the same way, so the next one starts from the latest work.

**If it isn't merged yet,** the session must pull the planning branch into its designated branch first:
```bash
git fetch origin claude/zen-clarke-bey6eo
git merge --no-edit origin/claude/zen-clarke-bey6eo   # brings docs/plan, CLAUDE.md, .claude/
```
The same applies to work from earlier build sessions that hasn't been merged: fetch and merge the previous session's branch (its name is in the STATUS session log). **Check `STATUS.md` after merging**; it's the source of truth for what's done.

Never force-push or rewrite history on a branch someone else has pushed to.

## 3. Environment facts learned the hard way (cloud container, Oct 2026)

| Fact | Consequence |
|---|---|
| System Python is 3.13 and pip installs to the system (not externally managed). numpy 2.x and Pillow 12 are preinstalled. | `pip install -e ".[dev]"` works without a venv. |
| `/root/.local/bin/{pytest,mypy,ruff}` are **uv-tool shims with their own interpreter**, first on PATH. | Bare `pytest` **cannot import `cellauto`**. The SessionStart hook prepends `/usr/local/bin` via `$CLAUDE_ENV_FILE`. If in doubt, use `python3 -m pytest` and `python3 -m mypy`. |
| `.claude/hooks/session-start.sh` runs at session start. It is a no-op until `pyproject.toml` exists (P01), then installs `cellauto[dev]` editable (~13 s). | After P01 you shouldn't need to install anything by hand. If you add an extra later, re-run `python3 -m pip install -e ".[dev]"`. |
| `download.pytorch.org` is **blocked** (proxy 403). PyPI's `torch` wheel is the CUDA build: it downloads, but import fails without GBs of NVIDIA libs. | Don't try to install torch in the cloud sandbox. Torch tests use `pytest.importorskip("torch")`, and CI's `torch` job runs them. Run training locally or on Colab/Kaggle. |
| PyPI and `raw.githubusercontent.com` are reachable. | P04's `animals.json` download works; Noto emoji for P14 should too. |
| Pillow ≥ 10 removed `Image.ANTIALIAS`. | Use `Image.Resampling.LANCZOS` / `NEAREST`. |
| GitHub access is via MCP tools, not `gh`. | For PRs, use the GitHub MCP tools. Create a PR only when the owner asks. |

## 4. How to run a build session

1. `git status`, `git log --oneline -5`, then read `CLAUDE.md` and `docs/plan/STATUS.md`.
2. Pick the **first phase not marked ✅** (or the one the owner names). Read `docs/plan/README.md` §2 (principles) and §7 (rules) once per session, then the phase file in full.
3. Implement exactly what the phase file says. Where it gives code, use it. Where it gives verified values, test against them **unchanged**.
4. Run the checks: `ruff check . && ruff format --check . && mypy src && pytest -q` (from P01 on).
5. Update `STATUS.md`: the phase row, commit SHAs, a session-log entry, and anything half-done (be specific: file, function, failing test).
6. Commit with the phase's commit message and push to your designated branch.
7. Prefer **one phase per session** for P07, P09–P12 (they're large). Small phases (P00–P02, P06) can share a session.

**Stopping mid-phase is fine** if STATUS says exactly where you stopped. Commit work-in-progress only if checks pass. Otherwise describe the state in STATUS and leave it uncommitted only if the session continues; never leave a red commit on the branch.

## 5. Pitfalls already discovered (don't rediscover them)

- **GF(2) nilpotency:** rule 90 (and 2D Replicator B1357/S1357) die completely on rings or tori whose size is a power of 2. Every analysis protocol uses prime sizes (251, 61, 127) for this reason. Don't "simplify" them to 256 or 64.
- **NCA checkpoint compatibility:** the conv kernel must be `register_buffer(..., persistent=False)`. A persistent buffer breaks `strict=True` loading of all 25 checkpoints.
- **NCA perception layout:** index = `3*channel + kernel`. The alive mask pads with −∞ (out of bounds is ignored), while the perception pads with 0. Alive must hold both before and after the update.
- **Damage masks in evaluation:** use separate RNG streams for the fire mask and the damage mask, or comparisons between models aren't paired.
- **Recovery metric:** only defined for converged models (MSE@299 ≤ 0.005). Otherwise it reports nonsense like "recovered in 1 step".
- **Mean-field roots:** near-degenerate roots (Coral) need deduplication and a "marginal" label.
- **Gosper gun RLE:** use exactly the string in P03 (`...12b2o6b2o12b2o...`). A one-character typo produces a pattern that silently decays.
- **Exports must be deterministic:** no timestamps or run metadata in `web/public/data/`; those go in `results/*.json` meta.
- **B0 rules** strobe. The atlas excludes them by design, and the B0-free set is *not* closed under duality.

## 6. Kickoff prompts (copy-paste into a new session)

**Continue the build-out (default):**
> Continue the cellauto build-out. First make sure this branch contains `docs/plan/` (if not, follow docs/plan/HANDOFF.md §2 to merge `claude/zen-clarke-bey6eo`). Then read CLAUDE.md and docs/plan/STATUS.md, execute the next unfinished phase exactly as its phase file specifies, run all checks, update STATUS.md, commit and push.

**A specific phase:**
> Read CLAUDE.md and docs/plan/STATUS.md, then execute docs/plan/P0X-<name>.md only. Don't start the next phase. Update STATUS.md, commit and push.

**Frontend kickoff (after P07, ideally after P12):**
> Read CLAUDE.md, docs/plan/README.md §5 (frontend-completeness matrix), docs/DATA_CONTRACT.md and docs/ALGORITHMS.md. Plan the frontend in `web/` (Vite + React + TypeScript, static, client-side simulation, Vitest parity tests against web/public/data/fixtures). Don't modify anything under src/ or the export format. If something the UI needs is missing from the data, stop and report it as a contract gap.

## 7. Who to ask

The owner (Gautam) decides the open items in `STATUS.md` → "Open decisions". If a phase file contradicts the repo state, or a verified value seems wrong after checking it with `docs/plan/reference/`, stop and ask. Don't improvise around the plan.
