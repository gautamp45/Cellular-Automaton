# Build-out status

**Update this file in the same commit that completes (or partially completes) a phase.** It's the first thing a fresh session reads after `CLAUDE.md`.

Legend: ⬜ not started · 🟨 in progress / partial · ✅ done · ⏭️ skipped (say why)

| Phase | Title | Tier | Status | Commit(s) | Notes |
|---|---|---|---|---|---|
| P00 | Hygiene, archive crypto, remove Rule Predictor | Must | ✅ | see log | Also removed the monorepo `packages/` + `tests/` from `main` (owner decision, see log) |
| P01 | Scaffold, tooling, CI | Must | ✅ | see log | ruff `extend-exclude` uses `docs/plan` (not just `/reference`): ruff's formatter also formats the code blocks in the plan's `.md` files |
| P02 | Elementary engine | Must | ⬜ | | |
| P03 | Life-like engine, RLE, catalogs | Must | ⬜ | | |
| P04 | Lenia | Should | ⬜ | | Needs a one-time download of `animals.json` (GitHub raw) |
| P05 | NCA torch-free inference + manifest | Must | ⬜ | | |
| P06 | Renderer + CLI | Must | ⬜ | | |
| P07 | Contract, export, fixtures, ALGORITHMS.md | Must | ⬜ | | Frontend may start after this (zero-change after P12) |
| P08 | NCA torch training port | Must | ⬜ | | Torch not installable in cloud sandbox; torch tests skip locally, CI runs them |
| P09 | ECA theory + behaviour atlas | Should | ⬜ | | |
| P10 | Life-like theory + full rule-space atlas | Should | ⬜ | | Full atlas ≈ 40 min CPU; chunked and resumable |
| P11 | Rule inference + ReLU compiler | Should | ⬜ | | Differentiable part needs torch |
| P12 | NCA rigorous evaluation | Should | ⬜ | | ≈ 15–20 min CPU |
| P13 | Learnability experiment | Could | ⬜ | | Needs torch |
| P14 | NCA retraining | Could | ⬜ | | Needs a GPU (Colab/Kaggle) and an emoji download |
| P15 | Docs, report, model cards | Must | ⬜ | | Run last; rerun after optional phases |

## Open decisions (from plan README §6), still to be confirmed by the owner

- [x] D2: delete `Rule Predictor/` (done in P00)
- [x] D3: archive crypto under `experiments/ca_crypto/` (done in P00)
- [ ] D5: MIT license; confirm with collaborator Manan Shah
- [ ] Attribution: was any of `neural-cellular-automata/src` adapted from another public repo? If yes, record the URL here: ______
- [ ] Merge the planning branch into `main` (see HANDOFF.md §2)

If the owner hasn't answered, proceed with the defaults and note it in the session log.

## Session log (newest first)

Add one entry per session: date, branch, phases touched, what's left mid-phase, surprises.

- **2026-10-14 (cont.), `claude/keen-turing-np4r23`:** **P01 done** (`pyproject.toml`, `src/cellauto/{__init__,__main__,cli,_types,_validate}.py`, `py.typed`, `tests/test_smoke.py`, `.github/workflows/ci.yml`). Deviation from the phase file: ruff `extend-exclude` has `docs/plan` instead of `docs/plan/reference`, because the installed ruff also lints/formats fenced code in `docs/plan/*.md`. Checks: ruff, ruff format, mypy, pytest all pass; SessionStart hook validated (`CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh`). **Next: P02.** (The owner asked that docs be updated at every step: each phase commit includes its STATUS row + log entry, and `CLAUDE.md` / `docs/` where the repo's shape changed.)

- **2026-10-14, `claude/keen-turing-np4r23`:** merged `main` already contained the plan. **P00 done.** **Surprise:** `main` had a parallel monorepo reorganisation (e68a05f) that the plan didn't know about: it *copied* the legacy code into `packages/{neural_ca,ca_cryptography,classical_ca}` (snake_case names, some small fixes, whitespace) and added `tests/test_*.py`, a `.gitignore` (which ignored `*.pt`, `**/data/*.png`, `**/animations/`) and a monorepo `README.md`, leaving the original dirs in place. It also had already removed the committed `__pycache__`/`.DS_Store`. Owner decisions: (1) delete `packages/` and the legacy `tests/` and follow the plan layout; (2) overwrite `.gitignore` with the P00 version. Done on top of the P00 steps: `.vscode/` untracked, `TD CA/UI.py` and `neural-cellular-automata/.gitignore` removed, crypto archived to `experiments/ca_crypto/` (original, unpatched copy) with README, `Rule Predictor/` and the Distill notebook deleted. **Left for later:** the root `README.md` still describes the deleted `packages/` monorepo (stale until P15 rewrites it); `GameofLife.py`, `OD_CA.py`, `TD CA/`, `TD_Rule_Web Scraper.py` and `neural-cellular-automata/` remain until their phases replace them. Nothing to lint or test yet (no `pyproject.toml` until P01). **Next: P01.**

- **2026-10-07, `claude/zen-clarke-bey6eo` (planning session):** audited the repo; wrote plan v1, then v2 (`docs/plan/`); verified every quoted number with the prototypes in `docs/plan/reference/`; added `CLAUDE.md`, `HANDOFF.md`, this file, and a SessionStart hook (`.claude/hooks/session-start.sh`). **No production code changed yet. Next: P00.**
