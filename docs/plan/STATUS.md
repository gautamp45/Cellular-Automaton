# Build-out status

**Update this file in the same commit that completes (or partially completes) a phase.** It's the first thing a fresh session reads after `CLAUDE.md`.

Legend: ⬜ not started · 🟨 in progress / partial · ✅ done · ⏭️ skipped (say why)

| Phase | Title | Tier | Status | Commit(s) | Notes |
|---|---|---|---|---|---|
| P00 | Hygiene, archive crypto, remove Rule Predictor | Must | ⬜ | | |
| P01 | Scaffold, tooling, CI | Must | ⬜ | | Remember to add `docs/plan/reference` to ruff `extend-exclude` |
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

- [ ] D2: delete `Rule Predictor/` (default yes)
- [ ] D3: archive crypto under `experiments/ca_crypto/` (default yes)
- [ ] D5: MIT license; confirm with collaborator Manan Shah
- [ ] Attribution: was any of `neural-cellular-automata/src` adapted from another public repo? If yes, record the URL here: ______
- [ ] Merge the planning branch into `main` (see HANDOFF.md §2)

If the owner hasn't answered, proceed with the defaults and note it in the session log.

## Session log (newest first)

Add one entry per session: date, branch, phases touched, what's left mid-phase, surprises.

- **2026-10-07, `claude/zen-clarke-bey6eo` (planning session):** audited the repo; wrote plan v1, then v2 (`docs/plan/`); verified every quoted number with the prototypes in `docs/plan/reference/`; added `CLAUDE.md`, `HANDOFF.md`, this file, and a SessionStart hook (`.claude/hooks/session-start.sh`). **No production code changed yet. Next: P00.**
