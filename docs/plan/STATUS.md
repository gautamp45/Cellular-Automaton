# Build-out status

**Update this file in the same commit that completes (or partially completes) a phase.** It's the first thing a fresh session reads after `CLAUDE.md`.

Legend: ⬜ not started · 🟨 in progress / partial · ✅ done · ⏭️ skipped (say why)

| Phase | Title | Tier | Status | Commit(s) | Notes |
|---|---|---|---|---|---|
| P00 | Hygiene, archive crypto, remove Rule Predictor | Must | ✅ | see log | Also removed the monorepo `packages/` + `tests/` from `main` (owner decision, see log) |
| P01 | Scaffold, tooling, CI | Must | ✅ | see log | ruff `extend-exclude` uses `docs/plan` (not just `/reference`): ruff's formatter also formats the code blocks in the plan's `.md` files |
| P02 | Elementary engine | Must | ✅ | see log | `OD_CA.py` removed |
| P03 | Life-like engine, RLE, catalogs | Must | ✅ | see log | `GameofLife.py`, `TD CA/`, `TD_Rule_Web Scraper.py` removed |
| P04 | Lenia | Should | ✅ | see log | `animals.json` downloaded once; the 3 presets are copied verbatim into `catalog/data/lenia_presets.json` |
| P05 | NCA torch-free inference + manifest | Must | ✅ | see log | 25 models moved to `models/nca/`, `.npz` + `manifest.json` generated |
| P06 | Renderer + CLI | Must | ✅ | see log | Only commands whose phase is done are registered; later phases add theirs |
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

- **P06 done.** `render.py` (Pillow only; magma LUT generated once from matplotlib 3.11 in a throwaway dir and pasted as a hex constant; `gray` LUT inline; palettes dark/light/green) and `cli.py` with `elementary`, `life`, `lenia`, `nca run`, `list {elementary,lifelike,patterns,lenia,models}`; `main(argv) -> int`, user errors → stderr + exit 2 (argparse usage errors too, `--help` → 0). **Deliberately not registered yet** (no stubs): `nca train` (P08), `export-web [--check]` (P07), `analyze …` (P09–P12), `experiment learnability` (P13); each of those phases must add its subparser in `cli.build_parser()`. Notes: Pillow merges identical consecutive GIF frames (so a static pattern stores fewer frames than supplied; playback unchanged), documented on `save_gif`; `life` defaults to a 0.3-density random soup when neither `--pattern` nor `--random` is given; `lenia` centres the preset cells in the grid; `nca run` uses the manifest's `grid_size`/`padding`, spawns separate RNG streams (fire mask vs damage mask) from `--seed`, and includes the seed as frame 0 (`steps + 1` frames). `rgb_to_image` treats a leading axis of 3 as channels-first. **Next: P07.**

- **P05 done.** Assets moved with `git mv`: 3 targets → `assets/nca/targets/`, 25 `.pt` → `models/nca/<id>.pt`, 25 GIFs → `assets/nca/animations/<id>.gif` (only `neural-cellular-automata/src/` remains, until P08). `scripts/convert_checkpoints.py [--manifest]` wrote 25 `.npz` + `models/nca/manifest.json` (torch-free reader; the torch cross-check path is implemented but **not exercised here**, no torch). New: `nca/{spec,image,numpy_model,checkpoints}.py`, `tests/nca/test_numpy_model.py`. Slow regression: all 13 sobel/scharr non-hinge models reach MSE@200 < 0.005 (**worst 0.00057**, plan said 0.0006); all 9 isotropic models are > 0.005 (**min 0.00939**; plan quoted ≥ 0.0095, a tiny float-precision difference, same conclusion, test threshold unchanged). Design notes for later phases: `NumpyNCA.run(x, steps, rng, damage_at, damage_rng)` yields the state after each update; for `t` in `damage_at` the state is multiplied by `1 − circle_mask(damage_rng)` right after update `t` (1-based) and the damaged state is what's yielded; `step(..., return_dx=True)` returns `(state, masked_dx)`; `NCAConfig` validates `fire_rate ∈ (0, 1]`; `find_repo_root()` tries CWD then the package location, so tests work from any directory; npz files are not byte-deterministic (zip timestamps), which is fine because they're committed inputs, not exports. **Next: P06.**

- **P04 done.** `engines/lenia.py` (`LeniaParams`, `kernel`, cached `kernel_fft`, `growth`, `step`, `run`, `decode_lenia_rle`, `life_as_lenia`), `catalog/data/lenia_presets.json` (O2u, O2ui, O2b from Chakazul/Lenia, MIT, plus the `life_as_lenia` non-pattern preset), `catalog.lenia_presets()`. Verified values passed unchanged: Orbium 20×20, sum 76.863, mass in [69, 74] for t = 100..500 in 128², glider displacement > 20 cells between t=100 and 200, Life-as-Lenia equals B3/S23 on 20 random grids, median 128² step < 10 ms. **Interpretation notes for P07/ALGORITHMS.md:** (1) `step(A, params, kfft=None)` takes the FFT of the origin-shifted kernel (origin at index `(H//2, W//2)`); `life_as_lenia()` returns `(params, kernel3x3)` rather than an FFT, and `step(A, life_params)` builds the 3×3 box kernel itself (`kernel_core='life_box'`, added to `LeniaParams.kernel_core`). (2) `growth='life_step'` rounds `u` with `np.rint` before comparing, to absorb FFT round-off. (3) State is returned as float32; kernel/FFT math is float64. (4) `lenia_presets.json` is `{license, source, presets: [...]}` (a wrapper, so the MIT credit lives in the file); `life_as_lenia` stores `mu=0, sigma=1` as unused placeholders. **Next: P05.**

- **P03 done** (two commits). `engines/lifelike.py` (LifeRule parse/code/table, batch-aware `neighbour_counts`, `step`, `step_batch`, `iterate`, `run`, `random_grid`), `catalog/{rle,patterns,rules}.py` + `catalog/data/{patterns,lifelike_rules,elementary_rules}.json` (12 patterns, 27 life-like rules, 18 reference ECA rules). Every verified pattern vector (periods, glider/LWSS shifts, R-pentomino 27/178, acorn 57, diehard 2→0, Gosper gun 36/41/46/51/56) passed unchanged; hypothesis checks `step` against a slow loop and all 2¹⁸ codes round-trip. Legacy `GameofLife.py`, `TD CA/`, `TD_Rule_Web Scraper.py` removed and dropped from ruff excludes; `CLAUDE.md` legacy list updated. Judgement call: catalog `description`/`tags` text for rules and patterns is my own wording (not in the phase file); `self-dual` is tagged only on `day_and_night` until P10 computes duality for all. **Next: P04.**

- **P02 done** (same session). `src/cellauto/engines/elementary.py` + `tests/engines/test_elementary.py` (18 tests incl. A051023 rule-30 column, Gould's sequence, B3 edge regression, GF(2) nilpotency trap, hypothesis reference/mirror properties). All verified vectors passed unchanged. `OD_CA.py` `git rm`'d and dropped from ruff `extend-exclude`; `CLAUDE.md` legacy list updated. Test-only note: variable `l` renamed `lt` (ruff E741). **Next: P03.**

- **2026-10-14 (cont.), `claude/keen-turing-np4r23`:** **P01 done** (`pyproject.toml`, `src/cellauto/{__init__,__main__,cli,_types,_validate}.py`, `py.typed`, `tests/test_smoke.py`, `.github/workflows/ci.yml`). Deviation from the phase file: ruff `extend-exclude` has `docs/plan` instead of `docs/plan/reference`, because the installed ruff also lints/formats fenced code in `docs/plan/*.md`. Checks: ruff, ruff format, mypy, pytest all pass; SessionStart hook validated (`CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh`). **Next: P02.** (The owner asked that docs be updated at every step: each phase commit includes its STATUS row + log entry, and `CLAUDE.md` / `docs/` where the repo's shape changed.)

- **2026-10-14, `claude/keen-turing-np4r23`:** merged `main` already contained the plan. **P00 done.** **Surprise:** `main` had a parallel monorepo reorganisation (e68a05f) that the plan didn't know about: it *copied* the legacy code into `packages/{neural_ca,ca_cryptography,classical_ca}` (snake_case names, some small fixes, whitespace) and added `tests/test_*.py`, a `.gitignore` (which ignored `*.pt`, `**/data/*.png`, `**/animations/`) and a monorepo `README.md`, leaving the original dirs in place. It also had already removed the committed `__pycache__`/`.DS_Store`. Owner decisions: (1) delete `packages/` and the legacy `tests/` and follow the plan layout; (2) overwrite `.gitignore` with the P00 version. Done on top of the P00 steps: `.vscode/` untracked, `TD CA/UI.py` and `neural-cellular-automata/.gitignore` removed, crypto archived to `experiments/ca_crypto/` (original, unpatched copy) with README, `Rule Predictor/` and the Distill notebook deleted. **Left for later:** the root `README.md` still describes the deleted `packages/` monorepo (stale until P15 rewrites it); `GameofLife.py`, `OD_CA.py`, `TD CA/`, `TD_Rule_Web Scraper.py` and `neural-cellular-automata/` remain until their phases replace them. Nothing to lint or test yet (no `pyproject.toml` until P01). **Next: P01.**

- **2026-10-07, `claude/zen-clarke-bey6eo` (planning session):** audited the repo; wrote plan v1, then v2 (`docs/plan/`); verified every quoted number with the prototypes in `docs/plan/reference/`; added `CLAUDE.md`, `HANDOFF.md`, this file, and a SessionStart hook (`.claude/hooks/session-start.sh`). **No production code changed yet. Next: P00.**
