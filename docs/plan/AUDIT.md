# Audit of the original repository (verified Oct 2026)

## 1. Inventory

```
OD_CA.py                         1D elementary CA (class, python loops, matplotlib)
GameofLife.py                    standalone pygame Game of Life (hard-coded B3/S23)
TD CA/CA.py                      2D life-like CA (B/S rules as lists of str), python loops
TD CA/game_board.py              pygame UI for TD CA
TD CA/main_run.py                runner (100x70, B3/S23); comments list other rules, incl. B0123478/S01234678 (= the dual of Life, see P10)
TD CA/grapher.py                 runs a 100-gen sim AT IMPORT TIME, matplotlib
TD CA/UI.py                      empty file
TD_Rule_Web Scraper.py           scrapes conwaylife.com rule table at import time
Rule Predictor/CA.py             duplicate of TD CA/CA.py
Rule Predictor/dataset_builder.py, drawing.py, Random Forest.ipynb (3 cells, no model)
CA_Cryptography System/*.py      1D-CA "encryption"; CA.py duplicates OD_CA.py; visualizer_2 ≈ visualiser
neural-cellular-automata/src/    PyTorch NCA (model, helper, train, visualize, vendored medmnist loader)
neural-cellular-automata/models/ 25 checkpoints (.pt, ~35 KB each)
neural-cellular-automata/animations/ 25 GIFs (~15 MB total)
neural-cellular-automata/data/   3 targets: chest/blood/retina.png (28x28 RGBA)
neural-cellular-automata/notebooks/Growing_Neural_Cellular_Automata.ipynb  (Google's Colab, unmodified, TF)
.DS_Store, .vscode/, 10 committed __pycache__/*.pyc files
```

There are no tests, no dependency manifest, no package structure, and no license.

## 2. Confirmed bugs: classical CA code

| # | Where | Bug | Evidence |
|---|-------|-----|----------|
| B1 | `GameofLife.py:21` | Neighbour count via `cells[row-1:row+2, ...]`. At row/col 0 the slice `-1:2` is empty, so edge counts are wrong, even negative. | A cell at (0,1) with 2 live neighbours gets a count of **−1.0** |
| B2 | `GameofLife.py` | Pure-Python double loop + one `pygame.draw.rect` per cell per frame. | – |
| B3 | `OD_CA.py:16`, crypto `CA.py` | `range(1, width-1)`: edge cells are never updated. | inspection |
| B4 | `TD CA/CA.py` | Rules are lists of chars matched with `str(n) in rule`. Fragile; O(H·W·9) Python loop. | – |
| B5 | `Rule Predictor/dataset_builder.py:27` | Passes `"B3/S23"` as both birth and survival, so `"2" in "B3/S23"` is True and the data is **B23/S23**. | verified |
| B6 | same file:44 | `str(np.ndarray)` into CSV truncates arrays >1000 elements with `...`. | verified |
| B7 | same file | `GRID_WIDTH` is defined only under `__main__`, so import raises NameError; unused imports. | – |
| B8 | `CA_Cryptography System/RSRG.py` | `secrets.SystemRandom(seed)` ignores the seed; `randint(0,256)` can return 256; `randint(0,1001)` can return 0 generations, and then `grid[-1]` raises IndexError. | verified |
| B9 | `Crypt_CA_1.py` | Evolves the plaintext forward. Most rules are non-invertible (only 6 of 256 ECA are reversible, see P09), so it cannot be decrypted. | verified |
| B10 | `Attacks.py` | Brute-force loop `break`s after rule 1; `recur_brute_force` is a stub; importing runs `Crypt_CA_1`. | – |
| B11 | several | Work runs at import time (no `__main__` guard). | – |

## 3. Confirmed issues: NCA code

| # | Where | Issue | Fixed in |
|---|-------|-------|----------|
| N1 | `helper.py:84` | `Image.ANTIALIAS` was removed in Pillow 10, so the code crashes. | P05 |
| N2 | `model.py:21` | `self.fire_rate = 0.5` ignores the argument. | P08 |
| N3 | `model.py:50` | `kernel` is not a buffer, so `.to(device)` misses it. Fix: `register_buffer(..., persistent=False)`; `persistent=False` keeps old checkpoints loadable with `strict=True`. | P08 |
| N4 | `model.py:108` | Random mask is created on CPU, then copied, every step. | P08 |
| N5 | `train.py:154`, `visualize.py:21` | `--config` is parsed then ignored; the path is hard-coded. | P08 |
| N6 | `train.py:96` | Unknown loss raises NameError. | P08 |
| N7 | `train.py:135` | Damage mask uses `img_size`, not the padded size. | P08 |
| N8 | `helper.py` | Imports medmnist plus a 315-line vendored copy of it, just to load PNGs. | P05/P08 |
| N9 | config | `device: mps` fails off-Mac. | P08 |
| N10 | train/visualize | `plt.show()` blocks; the GIF writer needs ImageMagick. | P06/P08 |
| N11 | checkpoints | `state_dict` holds only MLP weights; filter/loss live only in the path. | P05 manifest |

## 4. NCA facts established by running a numpy re-implementation

- All 25 checkpoints have keys `update_module.0.weight (128,48,1,1)`, `update_module.0.bias (128,)`, `update_module.2.weight (16,128,1,1)`.
- MSE vs target (RGBA, 200 steps, fire rate 0.5, seed 0):

| family | MSE@200 |
|---|---|
| sobel / scharr × L1 / L2 / Manhattan (3 targets) | 0.0001 – 0.0006 |
| laplacian / mean / gaussian (L2) | 0.0095 – 0.047 |
| hinge (sobel) | 0.035 – 0.052 |

- Isotropic kernels have `dx == dyᵀ == dx`, so the perception carries a duplicated channel and no direction. This is the anisotropy issue studied by Mordvintsev et al. 2022 (IsoNCA).
- Hinge ignores per-pixel errors < 0.5 by construction, so it cannot fit fine detail.
- Regeneration pilot (5 seeds, damage at t=300): `chest-l2-sobel-damage` recovers in **35 ± 8** steps vs **60 ± 17** for `chest-l2-sobel`. P12 makes this rigorous.
- All three target PNGs have alpha = 255 everywhere, so "growth" is filling a square.
- Perception layout: index = `3*c + k`, k ∈ (identity, dx, dy), zero padding. Alive = 3×3 max-pool of channel 3 (out-of-bounds ignored) > 0.1, required both before and after the update.
