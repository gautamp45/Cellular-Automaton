# Cellular-Automaton — Cleanup & Refactor Plan

**Goal:** turn this repo from a set of loose scripts into one tested Python package (`cellauto`). The package becomes the **reference implementation** and the **data/asset exporter** for a frontend that shows 1D (elementary) CA, 2D life-like CA (Game of Life + rule variants), and Neural CA.

**Audience:** this plan is written to be executed phase-by-phase by an AI coding agent (or a human) without further design work. Every phase lists exact file operations, API signatures, acceptance checks, and a commit message. Test vectors in this document were **computed and verified** against reference implementations. If a test using one of them fails, the implementation is wrong, not the vector.

---

## 0. How to execute this plan (read first)

1. Work on the current feature branch. Do **one phase per commit** (Phases 4 and 5 may be two commits each where noted).
2. Before every commit run:
   ```bash
   ruff check . && ruff format --check . && pytest -q
   ```
   All three must pass. (From Phase 1 onward.)
3. Use `git mv` for every file that is **moved** (models, images, GIFs, experiments) so history is preserved. Use `git rm` for deletions.
4. **Do not change NCA math** beyond the fixes explicitly listed in Phase 4. Existing checkpoints must keep loading and producing the same results.
5. Do not add dependencies beyond those listed in Phase 1.
6. Do not start the frontend. Section 13 only defines the contract it will consume.
7. If something in the repo contradicts this plan (e.g. a file has changed since it was written), stop and ask rather than improvising.

### Decision points (defaults already chosen; change them before execution if you disagree)

| # | Decision | Default in this plan | Alternative |
|---|----------|----------------------|-------------|
| D1 | Package name | `cellauto` | anything import-safe |
| D2 | `Rule Predictor/` | **Delete.** Nothing in it works (see §1.3). Record the idea in `docs/ROADMAP.md`. | Move to `experiments/` |
| D3 | `CA_Cryptography System/` | **Move as-is to `experiments/ca_crypto/`** with a README of known issues. Not part of the package. | Delete, or rebuild later as a "Rule 30 keystream" demo |
| D4 | Google's Distill Colab notebook | **Delete, link it in README.** It is unmodified third-party TensorFlow code that doesn't run outside Colab. | Keep under `docs/reference/` |
| D5 | License | **MIT** (check with collaborator *Manan Shah*, who authored the crypto commit) | Apache-2.0 |
| D6 | NCA GIFs (~15 MB) | **Keep**, move to `assets/nca/animations/`. They're the ablation results. | Keep a curated subset |
| D7 | Frontend execution model | **All simulation runs client-side** (§10). Python is not a runtime server. | FastAPI backend (not recommended, see §13) |

---

## 1. Audit — current state (verified)

### 1.1 Inventory

```
OD_CA.py                         1D elementary CA (class, python loops, matplotlib)
GameofLife.py                    standalone pygame Game of Life (hard-coded B3/S23)
TD CA/CA.py                      2D life-like CA (B/S rules as lists of str), python loops
TD CA/game_board.py              pygame UI for TD CA
TD CA/main_run.py                runner (100x70, B3/S23)
TD CA/grapher.py                 runs a 100-gen sim AT IMPORT TIME, matplotlib
TD CA/UI.py                      empty file
TD_Rule_Web Scraper.py           scrapes conwaylife.com rule table at import time
Rule Predictor/CA.py             duplicate of TD CA/CA.py
Rule Predictor/dataset_builder.py, drawing.py, Random Forest.ipynb (3 cells, no model)
CA_Cryptography System/*.py      1D-CA "encryption": CA.py duplicates OD_CA.py; visualizer_2 ≈ visualiser
neural-cellular-automata/src/    PyTorch NCA (model, helper, train, visualize, vendored medmnist loader)
neural-cellular-automata/models/ 25 checkpoints (.pt, ~35 KB each)
neural-cellular-automata/animations/ 25 GIFs (~15 MB total)
neural-cellular-automata/data/   3 targets: chest/blood/retina.png (28x28 RGBA)
neural-cellular-automata/notebooks/Growing_Neural_Cellular_Automata.ipynb  (Google's Colab, unmodified)
.DS_Store, .vscode/, 10 committed __pycache__/*.pyc files
```

There are no tests, no dependency manifest, no package structure, and no license. Directory names contain spaces, and there are three copies of the 1D/2D CA logic.

### 1.2 Confirmed bugs in the classical CA code

| # | Where | Bug | Evidence |
|---|-------|-----|----------|
| B1 | `GameofLife.py:21` | Neighbour count via `cells[row-1:row+2, ...]`. At row/col 0 the slice is `-1:2`, which is **empty**, so edge cells get wrong (even negative) counts. | For a cell at (0,1) with 2 live neighbours the expression returns **-1.0**. |
| B2 | `GameofLife.py` | Pure-Python double loop + one `pygame.draw.rect` per cell per frame. Very slow. | — |
| B3 | `OD_CA.py:16`, crypto `CA.py` | `range(1, width-1)`: edge cells are **never updated** (frozen at their initial value). | Code inspection. |
| B4 | `TD CA/CA.py` | Rules are lists of single-char strings and use `str(n) in rule`. Fragile: a full rulestring passed in silently matches the wrong digits (see B5). O(H·W·9) Python loop. | — |
| B5 | `Rule Predictor/dataset_builder.py:27` | Passes `"B3/S23"` as **both** birth and survival rule. `"2" in "B3/S23"` is True, so the generated "Life" data is actually **B23/S23**. | Verified. |
| B6 | `Rule Predictor/dataset_builder.py:44` | Writes numpy arrays into CSV cells via `str(array)`. Arrays >1000 elements are **truncated with `...`**, so the dataset is unrecoverable. | `'...' in str(np.random.randint(2,size=(60,80)))` returns True. |
| B7 | `Rule Predictor/dataset_builder.py` | `GRID_WIDTH/HEIGHT` are defined only under `__main__`, so importing the module raises NameError. Unused pygame/DrawingBoard imports. | — |
| B8 | `CA_Cryptography System/RSRG.py` | `secrets.SystemRandom(seed)` **ignores the seed** (it uses OS entropy). `randint(0,256)` can return 256, which is not an 8-bit rule. `randint(0,1001)` can return 0 generations, and then `grid[-1]` raises IndexError on an empty array. | Verified. |
| B9 | `CA_Cryptography System/Crypt_CA_1.py` | "Encrypts" by evolving the plaintext forward. Most rules are non-invertible, so this **cannot be decrypted** in general. | — |
| B10 | `CA_Cryptography System/Attacks.py` | Brute-force loop `break`s after rule 1. `recur_brute_force` is a stub. Importing `Crypt_CA_1` runs it (with a plot window). | — |
| B11 | `TD CA/grapher.py`, `TD_Rule_Web Scraper.py`, `Crypt_CA_1.py`, `main.py` | Do work at import time (no `if __name__ == "__main__"`). | — |

### 1.3 Confirmed issues in the NCA code

| # | Where | Issue | Fix in Phase 4 |
|---|-------|-------|----------------|
| N1 | `helper.py:84` | `Image.ANTIALIAS` was **removed in Pillow 10**, so `load_image` crashes on any current install. | Use `Image.Resampling.LANCZOS` |
| N2 | `model.py:21` | `self.fire_rate = 0.5` ignores the constructor argument. | Honour the argument (default stays 0.5) |
| N3 | `model.py:50` | `self.kernel` is a plain attribute, not a buffer, so `model.to(device)` doesn't move it. | `register_buffer("kernel", ..., persistent=False)`. **`persistent=False` is required** so existing checkpoints still load with `strict=True`. |
| N4 | `model.py:108` | Random mask is created on CPU then copied every step. | `torch.rand(..., device=x.device, generator=...)` |
| N5 | `train.py:154`, `visualize.py:21` | `--config` is parsed but ignored; the path is hard-coded relative to repo root. | Honour `--config` |
| N6 | `train.py:96-103` | Unknown `loss` causes NameError on `loss_batch`. | Loss registry dict, ValueError on unknown key |
| N7 | `train.py:135` | Damage mask uses `img_size`, not padded size, so it crashes if `padding > 0`. | Use the seed's spatial shape |
| N8 | `helper.py:7-10` | Imports `medmnist` (and a 315-line **vendored copy** of medmnist's dataset code) just to load PNGs. medmnist was only needed once, to create the 3 committed PNGs. | Drop medmnist entirely, delete `medmnist_loader.py` |
| N9 | `train_config.yaml` | `device: mps` makes it fail on non-Mac machines. | `device: null` (auto) |
| N10 | `train.py`/`visualize.py` | `plt.show()` blocks; the GIF writer needs ImageMagick. | Save loss CSV/PNG; write GIFs with Pillow |
| N11 | checkpoints | `state_dict` contains **only** the MLP weights (`update_module.0.weight (128,48,1,1)`, `update_module.0.bias (128,)`, `update_module.2.weight (16,128,1,1)`). Filter, channels, and loss are encoded **only in the file path**. | Explicit manifest (Phase 4) |

### 1.4 NCA facts established during the audit (important for design)

- **All 25 checkpoints share one architecture:** 16 channels, perception 48 → hidden 128 → 16, ~8.2K params.
- **Inference does not need PyTorch.** A ~60-line numpy forward pass (Appendix A) loads every checkpoint and converges to the targets. Measured MSE vs target (RGBA, 200 steps, fire_rate 0.5):

  | model family | MSE @200 steps |
  |---|---|
  | sobel / scharr with L1, L2, Manhattan (all 3 targets) | **0.0001 – 0.0006** |
  | laplacian / mean / gaussian (L2) | 0.0095 – 0.047 |
  | Hinge (sobel) | 0.035 – 0.052 |

  This is a genuine ablation result worth presenting in the frontend. **Isotropic filters fail** because for symmetric kernels `filter_x == filter_y`, so the cell perceives one redundant channel and gets no directional information. **Hinge loss plateaus** because it ignores per-pixel errors below 0.5.
- **The three target PNGs have alpha = 255 everywhere.** The "living" region is therefore the whole 28×28 square, and "growth" is the CA filling a square, not growing a shape. Damage → regeneration is the more compelling demo with the current models. Retraining on alpha-masked targets is in the Roadmap (Phase 9, optional).
- **Perception layout:** for channel `c` and kernel `k ∈ (identity, dx, dy)`, perception index = `3*c + k`. Zero padding (`padding=1`). The alive mask is a 3×3 max-pool on channel 3 (alpha) with **out-of-bounds ignored** (PyTorch pads max-pool with −inf), threshold `> 0.1`. A cell update is kept only where alive both **before and after** the update. `dy` kernel = transpose of `dx` kernel. Kernels are divided by the scalar (sobel 8, scharr 16, gaussian 16, laplacian 8, mean 9). The frontend must reproduce this exactly.

---

## 2. Target architecture

### 2.1 Principles

1. **Functional core:** every engine is `state in → state out` using pure numpy, with no globals, no UI, and no work at import time. Stateful classes caused the triplication; don't bring them back.
2. **Table-driven rules:** elementary rules are an 8-entry lookup table, and life-like rules are a `(2, 9)` lookup table `lut[state, neighbour_count]`. This is fast in numpy, trivial to unit-test, and maps 1:1 to a TypeScript or WebGL implementation (a shader uniform of 8 or 18 values).
3. **Python is the source of truth; the browser is the runtime.** Python produces verified JSON artefacts (rule catalogs, patterns, NCA weights) plus **golden fixtures**. The frontend's TypeScript port is tested against those fixtures, which makes cross-language parity testable instead of hoped-for.
4. **Torch is optional:** `pip install cellauto` gives numpy + Pillow only, and runs every automaton including NCA inference. `pip install cellauto[train]` adds torch for training.
5. **Deterministic outputs:** exports have sorted keys and no timestamps, so re-running the export produces no diff and CI can enforce that committed artefacts are up to date.

### 2.2 Target tree (end state)

```
.
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── .github/workflows/ci.yml
├── src/cellauto/
│   ├── __init__.py
│   ├── __main__.py              # python -m cellauto -> cli.main()
│   ├── py.typed
│   ├── elementary.py            # 1D Wolfram rules
│   ├── lifelike.py              # 2D outer-totalistic B/S rules
│   ├── patterns.py              # RLE parse/encode, placement
│   ├── catalog.py               # loads packaged JSON catalogs
│   ├── render.py                # numpy -> PIL PNG/GIF
│   ├── export.py                # web asset + fixture export
│   ├── cli.py
│   ├── data/
│   │   ├── elementary_rules.json
│   │   ├── lifelike_rules.json
│   │   └── patterns.json
│   └── nca/
│       ├── __init__.py
│       ├── spec.py              # filter kernels, constants, NCAConfig dataclass
│       ├── numpy_model.py       # torch-free inference (Appendix A)
│       ├── checkpoints.py       # manifest + .npz loading
│       ├── image.py             # load_target, rgba_to_rgb, make_seed, circle_mask (numpy)
│       └── training/            # requires [train] extra
│           ├── __init__.py
│           ├── model.py         # torch NCA (fixed per §1.3)
│           ├── losses.py
│           └── train.py
├── configs/nca/chest-l2-sobel-damage.yaml
├── models/nca/
│   ├── manifest.json
│   ├── <id>.npz                 # canonical, torch-free weights
│   └── <id>.pt                  # original torch checkpoints (for resuming training)
├── assets/nca/
│   ├── targets/{chest,blood,retina}.png
│   └── animations/<id>.gif
├── scripts/convert_checkpoints.py
├── tests/
│   ├── test_elementary.py
│   ├── test_lifelike.py
│   ├── test_patterns.py
│   ├── test_catalog.py
│   ├── test_render.py
│   ├── test_cli.py
│   ├── test_export.py
│   ├── test_nca_numpy.py
│   └── test_nca_torch_parity.py # skipped if torch missing
├── web/public/data/             # generated by `cellauto export-web` (committed)
├── experiments/ca_crypto/       # archived, excluded from lint/tests/package
└── docs/
    ├── REFACTOR_PLAN.md         # this file
    ├── ARCHITECTURE.md
    ├── DATA_CONTRACT.md         # JSON schemas the frontend consumes
    └── ROADMAP.md
```

**Model IDs:** `{target}-{loss}-{filter}[-damage]`, all lowercase, where loss ∈ `l1|l2|manhattan|hinge`. Example: `neural-cellular-automata/models/chest/L2/chest_sobel_damage.pt` becomes `chest-l2-sobel-damage`. There are 25 IDs.

---

## 3. Phase 0 — Repo hygiene

**Goal:** remove junk and make ignore rules correct. No code changes.

1. `git rm --cached -r` and delete: `.DS_Store`, `.vscode/`, every `__pycache__/` directory, `TD CA/UI.py` (empty).
2. Delete `neural-cellular-automata/.gitignore` (it has the typo `__pychache__` and moves to root).
3. Create root `.gitignore`:
   ```gitignore
   # Python
   __pycache__/
   *.py[cod]
   *.egg-info/
   .venv/
   venv/
   build/
   dist/
   .pytest_cache/
   .ruff_cache/
   # Data / logs
   logs/
   *.npz.tmp
   # OS / editors
   .DS_Store
   .vscode/
   .idea/
   # Node (future frontend)
   node_modules/
   web/dist/
   ```
4. Replace `.gitattributes` with:
   ```gitattributes
   * text=auto
   *.pt binary
   *.npz binary
   *.gif binary
   *.png binary
   ```

**Accept:** `git ls-files | grep -E "pycache|DS_Store|vscode"` prints nothing.
**Commit:** `chore: remove committed caches/OS files, add root gitignore`

---

## 4. Phase 1 — Package scaffold, tooling, CI

1. Create `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "cellauto"
   version = "0.1.0"
   description = "Elementary, life-like and neural cellular automata: reference implementations and web asset exporter."
   readme = "README.md"
   requires-python = ">=3.10"
   license = { text = "MIT" }
   dependencies = ["numpy>=1.24", "pillow>=10.0"]

   [project.optional-dependencies]
   train = ["torch>=2.1", "pyyaml>=6.0", "tqdm>=4.60"]
   dev = ["pytest>=8.0", "ruff>=0.6"]

   [project.scripts]
   cellauto = "cellauto.cli:main"

   [tool.hatch.build.targets.wheel]
   packages = ["src/cellauto"]

   [tool.ruff]
   line-length = 100
   target-version = "py310"
   extend-exclude = ["experiments", "web"]

   [tool.ruff.lint]
   select = ["E", "F", "I", "UP", "B", "NPY", "SIM"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   markers = ["torch: requires PyTorch", "slow: takes more than ~5 seconds"]
   ```
2. Create `src/cellauto/__init__.py` with `__version__ = "0.1.0"` and nothing else for now. Create an empty `src/cellauto/py.typed`. Create `src/cellauto/__main__.py`:
   ```python
   from cellauto.cli import main

   raise SystemExit(main())
   ```
   Create a temporary `src/cellauto/cli.py` with `def main(argv=None) -> int: return 0`.
3. Create `tests/test_smoke.py` asserting `import cellauto; cellauto.__version__ == "0.1.0"`.
4. Create `.github/workflows/ci.yml`:
   ```yaml
   name: ci
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix: { python-version: ["3.10", "3.12"] }
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "${{ matrix.python-version }}" }
         - run: pip install -e ".[dev]"
         - run: ruff check .
         - run: ruff format --check .
         - run: pytest -q
     torch-parity:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.12" }
         - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
         - run: pip install -e ".[dev,train]"
         - run: pytest -q -m torch
   ```
   (Phase 6 adds an "exports up to date" step.)

**Accept:** `pip install -e ".[dev]"` succeeds, then `ruff check . && ruff format --check . && pytest -q` passes. Ruff must not lint the legacy scripts: add the legacy top-level dirs/files to `extend-exclude` temporarily (`"OD_CA.py", "GameofLife.py", "TD CA", "Rule Predictor", "CA_Cryptography System", "TD_Rule_Web Scraper.py", "neural-cellular-automata"`) and remove each entry in the phase that deletes it.
**Commit:** `build: add pyproject, ruff, pytest and CI scaffold`

---

## 5. Phase 2 — Elementary (1D) engine

**File:** `src/cellauto/elementary.py`. Replaces `OD_CA.py` and `CA_Cryptography System/CA.py` (the latter stays in experiments untouched).

### API

```python
from typing import Literal
import numpy as np

Boundary = Literal["wrap", "dead"]

def rule_table(rule: int) -> np.ndarray:
    """uint8 array of shape (8,). Index = (left << 2) | (centre << 1) | right.
    Raises ValueError unless 0 <= rule <= 255 and type is int (reject bool)."""
    # implementation: ((rule >> np.arange(8)) & 1).astype(np.uint8)

def step(row: np.ndarray, rule: int, boundary: Boundary = "wrap") -> np.ndarray:
    """One generation. row: 1D array of 0/1. Returns new uint8 array, same shape.
    wrap: periodic (np.roll). dead: out-of-range neighbours are 0.
    EVERY cell is updated, including the edges (fixes B3)."""

def run(initial: np.ndarray, rule: int, generations: int, boundary: Boundary = "wrap") -> np.ndarray:
    """Space-time diagram, uint8 shape (generations + 1, width); row 0 == initial.
    generations=0 is valid. Negative raises ValueError. Precompute rule_table once."""

def single_cell(width: int) -> np.ndarray:   # zeros, 1 at width // 2
def random_row(width: int, density: float = 0.5, seed: int | None = None) -> np.ndarray  # np.random.default_rng(seed)
def from_bits(bits: str) -> np.ndarray       # "0110" -> array; ValueError on other chars
def to_bits(row: np.ndarray) -> str
```

Validate inputs: `row` must be 1D and contain only 0/1, otherwise ValueError. `width >= 1`.

### Tests (`tests/test_elementary.py`), all verified values

- `rule_table(30).tolist() == [0,1,1,1,1,0,0,0]`
- `rule_table(110).tolist() == [0,1,1,1,0,1,1,0]`
- `rule_table(90).tolist() == [0,1,0,1,1,0,1,0]`
- `rule_table(256)`, `rule_table(-1)`, `rule_table(True)` raise ValueError.
- Rule 30, `run(single_cell(201), 30, 40)`: the centre column `[:25, 100]` equals
  `[1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1,0,0,1,1,1]` (OEIS A051023).
- Rule 30 row 1, cells `97:104` == `[0,0,1,1,1,0,0]`. Row 2, cells `96:105` == `[0,0,1,1,0,0,1,0,0]`.
- Rule 110, `run(single_cell(11), 110, 1)[1].tolist() == [0,0,0,0,1,1,0,0,0,0,0]`
- Rule 90 Sierpiński property: `h = run(single_cell(129), 90, 32)`, and for all `t` in `0..31`, `h[t].sum() == 2 ** bin(t).count("1")`.
- Edge update (regression for B3): width 5, row `[1,0,0,0,0]`, rule 90 (XOR of neighbours):
  - `wrap` gives `[0,1,0,0,1]`
  - `dead` gives `[0,1,0,0,0]`
- `run(x, r, 0).shape == (1, len(x))`. Negative generations raise ValueError.
- `to_bits(from_bits("0110")) == "0110"`.
- `random_row(100, 0.5, seed=1)` is deterministic (two calls are equal).

**Then:** `git rm OD_CA.py`, and remove it from ruff `extend-exclude`.
**Commit:** `feat(elementary): vectorised 1D Wolfram CA with wrap/dead boundaries`

---

## 6. Phase 3 — Life-like (2D) engine, patterns, catalogs

### 3a. `src/cellauto/lifelike.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LifeRule:
    birth: frozenset[int]
    survival: frozenset[int]

    @classmethod
    def parse(cls, text: str) -> "LifeRule":
        """Accepts B/S notation, case-insensitive, optional '/', either order:
        'B3/S23', 'b3s23', 'S23/B3', 'B2/S' (empty survival), 'B/S' (empty both).
        Digits must be 0-8; duplicates allowed but collapsed. Anything else -> ValueError
        (including legacy '23/3' notation: reject with a helpful message)."""

    @property
    def rulestring(self) -> str:   # canonical 'B3/S23' (sorted digits)

    def table(self) -> np.ndarray:
        """uint8 (2, 9): lut[0, n] = 1 if n in birth; lut[1, n] = 1 if n in survival."""

def neighbour_counts(grid: np.ndarray, boundary: Boundary = "wrap") -> np.ndarray:
    """Moore-neighbourhood live counts, uint8, same shape.
    wrap: sum of the 8 np.roll shifts. dead: np.pad(grid, 1) then sum the 8 shifted slices.
    Do NOT use scipy (no new dependency)."""

def step(grid: np.ndarray, rule: LifeRule | str, boundary: Boundary = "wrap") -> np.ndarray:
    """new = rule.table()[grid, neighbour_counts(grid, boundary)] (uint8)."""

def iterate(grid, rule, boundary="wrap") -> Iterator[np.ndarray]:  # yields successive generations forever
def run(grid, rule, steps: int, boundary="wrap") -> np.ndarray:    # (steps + 1, H, W) uint8, [0] == grid
def random_grid(height: int, width: int, density: float = 0.5, seed: int | None = None) -> np.ndarray
```

`Boundary` is imported from `elementary` (or move it to a small `cellauto/_types.py`). Validate that `grid` is 2D with values 0/1.

**Why the LUT-index approach:** `lut[grid, counts]` is a single fancy-indexing op, so it is fully vectorised. It also replaces string matching (B4/B5) with integers, and it is exactly what a GPU shader will do (a texture lookup into 18 values).

### 3b. `src/cellauto/patterns.py`

```python
def parse_rle(text: str) -> np.ndarray:
    """Standard Life RLE. Ignore lines starting with '#'. Ignore an optional header line
    'x = <w>, y = <h>[, rule = ...]' (if present, pad the result to w x h).
    Body tokens: optional run count + one of 'b' (dead), 'o' (alive), '$' (end row; count = rows),
    terminated by '!'. Whitespace/newlines inside the body are ignored.
    Pad rows on the right with 0 to the max width. Unknown tokens -> ValueError."""

def to_rle(grid: np.ndarray) -> str:   # inverse, no header, ends with '!'; parse_rle(to_rle(g)) == trimmed g
def place(pattern: np.ndarray, shape: tuple[int, int], at: tuple[int, int] | None = None) -> np.ndarray:
    """Zero grid of `shape` with pattern's top-left at `at` (default: centred via (H-h)//2, (W-w)//2).
    ValueError if it doesn't fit."""
def get(pattern_id: str) -> np.ndarray   # parse_rle of the catalog entry
```

### 3c. Catalogs (`src/cellauto/data/*.json`) and `src/cellauto/catalog.py`

`catalog.py` loads them with `importlib.resources.files("cellauto.data")` and exposes:
`elementary_rules() -> list[dict]`, `lifelike_rules() -> list[dict]`, `patterns() -> list[dict]`, `lifelike_rule(id) -> LifeRule`.

**`patterns.json`.** Use these exact RLE strings; their behaviour below was verified:

```json
[
  {"id": "block", "name": "Block", "category": "still_life", "rle": "2o$2o!"},
  {"id": "beehive", "name": "Beehive", "category": "still_life", "rle": "b2o$o2bo$b2o!"},
  {"id": "blinker", "name": "Blinker", "category": "oscillator", "period": 2, "rle": "3o!"},
  {"id": "toad", "name": "Toad", "category": "oscillator", "period": 2, "rle": "b3o$3o!"},
  {"id": "beacon", "name": "Beacon", "category": "oscillator", "period": 2, "rle": "2o$2o$2b2o$2b2o!"},
  {"id": "pulsar", "name": "Pulsar", "category": "oscillator", "period": 3, "rle": "2b3o3b3o2b2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2b2$2b3o3b3o2b$o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!"},
  {"id": "glider", "name": "Glider", "category": "spaceship", "period": 4, "rle": "bo$2bo$3o!"},
  {"id": "lwss", "name": "Lightweight spaceship", "category": "spaceship", "period": 4, "rle": "bo2bo$o4b$o3bo$4o!"},
  {"id": "r_pentomino", "name": "R-pentomino", "category": "methuselah", "rle": "b2o$2o$bo!"},
  {"id": "acorn", "name": "Acorn", "category": "methuselah", "rle": "bo5b$3bo3b$2o2b3o!"},
  {"id": "diehard", "name": "Diehard", "category": "methuselah", "rle": "6bob$2o6b$bo3b3o!"},
  {"id": "gosper_glider_gun", "name": "Gosper glider gun", "category": "gun", "period": 30, "rle": "24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!"}
]
```
Add a one-sentence `description` to each entry (free text).

**`lifelike_rules.json`.** Each entry has `id`, `name`, `rulestring`, and `description` (one sentence on its character). Rulestrings:

| id | name | rulestring |
|---|---|---|
| life | Conway's Life | B3/S23 |
| highlife | HighLife | B36/S23 |
| day_and_night | Day & Night | B3678/S34678 |
| seeds | Seeds | B2/S |
| life_without_death | Life without Death | B3/S012345678 |
| replicator | Replicator | B1357/S1357 |
| maze | Maze | B3/S12345 |
| mazectric | Mazectric | B3/S1234 |
| coral | Coral | B3/S45678 |
| two_by_two | 2x2 | B36/S125 |
| life_34 | 34 Life | B34/S34 |
| diamoeba | Diamoeba | B35678/S5678 |
| morley | Morley (Move) | B368/S245 |
| anneal | Anneal | B4678/S35678 |
| gnarl | Gnarl | B1/S1 |
| long_life | Long Life | B345/S5 |
| amoeba | Amoeba | B357/S1358 |
| assimilation | Assimilation | B345/S4567 |
| coagulations | Coagulations | B378/S235678 |
| walled_cities | Walled Cities | B45678/S2345 |
| stains | Stains | B3678/S235678 |
| dotlife | DotLife | B3/S023 |
| serviettes | Serviettes | B234/S |
| pseudo_life | Pseudo Life | B357/S238 |
| flock | Flock | B3/S12 |
| live_free_or_die | Live Free or Die | B2/S0 |

**`elementary_rules.json`.** Each entry has `rule`, `name` (optional), `wolfram_class` (1–4), and `description`:

| rule | class | note |
|---|---|---|
| 0 | 1 | everything dies |
| 4 | 2 | stable, single cell persists |
| 108 | 2 | periodic structures |
| 184 | 2 | traffic-flow model / particle conservation |
| 30 | 3 | chaotic; used as a PRNG (Mathematica) |
| 45 | 3 | chaotic |
| 90 | 3 | additive (XOR); Sierpiński triangle |
| 150 | 3 | additive (3-XOR) |
| 54 | 4 | complex localized structures |
| 110 | 4 | complex; proven Turing-complete (Cook 2004) |

The scraper `TD_Rule_Web Scraper.py` is replaced by this static file. Runtime scraping is fragile, and the catalog is static knowledge.

### 3d. Tests, all verified

`tests/test_lifelike.py`
- `LifeRule.parse("b3s23").rulestring == "B3/S23"`. `parse("S23/B3")` equals `parse("B3/S23")`. `parse("B2/S").survival == frozenset()`. `parse("B9/S23")`, `parse("23/3")`, and `parse("hello")` raise ValueError.
- `parse("B3/S23").table().tolist() == [[0,0,0,1,0,0,0,0,0],[0,0,1,1,0,0,0,0,0]]`
- **Regression B1:** a 5×5 grid with cells (0,0),(0,1),(1,0) alive, `boundary="dead"`: `neighbour_counts(...)[0,1] == 2`. With `wrap`, a single live cell at (0,0) of a 5×5 grid gives a count of 1 at (4,4).
- Blinker on a 5×5 **wrap** grid returns to itself after 2 steps.
- Every catalog rulestring parses, and `parse(x).rulestring == x`.

`tests/test_patterns.py`. Use Life (`B3/S23`), `boundary="dead"`, pattern centred with `place` in an 80×80 grid (diehard: 120×120):

| pattern | shape (h,w) | pop₀ | expected |
|---|---|---|---|
| block, beehive | (2,2), (3,4) | 4, 6 | step(g) == g |
| blinker / toad / beacon | (1,3) / (2,4) / (4,4) | 3 / 6 / 8 | g₂ == g₀ |
| pulsar | (13,13) | 48 | g₃ == g₀ |
| glider | (3,3) | 5 | g₄ == g₀ shifted by **(+1 row, +1 col)** |
| lwss | (4,5) | 9 | g₄ == g₀ shifted by **(0 rows, −2 cols)** |
| r_pentomino | (3,3) | 5 | pop@30 == 27, pop@130 == 178 |
| acorn | (3,7) | 7 | pop@30 == 57 |
| diehard (120×120) | (3,8) | 7 | pop@129 == 2, pop@130 == 0 |
| gosper_glider_gun | (9,36) | 36 | placed at top-left (2,2) of a 120×120 grid: pop at t = 0, 30, 60, 90, 120 == 36, 41, 46, 51, 56 |

Plus: `parse_rle(to_rle(g))` round-trips for every pattern. RLE with a header and `#C` comments parses. `"3x!"` raises ValueError.

**Then:** `git rm -r GameofLife.py "TD CA" "TD_Rule_Web Scraper.py"` and remove them from ruff `extend-exclude`.
**Commit (2 commits OK):** `feat(lifelike): vectorised B/S engine with rule parser` and `feat(patterns): RLE parser, pattern + rule catalogs`

---

## 7. Phase 4 — Neural CA

Split into **4a (inference, torch-free)** and **4b (training, torch)**. Commit each separately.

### 4a. Move assets, convert checkpoints, numpy inference

1. Move files with `git mv`:
   - `neural-cellular-automata/data/{chest,blood,retina}.png` → `assets/nca/targets/`
   - each `neural-cellular-automata/models/<target>/<Loss>/<target>_<filter>[_damage].pt` → `models/nca/<id>.pt`
   - each `neural-cellular-automata/animations/<target>/<Loss>/<target>_<filter>[_damage].gif` → `assets/nca/animations/<id>.gif`
   - ID rule: `f"{target}-{loss.lower()}-{filter}" + ("-damage" if damage else "")`. Write a throwaway shell or Python loop for this; don't commit it.
2. `scripts/convert_checkpoints.py`: for each `models/nca/*.pt`, write `models/nca/<id>.npz` with keys `w1` (128,48) float32, `b1` (128,) float32, `w2` (16,128) float32. That is, squeeze the 1×1 conv dims: `w1 = state["update_module.0.weight"][:, :, 0, 0]`. Use the **torch-free reader in Appendix B**, which was verified on all 25 files, so conversion works without installing torch. If torch is importable, also assert equality with `torch.load(p, map_location="cpu", weights_only=True)`.
3. `models/nca/manifest.json` is a list sorted by `id`, one entry per model:
   ```json
   {
     "id": "chest-l2-sobel-damage",
     "target": "chest",
     "target_image": "assets/nca/targets/chest.png",
     "loss": "l2",
     "filter": "sobel",
     "trained_with_damage": true,
     "n_channels": 16,
     "hidden_channels": 128,
     "fire_rate": 0.5,
     "grid_size": 28,
     "padding": 0,
     "weights": "models/nca/chest-l2-sobel-damage.npz",
     "torch_checkpoint": "models/nca/chest-l2-sobel-damage.pt",
     "animation": "assets/nca/animations/chest-l2-sobel-damage.gif"
   }
   ```
   Generate it with the convert script (add a `--manifest` flag), then commit the JSON.
4. `src/cellauto/nca/spec.py`:
   ```python
   FILTERS: dict[str, tuple[list[list[int]], float]] = {
       "sobel":     ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 8.0),
       "scharr":    ([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 16.0),
       "gaussian":  ([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 16.0),
       "laplacian": ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 8.0),
       "mean":      ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 9.0),
   }
   ALPHA_CHANNEL = 3
   ALIVE_THRESHOLD = 0.1
   def perception_kernels(filter_name: str) -> np.ndarray:  # (3, 3, 3) float32: [identity, dx, dy]; dy = dx.T; both / scalar

   @dataclass(frozen=True)
   class NCAConfig:
       n_channels: int = 16
       hidden_channels: int = 128
       filter: str = "sobel"
       fire_rate: float = 0.5
   ```
5. `src/cellauto/nca/image.py` (numpy, channel-first `(C, H, W)` float32):
   `load_target(path, size) -> (4, size, size)` (RGBA in [0,1], RGB premultiplied by alpha, resample with `Image.Resampling.LANCZOS`, which fixes N1); `rgba_to_rgb(x) -> (3,H,W)` = `clip(1 - clip(alpha,0,1) + rgb, 0, 1)`; `make_seed(size, n_channels=16, padding=0) -> (C, S, S)` with channels `3:` set to 1 at the centre; `circle_mask(size, rng) -> (S, S)` float32 (same distribution as `helper.make_circle_masks`: centre ~ U(-0.5,0.5)², radius ~ U(0.1,0.4) on a [-1,1] grid); `pad(x, p)`.
6. `src/cellauto/nca/numpy_model.py`: implement Appendix A as a class:
   ```python
   class NumpyNCA:
       def __init__(self, w1, b1, w2, config: NCAConfig): ...
       @classmethod
       def from_npz(cls, path, config: NCAConfig) -> "NumpyNCA": ...
       def step(self, x: np.ndarray, rng: np.random.Generator | None = None,
                fire_rate: float | None = None) -> np.ndarray: ...   # x: (C,H,W) float32
       def run(self, x, steps, rng=None, damage_at: Iterable[int] = ()) -> Iterator[np.ndarray]: ...
   ```
   `fire_rate=1.0` must be deterministic and must not consume RNG.
7. `src/cellauto/nca/checkpoints.py`: `load_manifest(repo_root: Path | None = None) -> list[dict]`, `load_model(model_id, repo_root=None) -> NumpyNCA`. Resolve the repo root by walking up from the CWD until `models/nca/manifest.json` is found; otherwise raise FileNotFoundError with a clear message. Models are repo data and are **not** shipped in the wheel.

**Tests `tests/test_nca_numpy.py`** (no torch):
- The manifest has 25 entries. Every `weights` file loads with the expected shapes.
- `perception_kernels("sobel")[1]` equals the sobel kernel / 8, and `[2]` is its transpose.
- With `fire_rate=1.0`, two runs from the seed are bit-identical.
- After 1 step from the seed, `x[ALPHA_CHANNEL]` is nonzero only inside the 3×3 neighbourhood of the centre.
- **Convergence regression** (mark `slow`): for every model whose filter ∈ {sobel, scharr} and loss ≠ hinge (13 models), run 200 steps from the seed with `rng=np.random.default_rng(0)`, fire_rate 0.5. Then `mean((x[:4] - target)**2) < 0.005`. Measured worst case is 0.0006, so the threshold has margin.

### 4b. Torch training package

Create `src/cellauto/nca/training/{model.py,losses.py,train.py}` by porting `neural-cellular-automata/src/{model,helper,train}.py` with **only** these changes:
- N2: honour `fire_rate`.
- N3: `self.register_buffer("kernel", k, persistent=False)`, built from `spec.perception_kernels` (so filter definitions exist in exactly one place).
- N4: `torch.rand(x[:, :1].shape, device=x.device, generator=self.generator)` with an optional `generator` constructor arg.
- N6: `LOSSES = {"l1": l1, "l2": l2, "manhattan": manhattan, "hinge": hinge}`. Accept keys case-insensitively and raise ValueError on unknown keys. Keep each loss formula **identical**.
- N7: the damage mask uses the padded spatial size of the seed.
- N8: no medmnist import. Targets load via `cellauto.nca.image.load_target`, then `torch.from_numpy`.
- N9/N10: `device: null` means auto (cuda → mps → cpu). No `plt.show()`. Write `<out>.losses.csv`. At the end, save **both** `<id>.pt` (state_dict) and `<id>.npz` (same keys as 4a).
- N5: `train(config: dict)` plus CLI entry `cellauto nca train --config PATH`. Paths in the config are relative to the current working directory (document "run from repo root").
- `TorchNCA.load(path_pt_or_npz, config)` classmethod.
- Keep the training loop logic (pool sampling, worst-sample reseed, damage of best 3) unchanged.

Create `configs/nca/chest-l2-sobel-damage.yaml` from the old `train_config.yaml` with new paths, `device: null`, `id: chest-l2-sobel-damage`.

**Tests `tests/test_nca_torch_parity.py`**: put `pytest.importorskip("torch")` at the top and mark it `torch`.
- `TorchNCA` loads every `.pt` with `strict=True`. This proves `persistent=False` kept compatibility.
- Parity: for `chest-l2-sobel`, from the seed with fire_rate 1.0, 10 steps in torch and numpy agree with `atol=1e-4`.
- A tiny train smoke test: `iterations=2, pool_size=8, batch_size=2` on chest runs and writes `.pt` + `.npz` to `tmp_path`.

Note: download.pytorch.org may be blocked in some sandboxes. If torch can't be installed locally, these tests skip, and CI's `torch-parity` job covers them.

**Then:** `git rm -r neural-cellular-automata` (src, notebook, config files; models/assets are already moved). Remove it from ruff `extend-exclude`.
**Commits:** `feat(nca): torch-free inference, checkpoint manifest, assets reorganised` and `refactor(nca): port torch training into cellauto.nca.training with fixes`

---

## 8. Phase 5 — Rendering + CLI

### `src/cellauto/render.py` (Pillow only, no matplotlib, no pygame)

```python
def grid_to_image(grid: np.ndarray, scale: int = 4, on=(240, 240, 240), off=(12, 12, 16)) -> Image.Image
def rgb_to_image(rgb: np.ndarray, scale: int = 4) -> Image.Image        # (3,H,W) or (H,W,3) float [0,1]
def save_png(img: Image.Image, path) -> None
def save_gif(frames: Iterable[Image.Image], path, fps: int = 20) -> None
    # frames[0].save(path, save_all=True, append_images=rest, duration=int(1000/fps), loop=0)
```
Scaling uses `Image.Resampling.NEAREST`.

### `src/cellauto/cli.py` (argparse, subcommands)

```
cellauto elementary --rule 30 [--width 201] [--steps 100] [--init center|random] [--density 0.5]
                    [--seed N] [--boundary wrap|dead] [--scale 4] --out rule30.png
cellauto life [--rule B3/S23 | --preset highlife] [--pattern glider | --random 0.3] [--seed N]
              [--size 64x64] [--steps 200] [--boundary wrap|dead] [--fps 20] [--scale 6] --out life.gif
cellauto nca run --model chest-l2-sobel [--steps 200] [--fire-rate 0.5] [--seed 0]
                 [--damage-at 100 ...] [--scale 8] --out chest.gif
cellauto nca train --config configs/nca/chest-l2-sobel-damage.yaml      # needs [train]
cellauto list {elementary|rules|patterns|models}
cellauto export-web [--out web/public/data]
```
`main(argv: list[str] | None = None) -> int`. Errors go to stderr with exit code 2. `nca train` without torch prints `install with: pip install -e ".[train]"` and returns 2.

**Tests `tests/test_cli.py`**: call `main([...])` with `tmp_path` outputs. Check the files exist and the PNG size is `(width*scale, (steps+1)*scale)`. `list patterns` output contains `glider`. An invalid rule returns 2.

**Commit:** `feat: pillow renderer and cellauto CLI`

---

## 9. Phase 6 — Web export + golden fixtures (the frontend contract)

`src/cellauto/export.py` → `export_web(out_dir: Path, repo_root: Path) -> list[Path]`. Write JSON with `json.dumps(obj, sort_keys=True, separators=(",", ":"))` and **no timestamps**, so output is deterministic.

```
web/public/data/
├── index.json                      # {"format_version": 1, "files": {...relative paths...}}
├── elementary_rules.json           # copy of catalog
├── lifelike_rules.json             # catalog + "table": [[...9],[...9]] per rule
├── patterns.json                   # catalog + "width","height","cells": [[r,c],...] (live cells)
├── nca/
│   ├── models.json                 # manifest entries (paths rewritten relative to web data dir), + "mse_200" metric
│   ├── <id>.json                   # weights, see below
│   ├── targets/<name>.png          # copied
│   └── animations/<id>.gif         # copied
└── fixtures/
    ├── elementary.json
    ├── lifelike.json
    └── nca_chest-l2-sobel.json
```

**NCA weight file `<id>.json`**:
```json
{
  "format_version": 1,
  "id": "chest-l2-sobel",
  "n_channels": 16, "hidden_channels": 128,
  "fire_rate": 0.5, "alive_threshold": 0.1, "alpha_channel": 3,
  "grid_size": 28, "padding": 0,
  "filter": "sobel",
  "kernels": [[[0,0,0],[0,1,0],[0,0,0]], [[-0.125,0,0.125],...], [[...dy...]]],
  "perception_layout": "index = 3*channel + kernel, kernel order [identity, dx, dy]; zero padding; cross-correlation",
  "w1": {"shape": [128, 48], "dtype": "float32-le", "b64": "..."},
  "b1": {"shape": [128],     "dtype": "float32-le", "b64": "..."},
  "w2": {"shape": [16, 128], "dtype": "float32-le", "b64": "..."}
}
```
Row-major, `base64.b64encode(arr.astype("<f4").tobytes())`. Kernels are exported **already scaled**, so the frontend never re-implements filter definitions. (~45 KB per model, ~1.1 MB total.)

**Fixtures** (for TypeScript parity tests):
- `elementary.json`: for rules {30, 90, 110, 184} × boundary {wrap, dead}: `{rule, boundary, width: 31, generations: 15, initial: "<bits>", rows: ["<bits>", ...16 rows]}`. Include one `random_row(31, seed=7)` initial per rule.
- `lifelike.json`: cases `{rule, boundary, height, width, initial: [bits rows], steps, expected: [bits rows]}`. Cases: glider 16×16 wrap, 8 steps. `random_grid(24,24,0.35,seed=0)` under B3/S23, B36/S23, B2/S, and B3678/S34678, wrap, 10 steps. The same Life case with `dead`.
- `nca_chest-l2-sobel.json`: seed state, fire_rate 1.0, and full `(16,28,28)` states at steps `[1, 2, 5, 10]` as float32-le base64, plus `"atol": 1e-4`. Keep it short: float32 drift compounds over many steps, and the alive threshold makes it discontinuous.

`docs/DATA_CONTRACT.md` documents every file above (field, type, meaning). The frontend depends on this document.

**Tests `tests/test_export.py`**: export into `tmp_path` twice and check the outputs are byte-identical. Decode `<id>.json` weights and check they equal the `.npz`. Re-run the elementary/lifelike fixtures through the Python engines and check they match.

**CI:** add a step to the `test` job:
```yaml
- run: cellauto export-web --out /tmp/webdata && diff -r /tmp/webdata web/public/data
```
Then run `cellauto export-web` and commit `web/public/data/`.

**Commit:** `feat(export): deterministic web data export, NCA weights JSON, golden fixtures`

---

## 10. Phase 7 — Experiments, removals

1. `git mv "CA_Cryptography System" experiments/ca_crypto`. Leave the code untouched (pycache already gone). Add `experiments/ca_crypto/README.md`: what it is (1D CA as a keystream generator; `main.py` is the XOR-keystream variant, `Crypt_CA_1.py` the forward-evolution variant, `Attacks.py` a preimage brute-force), how to run it (`cd experiments/ca_crypto && python main.py`, needs matplotlib/opencv), credit to Manan Shah for the attack/visualiser commit, and **Known issues** B8, B9, B10 from §1.2. Add one honest sentence: security reduces entirely to the random initial state (a one-time pad with extra steps), and the rule/generation "key" adds ≤ ~18 bits.
2. `git rm -r "Rule Predictor"` (D2).
3. Delete the remaining `extend-exclude` legacy entries in `pyproject.toml` (keep `experiments`, `web`).

**Accept:** at repo root, `ls` shows only: `.github assets configs docs experiments models scripts src tests web` plus `README.md LICENSE pyproject.toml .gitignore .gitattributes`.
**Commit:** `chore: archive crypto experiment, remove broken rule predictor`

---

## 11. Phase 8 — Documentation

1. **`README.md`** (rewrite):
   - One-paragraph pitch, plus 3 images generated with the CLI (rule 30 PNG, Gosper gun GIF, one NCA GIF) saved under `docs/media/`.
   - Quickstart: `pip install -e .`, three CLI examples, plus `pip install -e ".[train]"` for training.
   - "What's inside": elementary / life-like / NCA, each with 2–3 lines.
   - **NCA ablation table** from §1.4 (filters × losses, MSE@200) and the explanation of why isotropic filters and hinge loss underperform. This is the most interesting finding in the repo; lead with it.
   - Architecture (link `docs/ARCHITECTURE.md`) and data contract (link `docs/DATA_CONTRACT.md`).
   - **Credits:** Mordvintsev et al., *Growing Neural Cellular Automata*, Distill 2020 (link the original Colab instead of vendoring it, D4); MedMNIST (Yang et al.) as the source of the 3 target images; Wolfram, *A New Kind of Science*; LifeWiki for rules and patterns. **If any of `neural-cellular-automata/src` was adapted from another public repository, credit that repository explicitly**; hiring reviewers check this.
2. **`docs/ARCHITECTURE.md`:** the §2.1 principles, a module diagram, the "Python is reference + exporter, browser is runtime" rationale, and the parity-fixture strategy.
3. **`docs/ROADMAP.md`:** Phase 9 items below, plus a "Frontend" pointer to §12.
4. **`LICENSE`:** MIT (D5) with the copyright holder(s).

**Commit:** `docs: README, architecture, data contract, roadmap, license`

---

## 12. Phase 9 — Optional follow-ups (not part of the cleanup; list in ROADMAP)

- **Retrain NCA on alpha-masked targets** (transparent background, padding 8–16) so the "growth" demo actually grows a shape. This needs a GPU/MPS and roughly 2–8k iterations per model. Also consider Distill's per-parameter gradient normalisation and an LR schedule.
- **Rule inference ML project (replaces the deleted Rule Predictor):** generate `(initial, after-k-steps)` pairs with `lifelike.run` over the catalog, saved as `.npz`, not CSV. Train a classifier or regressor to recover B/S bits. Baselines: a per-count transition-frequency feature plus a logistic model (it should be near-perfect for k=1, which is a good sanity check), then a small CNN for k>1. This is a strong, self-contained ML portfolio piece.
- Rule 30 keystream demo built on `elementary` (replaces the archived crypto code, with honest framing as a PRNG, not encryption).
- Larger-than-life / Lenia (continuous CA) as a bridge between life-like and NCA.

---

## 13. Frontend: contract and recommended architecture (for the next plan)

**Recommendation: run everything client-side.** Every automaton here is cheap:
- Life at 256×256 is ~65K cells × 9 lookups per step. TypeScript with typed arrays runs it at 60 fps, and a WebGL fragment shader handles 1024²+.
- Elementary CA is trivial.
- NCA at 28×28 is 784 cells × (48·128 + 128·16) ≈ 6.4M multiply-adds per step, roughly 5–10 ms in plain TS. Run it in a Web Worker.

A FastAPI backend that streams frames would add hosting cost, cold starts, and network latency per frame for no benefit. Simulation state lives where it's rendered. (If you want a backend for portfolio reasons, give it real server-side work, e.g. a training-job queue for the Rule-inference project. Don't put it in the render loop.)

Suggested stack: **Vite + React + TypeScript**, deployed as a static site (GitHub Pages / Vercel). Next.js `output: "export"` also works if you'd rather stay on familiar tooling; you don't need SSR. Canvas2D `ImageData` for rendering. One `web/src/engines/{elementary,lifelike,nca}.ts` mirroring the Python modules, with **Vitest tests that load `web/public/data/fixtures/*.json`** and assert parity.

Pages: 1D explorer (rule number + 8 toggleable rule bits + class badges), 2D playground (rule presets, B/S checkbox editor, pattern stamping, draw/erase, wrap/dead toggle, speed, step), NCA lab (model picker, live run with fire-rate slider, click-to-damage, side-by-side filter/loss ablation using the pre-rendered GIFs for the 25-model grid and live inference for the selected one).

---

## Appendix A — Reference numpy NCA forward pass (verified on all 25 checkpoints)

This exact logic produced the §1.4 table. Port it into `NumpyNCA` and keep the semantics identical.

```python
import numpy as np

def corr3(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Zero-padded 3x3 cross-correlation applied to every channel. x: (C,H,W)."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    H, W = x.shape[1:]
    return sum(k[i, j] * p[:, i:i + H, j:j + W] for i in range(3) for j in range(3))

def alive(x: np.ndarray) -> np.ndarray:
    """3x3 max-pool of alpha (channel 3), out-of-bounds ignored, > 0.1."""
    a = np.pad(x[3], 1, constant_values=-np.inf)
    H, W = x.shape[1:]
    return np.max([a[i:i + H, j:j + W] for i in range(3) for j in range(3)], axis=0) > 0.1

def step(x, w1, b1, w2, kernels, fire_rate=0.5, rng=None):
    """x: (C,H,W) float32; w1: (128, 3C); b1: (128,); w2: (C, 128); kernels: (3,3,3) [id, dx, dy]."""
    C = x.shape[0]
    per = np.stack([corr3(x, kernels[k]) for k in range(3)], axis=1).reshape(3 * C, *x.shape[1:])  # 3c+k
    h = np.maximum(np.einsum("oc,chw->ohw", w1, per) + b1[:, None, None], 0.0)
    dx = np.einsum("oc,chw->ohw", w2, h)
    if fire_rate < 1.0:
        dx = dx * (rng.random(x.shape[1:]) <= fire_rate)
    new = x + dx
    return (new * (alive(x) & alive(new))).astype(np.float32)
```

## Appendix B — Torch-free `.pt` reader (verified on all 25 checkpoints)

These files are PyTorch zip checkpoints containing a pickled `OrderedDict` of tensors. Use this reader **only** inside `scripts/convert_checkpoints.py`, a one-time conversion of trusted, repo-owned files. Don't use it in library code. (Unpickling untrusted files is unsafe.)

```python
import collections, pickle, zipfile
import numpy as np

def load_pt_state_dict(path) -> dict[str, np.ndarray]:
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *_):
        return np.lib.stride_tricks.as_strided(
            storage[offset:], shape=size, strides=[s * 4 for s in stride]).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "OrderedDict":
                return collections.OrderedDict
            return lambda *a, **k: None

        def persistent_load(self, pid):  # ('storage', type, key, location, numel)
            return np.frombuffer(z.read(f"{root}/data/{pid[2]}"), dtype="<f4")

    return dict(Unpickler(z.open(f"{root}/data.pkl")).load())
```

---

## Phase checklist

| Phase | Deliverable | Commit(s) |
|---|---|---|
| 0 | Hygiene | 1 |
| 1 | pyproject, ruff, pytest, CI | 1 |
| 2 | `elementary.py` + tests | 1 |
| 3 | `lifelike.py`, `patterns.py`, catalogs + tests | 1–2 |
| 4 | NCA numpy inference + manifest (4a), torch training (4b) | 2 |
| 5 | renderer + CLI | 1 |
| 6 | web export + fixtures + data contract | 1 |
| 7 | archive crypto, delete rule predictor | 1 |
| 8 | README, docs, LICENSE | 1 |
