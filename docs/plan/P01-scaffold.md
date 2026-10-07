# P01: Package scaffold, tooling, CI

**Tier:** Must · **Prereqs:** P00

## Steps

1. `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "cellauto"
   version = "0.1.0"
   description = "Elementary, life-like, continuous and neural cellular automata: reference implementations, analysis, and a static data backend for the web."
   readme = "README.md"
   requires-python = ">=3.10"
   license = { text = "MIT" }
   dependencies = ["numpy>=1.24", "pillow>=10.0", "pydantic>=2.6"]

   [project.optional-dependencies]
   train = ["torch>=2.1", "pyyaml>=6.0", "tqdm>=4.60"]
   dev = ["pytest>=8.0", "hypothesis>=6.100", "ruff>=0.6", "mypy>=1.10"]

   [project.scripts]
   cellauto = "cellauto.cli:main"

   [tool.hatch.build.targets.wheel]
   packages = ["src/cellauto"]

   [tool.ruff]
   line-length = 100
   target-version = "py310"
   extend-exclude = ["experiments", "web", "docs/plan/reference",
     # legacy, removed phase by phase:
     "OD_CA.py", "GameofLife.py", "TD CA", "TD_Rule_Web Scraper.py", "neural-cellular-automata"]

   [tool.ruff.lint]
   select = ["E", "F", "I", "UP", "B", "NPY", "SIM", "RUF"]

   [tool.mypy]
   python_version = "3.10"
   files = ["src/cellauto"]
   ignore_missing_imports = true
   warn_unused_ignores = true
   disallow_untyped_defs = true
   exclude = ["src/cellauto/nca/training", "src/cellauto/learnability", "src/cellauto/inference/differentiable.py"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   addopts = "-m 'not slow'"
   markers = ["torch: requires PyTorch", "slow: > ~5 s; run with -m slow"]
   ```
   **Why mypy only on the non-torch code:** strict typing on the library surface is the professional signal; torch typing is noisy and low value.
2. Files: `src/cellauto/__init__.py` (`__version__ = "0.1.0"`), empty `src/cellauto/py.typed`, and `src/cellauto/__main__.py`:
   ```python
   from cellauto.cli import main

   raise SystemExit(main())
   ```
   A stub `src/cellauto/cli.py`:
   ```python
   def main(argv: list[str] | None = None) -> int:
       return 0
   ```
3. `src/cellauto/_types.py`:
   ```python
   from typing import Literal
   import numpy as np
   import numpy.typing as npt

   Boundary = Literal["wrap", "dead"]
   BinaryArray = npt.NDArray[np.uint8]
   FloatArray = npt.NDArray[np.float32]
   ```
4. `src/cellauto/_validate.py`: `as_binary(arr, ndim) -> BinaryArray` (raises ValueError unless the values are ⊂ {0,1} and the dimension matches; returns a uint8 copy), `check_boundary(b)`, and `check_int_range(name, v, lo, hi)` (rejects bool).
5. `tests/test_smoke.py`: import the package and check the version.
6. `.github/workflows/ci.yml`:
   ```yaml
   name: ci
   on:
     push:
     pull_request:
     workflow_dispatch:
     schedule: [{ cron: "0 3 * * 1" }]   # weekly: runs slow tests
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy: { matrix: { python-version: ["3.10", "3.12"] } }
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "${{ matrix.python-version }}" }
         - run: pip install -e ".[dev]"
         - run: ruff check . && ruff format --check .
         - run: mypy src
         - run: pytest -q
     slow:
       if: github.event_name != 'pull_request'
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.12" }
         - run: pip install -e ".[dev]"
         - run: pytest -q -m slow
     torch:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.12" }
         - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
         - run: pip install -e ".[dev,train]"
         - run: pytest -q -m torch
   ```
   P07 adds an "export is up to date" step.

Note: some sandboxes block `download.pytorch.org`. Torch tests must `pytest.importorskip("torch")`, so they skip locally and run in CI.

7. Keep `CLAUDE.md` and `.claude/` (SessionStart hook) as they are. Once `pyproject.toml` exists, the hook starts installing `cellauto[dev]` automatically in cloud sessions. Validate it with `CLAUDE_CODE_REMOTE=true .claude/hooks/session-start.sh`.

## Done when
`pip install -e ".[dev]"`, then `ruff check . && ruff format --check . && mypy src && pytest -q` passes.

**Commit:** `build: pyproject, ruff, mypy, pytest+hypothesis, CI`
