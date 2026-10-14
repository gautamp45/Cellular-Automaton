# P02: Elementary (1D) engine

**Tier:** Must · **Prereqs:** P01 · **File:** `src/cellauto/engines/elementary.py` (+ `engines/__init__.py`)

## API

```python
def rule_table(rule: int) -> BinaryArray:
    """shape (8,). Index = (left << 2) | (centre << 1) | right  (Wolfram convention: '111' is bit 7).
    ValueError unless int (not bool) with 0 <= rule <= 255.
    Implementation: ((rule >> np.arange(8)) & 1).astype(np.uint8)"""

def neighbourhood_index(row: BinaryArray, boundary: Boundary = "wrap") -> np.ndarray:
    """(l<<2)|(c<<1)|r for every cell. wrap: np.roll. dead: neighbours outside are 0.
    Exposed because P09 analyses reuse it."""

def step(row, rule: int, boundary: Boundary = "wrap") -> BinaryArray:
    """rule_table(rule)[neighbourhood_index(row, boundary)]. EVERY cell updates, edges included (fixes B3)."""

def run(initial, rule: int, generations: int, boundary: Boundary = "wrap") -> BinaryArray:
    """Space-time diagram (generations + 1, width); row 0 == initial. generations=0 valid; negative → ValueError.
    Build the table once, not per step."""

def single_cell(width: int) -> BinaryArray          # 1 at width // 2
def random_row(width: int, density: float = 0.5, seed: int | None = None) -> BinaryArray  # default_rng(seed)
def from_bits(bits: str) -> BinaryArray             # "0110"; ValueError on other chars
def to_bits(row) -> str
```

## Tests: `tests/engines/test_elementary.py` (all values verified)

- `rule_table(30) == [0,1,1,1,1,0,0,0]`, `rule_table(110) == [0,1,1,1,0,1,1,0]`, `rule_table(90) == [0,1,0,1,1,0,1,0]`.
- `rule_table(256)`, `rule_table(-1)` and `rule_table(True)` raise ValueError.
- Rule 30, `run(single_cell(201), 30, 40)[:25, 100]` ==
  `[1,1,0,1,1,1,0,0,1,1,0,0,0,1,0,1,1,0,0,1,0,0,1,1,1]` (OEIS A051023).
- Rule 30: row 1 `[97:104]` == `[0,0,1,1,1,0,0]`; row 2 `[96:105]` == `[0,0,1,1,0,0,1,0,0]`.
- `run(single_cell(11), 110, 1)[1] == [0,0,0,0,1,1,0,0,0,0,0]`.
- Rule 90 (Sierpiński / Gould's sequence): `h = run(single_cell(129), 90, 32)`; for every t in 0..31, `h[t].sum() == 2**bin(t).count("1")`.
- Edges (regression B3), rule 90 on `[1,0,0,0,0]`: `wrap` gives `[0,1,0,0,1]`, `dead` gives `[0,1,0,0,0]`.
- **GF(2) nilpotency trap** (worth a test because it bites P09): `run(random_row(256, seed=3), 90, 128, "wrap")[-1].sum() == 0` for any initial row. On a ring of width 2ᵏ, rule 90 is the linear map S + S⁻¹ over GF(2), and (S + S⁻¹)^(2ᵏ⁻¹) = 0. A width of 255 must *not* die.
- `hypothesis` property tests:
  - for random rule and row (width 1..64), `step` with `wrap` equals a slow pure-Python reference implementation written in the test;
  - mirror symmetry: `step(row[::-1], reflect(rule)) == step(row, rule)[::-1]`, with `reflect` implemented in the test as a bit permutation (P09 moves it into the library).
- `to_bits(from_bits(s)) == s`; `random_row(100, seed=1)` is deterministic.

## Finish
`git rm OD_CA.py`; remove it from ruff `extend-exclude`.

**Commit:** `feat(engines): vectorised elementary CA with wrap/dead boundaries`
