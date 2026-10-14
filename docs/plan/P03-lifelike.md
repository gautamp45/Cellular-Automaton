# P03: Life-like (2D) engine, rulestrings, RLE, catalogs

**Tier:** Must · **Prereqs:** P02

## 3a. `src/cellauto/engines/lifelike.py`

```python
@dataclass(frozen=True)
class LifeRule:
    birth: frozenset[int]
    survival: frozenset[int]

    @classmethod
    def parse(cls, text: str) -> "LifeRule":
        """B/S notation, case-insensitive, optional '/', either order:
        'B3/S23', 'b3s23', 'S23/B3', 'B2/S', 'B/S'. Digits 0-8 only.
        Reject legacy '23/3' with a message pointing to B/S notation. Else ValueError."""
    @classmethod
    def from_code(cls, code: int) -> "LifeRule":
        """18-bit rule code: bit n (0..8) = birth on n; bit 9+n = survival on n."""
    @property
    def code(self) -> int
    @property
    def rulestring(self) -> str            # canonical 'B3/S23'
    def table(self) -> BinaryArray         # (2, 9): lut[0, n] birth, lut[1, n] survival

def neighbour_counts(grid, boundary: Boundary = "wrap") -> np.ndarray:
    """uint8 Moore counts. wrap: sum of 8 np.roll shifts. dead: np.pad(grid, 1) and sum 8 slices.
    Must also work on a batch (..., H, W): roll/pad only the last two axes (P10 relies on this)."""

def step(grid, rule: LifeRule | str, boundary: Boundary = "wrap") -> BinaryArray:
    """rule.table()[grid, neighbour_counts(grid, boundary)]"""

def step_batch(grids: BinaryArray, tables: BinaryArray, boundary="wrap") -> BinaryArray:
    """grids (R,H,W), tables (R,2,9): tables[arange(R)[:,None,None], grids, counts]. Used by P10/P11."""

def iterate(grid, rule, boundary="wrap") -> Iterator[BinaryArray]
def run(grid, rule, steps: int, boundary="wrap") -> BinaryArray      # (steps+1, H, W)
def random_grid(height, width, density=0.5, seed=None) -> BinaryArray
```

**Why the LUT-index approach:** `lut[grid, counts]` is one vectorised fancy-index. It replaces string matching (bugs B4/B5) and is exactly what a shader does. The 18-bit `code` gives every rule an integer identity; P10's atlas uses `row = code >> 1` for B0-free rules.

## 3b. `src/cellauto/catalog/rle.py` and `patterns.py`

```python
def parse_rle(text: str) -> BinaryArray:
    """Life RLE: ignore lines starting with '#'; optional header 'x = W, y = H[, rule = ...]' → pad to W×H.
    Body: [count]('b'|'o'|'$') ... '!'. Whitespace ignored. Pad rows right with 0. Unknown token → ValueError."""
def to_rle(grid) -> str          # no header, trims empty border, ends with '!'
def place(pattern, shape, at=None) -> BinaryArray   # default centred at ((H-h)//2, (W-w)//2); ValueError if no fit
def get_pattern(pattern_id) -> BinaryArray
```

## 3c. Catalog data: `src/cellauto/catalog/data/*.json`, loaded by `catalog/rules.py`

Load the data with `importlib.resources.files("cellauto.catalog") / "data"`, and expose `elementary_rules()`, `lifelike_rules()`, `patterns()`, `lifelike_rule(id) -> LifeRule`.

**`patterns.json`** (RLEs verified):
```json
[
 {"id":"block","name":"Block","category":"still_life","rle":"2o$2o!"},
 {"id":"beehive","name":"Beehive","category":"still_life","rle":"b2o$o2bo$b2o!"},
 {"id":"blinker","name":"Blinker","category":"oscillator","period":2,"rle":"3o!"},
 {"id":"toad","name":"Toad","category":"oscillator","period":2,"rle":"b3o$3o!"},
 {"id":"beacon","name":"Beacon","category":"oscillator","period":2,"rle":"2o$2o$2b2o$2b2o!"},
 {"id":"pulsar","name":"Pulsar","category":"oscillator","period":3,"rle":"2b3o3b3o2b2$o4bobo4bo$o4bobo4bo$o4bobo4bo$2b3o3b3o2b2$2b3o3b3o2b$o4bobo4bo$o4bobo4bo$o4bobo4bo2$2b3o3b3o!"},
 {"id":"glider","name":"Glider","category":"spaceship","period":4,"velocity":"c/4 diagonal","rle":"bo$2bo$3o!"},
 {"id":"lwss","name":"Lightweight spaceship","category":"spaceship","period":4,"velocity":"c/2 orthogonal","rle":"bo2bo$o4b$o3bo$4o!"},
 {"id":"r_pentomino","name":"R-pentomino","category":"methuselah","rle":"b2o$2o$bo!"},
 {"id":"acorn","name":"Acorn","category":"methuselah","rle":"bo5b$3bo3b$2o2b3o!"},
 {"id":"diehard","name":"Diehard","category":"methuselah","rle":"6bob$2o6b$bo3b3o!"},
 {"id":"gosper_glider_gun","name":"Gosper glider gun","category":"gun","period":30,"rle":"24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!"}
]
```
Add `description` (one sentence) and `rule: "B3/S23"` to each entry.

**`lifelike_rules.json`**: entries have `id`, `name`, `rulestring`, `description`, and `tags` (e.g. `["chaotic"]`, `["self-dual"]`):

| id | rulestring | id | rulestring |
|---|---|---|---|
| life | B3/S23 | diamoeba | B35678/S5678 |
| highlife | B36/S23 | morley | B368/S245 |
| day_and_night | B3678/S34678 | anneal | B4678/S35678 |
| seeds | B2/S | gnarl | B1/S1 |
| life_without_death | B3/S012345678 | long_life | B345/S5 |
| replicator | B1357/S1357 | amoeba | B357/S1358 |
| maze | B3/S12345 | assimilation | B345/S4567 |
| mazectric | B3/S1234 | coagulations | B378/S235678 |
| coral | B3/S45678 | walled_cities | B45678/S2345 |
| two_by_two | B36/S125 | stains | B3678/S235678 |
| life_34 | B34/S34 | dotlife | B3/S023 |
| serviettes | B234/S | pseudo_life | B357/S238 |
| flock | B3/S12 | live_free_or_die | B2/S0 |
| antilife | B0123478/S01234678 | | |

`antilife` is the dual of Life (see P10) and was already present as a comment in the original `TD CA/main_run.py`. It is a B0 rule: the engine handles it correctly on a finite grid, but it strobes visually. Tag it `["b0", "dual-of-life"]`.

**`elementary_rules.json`**: entries have `rule`, `name` (optional), `wolfram_class` (1–4, **only** for these reference rules), and `description`:
0 (1), 8 (1), 32 (1), 40 (1), 1 (2), 4 (2), 108 (2), 184 (2, traffic flow), 18 (3), 22 (3), 30 (3, PRNG), 45 (3), 90 (3, additive, Sierpiński), 126 (3), 150 (3, additive), 54 (4), 106 (4, disputed; say so), 110 (4, Turing-complete, Cook 2004).

Delete `TD_Rule_Web Scraper.py`; this static catalog replaces runtime scraping.

## 3d. Tests (all values verified)

`tests/engines/test_lifelike.py`
- Parse: `parse("b3s23").rulestring == "B3/S23"`; `parse("S23/B3") == parse("B3/S23")`; `parse("B2/S").survival == frozenset()`. `parse("B9/S23")`, `parse("23/3")` and `parse("hello")` raise.
- `parse("B3/S23").table().tolist() == [[0,0,0,1,0,0,0,0,0],[0,0,1,1,0,0,0,0,0]]`.
- `LifeRule.from_code(r.code) == r` for every catalog rule (hypothesis: for every code in 0..2¹⁸−1).
- Regression B1: a 5×5 grid with (0,0),(0,1),(1,0) alive under `dead` gives `counts[0,1] == 2`. With `wrap`, a single cell at (0,0) gives `counts[4,4] == 1`.
- `step_batch` equals per-rule `step` for 16 random rules.
- Every catalog rulestring round-trips.
- Hypothesis: `step` equals a slow reference loop for random rule, grid (≤ 12×12) and boundary.

`tests/catalog/test_patterns.py`: Life, `dead` boundary, pattern centred in 80×80 (diehard in 120×120):

| pattern | shape | pop₀ | expectation |
|---|---|---|---|
| block / beehive | (2,2) / (3,4) | 4 / 6 | `step(g) == g` |
| blinker / toad / beacon | (1,3) / (2,4) / (4,4) | 3 / 6 / 8 | g₂ == g₀ |
| pulsar | (13,13) | 48 | g₃ == g₀ |
| glider | (3,3) | 5 | g₄ == g₀ shifted (+1 row, +1 col) |
| lwss | (4,5) | 9 | g₄ == g₀ shifted (0, −2) |
| r_pentomino | (3,3) | 5 | pop@30 = 27, pop@130 = 178 |
| acorn | (3,7) | 7 | pop@30 = 57 |
| diehard (120²) | (3,8) | 7 | pop@129 = 2, pop@130 = 0 |
| gosper_glider_gun at (2,2) in 120² | (9,36) | 36 | pop at t = 0, 30, 60, 90, 120 = 36, 41, 46, 51, 56 |

Also: `parse_rle(to_rle(p))` round-trips for every pattern; a header plus `#C` comments parses; `"3x!"` raises.

## Finish
`git rm -r GameofLife.py "TD CA" "TD_Rule_Web Scraper.py"`; remove them from ruff excludes.

**Commits:** `feat(engines): vectorised life-like engine, rule codes, batch stepping` and `feat(catalog): RLE parser, pattern and rule catalogs`
