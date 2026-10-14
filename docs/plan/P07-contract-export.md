# P07: Data contract, web export, golden fixtures, algorithm spec

**Tier:** Must · **Prereqs:** P02–P06

This phase makes the backend **frontend-complete**. Afterwards the frontend reads `web/public/data/index.json` and never needs Python changes. Later phases (P09–P13) only **add** datasets through the same mechanism.

## Contract versioning rule (write it into DATA_CONTRACT.md)

- `format_version` is an integer per dataset. A breaking change bumps it, and that is only allowed with a frontend PR in the same change.
- After P07, phases may only make **additive** changes: new datasets, or new *optional* fields. Every field that a later phase fills is declared **now** as `Optional[...] = None`. For example, `LifeLikeRule.mean_field` is declared in P07 and populated in P10, so the schema doesn't change shape.
- **Recommendation:** start the frontend after P12 for literally zero backend changes. Starting after P07 is also safe, but pages for P09–P13 data must wait for those phases.

## 1. `src/cellauto/contract/models.py` (Pydantic v2: the single source of truth)

All models use `model_config = ConfigDict(extra="forbid", frozen=True)`. Field names are snake_case, and the JSON keeps snake_case (no aliases, except `lambda_` → `"lambda"` via `Field(alias="lambda")` with `populate_by_name=True`).

```python
class Tensor(BaseModel):            shape: list[int]; dtype: Literal["float32-le"]; b64: str
class ParamSpec(BaseModel):
    name: str; label: str; description: str
    kind: Literal["int","float","enum","bool","rulestring","bitstring","float_list","ref"]
    default: int | float | str | bool | list[float] | None
    min: float | None = None; max: float | None = None; step: float | None = None
    options: list[str] | None = None          # enum
    ref: str | None = None                    # dataset id for kind="ref", e.g. "lifelike_rules", "nca_models"
    unit: str | None = None
class EngineSpec(BaseModel):
    id: Literal["elementary","lifelike","lenia","nca"]; name: str; summary: str
    state: Literal["binary","continuous","multichannel"]; dimensions: Literal[1,2]
    params: list[ParamSpec]; algorithm_anchor: str     # anchor in docs/ALGORITHMS.md
    fixtures: str; content_slug: str
class DatasetRef(BaseModel):
    id: str; path: str; schema_path: str; format_version: int; sha256: str; bytes: int
    title: str; description: str; produced_by: str      # e.g. "cellauto analyze eca-atlas"
class Index(BaseModel):
    format_version: Literal[1]; package_version: str; datasets: list[DatasetRef]

class ElementaryRule(BaseModel):
    rule: int; name: str | None; description: str; wolfram_class: int | None = None
    table: list[int]                                           # 8 entries
    equivalence_class: list[int] | None = None                 # P09
    lambda_: float | None = Field(None, alias="lambda")        # P09
    surjective: bool | None = None; reversible: bool | None = None   # P09
class MeanField(BaseModel):                                    # P10
    fixed_points: list[FixedPoint]; simulated_density: float; correlation_gap: float
class FixedPoint(BaseModel): rho: float; derivative: float; stability: Literal["stable","unstable","marginal"]
class LifeLikeRule(BaseModel):
    id: str; name: str; rulestring: str; code: int; table: list[list[int]]   # (2,9)
    description: str; tags: list[str]
    dual: str | None = None; self_dual: bool | None = None                  # P10
    lambda_: float | None = Field(None, alias="lambda")                     # P10
    mean_field: MeanField | None = None                                     # P10
class Pattern(BaseModel):
    id: str; name: str; category: Literal["still_life","oscillator","spaceship","methuselah","gun"]
    rule: str; rle: str; width: int; height: int; cells: list[tuple[int,int]]   # live (row, col)
    period: int | None = None; velocity: str | None = None; description: str
class LeniaPreset(BaseModel):
    id: str; name: str; R: int; T: float; mu: float; sigma: float; peaks: list[float]
    growth: Literal["gaussian","life_step"]; kernel_core: Literal["exp_bump","life_box"]
    cells_rle: str | None; width: int | None; height: int | None; source: str
class NcaMetrics(BaseModel):                                   # P12 fills
    mse_200: Interval; mse_1000: Interval; regen_steps: Interval | None; n_seeds: int
class Interval(BaseModel): mean: float; ci_low: float; ci_high: float
class NcaModel(BaseModel):
    id: str; target: str; loss: str; filter: str; trained_with_damage: bool; isotropic: bool
    n_channels: int; hidden_channels: int; fire_rate: float; grid_size: int; padding: int
    weights_path: str; target_path: str; animation_path: str | None
    metrics: NcaMetrics | None = None
class NcaWeights(BaseModel):
    format_version: Literal[1]; id: str; n_channels: int; hidden_channels: int
    fire_rate: float; alive_threshold: float; alpha_channel: int; grid_size: int; padding: int
    filter: str; kernels: list[list[list[float]]]            # (3,3,3) already scaled: identity, dx, dy
    perception_layout: str; w1: Tensor; b1: Tensor; w2: Tensor
class ColumnSpec(BaseModel):
    name: str; dtype: Literal["uint8","uint16","float32"]; offset: int; length: int
    scale: float | None = None; zero: float | None = None   # value = zero + scale * q  (quantised uint16)
    unit: str | None = None; description: str
class BinaryTableHeader(BaseModel):
    format: Literal["cabin"]; version: Literal[1]; rows: int; row_key: str
    bin_path: str; bin_sha256: str; columns: list[ColumnSpec]
class ContentPage(BaseModel): slug: str; title: str; page: str; order: int; path: str
class Fixture(BaseModel):                    # generic container; `cases` items are engine-specific dicts
    engine: str; description: str; tolerance: float | None; cases: list[dict[str, Any]]
```
P09–P13 each add their own result model in this file (EcaAtlas, LifelikeAtlasMeta, NcaEvaluation, InferenceStudy, ReluExamples, LearnabilityResults).

`src/cellauto/contract/schema.py`: `write_schemas(out_dir)` writes `TypeAdapter(Model).json_schema(by_alias=True)` for every top-level model to `schema/<snake_name>.schema.json`.

**Why Pydantic as the contract:** one definition gives (1) validation of everything we export (a bad export fails in CI, not in a browser), (2) JSON Schema, from which the frontend generates TypeScript types (`npx json-schema-to-typescript`), and (3) self-documenting fields. This replaces hand-maintained TS interfaces drifting from Python dicts.

## 2. `src/cellauto/export/` and the output layout

`export_web(out_dir: Path, repo_root: Path) -> Index`. Every JSON goes through `model.model_dump(mode="json", by_alias=True)`, then `json.dumps(sort_keys=True, separators=(",",":"), ensure_ascii=False)`. No timestamps. Paths inside JSON are relative to `out_dir`.

```
web/public/data/
├── index.json                       # Index (dataset list + sha256 + bytes)
├── engines.json                     # list[EngineSpec]
├── schema/*.schema.json
├── content/index.json + *.md        # list[ContentPage]; md copied from repo content/
├── elementary/rules.json            # list[ElementaryRule] for ALL 256 rules (catalog names/classes merged in)
├── lifelike/rules.json              # list[LifeLikeRule] (catalog)
├── lifelike/patterns.json           # list[Pattern]
├── lenia/presets.json               # list[LeniaPreset]
├── nca/models.json                  # list[NcaModel]
├── nca/weights/<id>.json            # NcaWeights
├── nca/targets/*.png   nca/animations/*.gif
└── fixtures/{elementary,lifelike,rle,lenia,nca}.json   # Fixture
```

**`engines.json` parameter specs** (the frontend builds controls from these; use exactly these defaults and ranges):

| engine | params |
|---|---|
| elementary | `rule` int 0–255 (30); `width` int 16–2048 (256); `generations` int 1–2048 (256); `boundary` enum wrap/dead (wrap); `init` enum single/random/bits (single); `density` float 0–1 step 0.01 (0.5); `seed` int 0–2³¹−1 (0) |
| lifelike | `rule` rulestring (B3/S23), ref `lifelike_rules`; `width`,`height` int 16–2048 (128); `boundary` (wrap); `init` enum random/pattern/empty (random); `density` (0.35); `pattern` ref `lifelike_patterns` (glider); `seed` |
| lenia | `preset` ref `lenia_presets` (O2u); `R` int 3–40 (13); `T` float 1–50 (10); `mu` float 0.01–0.6 step 0.001 (0.15); `sigma` float 0.001–0.2 step 0.0005 (0.015); `peaks` float_list ([1.0]); `size` int 32–512 (128) |
| nca | `model` ref `nca_models` (chest-l2-sobel-damage); `fire_rate` float 0–1 step 0.05 (0.5); `damage_radius` float 0.1–0.4 (0.25, in [−1,1] grid units); `seed` |

**Fixtures** (TypeScript parity tests load these):
- `elementary.json`: rules {30, 54, 90, 110, 184} × boundary {wrap, dead} × init {single_cell(31), random_row(31, seed=7)}, 15 generations. Case = `{rule, boundary, initial: bits, rows: [bits×16]}`.
- `lifelike.json`: glider 16×16 wrap (8 steps); `random_grid(24,24,0.35,seed=0)` under B3/S23, B36/S23, B2/S, B3678/S34678, B1357/S1357, wrap (10 steps); Life with `dead`. Case = `{rule, boundary, initial: [bits rows], steps, expected: [bits rows]}`.
- `rle.json`: every catalog pattern `{rle, width, height, cells}` plus 3 edge cases (header, comments, multi-line body).
- `lenia.json`: Orbium in 64×64 for 5 steps, float32 b64 states at steps 1..5, `tolerance: 1e-5`; plus Life-as-Lenia on a random 16×16 for 3 steps (exact).
- `nca.json`: `chest-l2-sobel` from the seed at fire_rate 1.0: full (16,28,28) states at steps [1,2,5,10] plus `hidden_activations` (128, 28·28) at step 1, as float32 b64, `tolerance: 1e-4`. Keep it short because float32 drift compounds, and the alive threshold makes errors discontinuous.

**Binary columnar format "cabin"** (`export/binary.py`, used from P10). This is a header JSON (`BinaryTableHeader`) plus one `.bin`. The `.bin` is the concatenation of columns, each little-endian and contiguous, and each column offset 8-byte aligned. A uint16 column is quantised as `q = round((v − zero)/scale)`, with `zero = min` and `scale = (max − min)/65535` (scale 1 if max == min). The browser decodes with `new Uint16Array(buf, offset, rows)`. Write `write_cabin(path_stem, columns: dict[str, np.ndarray], specs) -> BinaryTableHeader` and `read_cabin(header_path) -> dict[str, np.ndarray]` (round-trip test: max abs error ≤ scale/2).

**`content/`** (repo root): markdown with a minimal front-matter block of `key: value` lines between `---` fences (`title`, `page`, `order`), parsed without YAML. Create stubs now for these slugs: `intro`, `elementary`, `lifelike`, `lenia`, `nca`, `inference`, `learnability`, `about`. P15 fills them.

## 3. CLI and CI

- `cellauto export-web --out web/public/data` writes everything.
- `cellauto export-web --check` exports into a temp directory and diffs it against `--out`. It exits 1 on any difference and prints the changed files.
- Add to the CI `test` job: `cellauto export-web --check`.
- Commit the generated `web/public/data/`.

## 4. `docs/ALGORITHMS.md` (the browser implementation spec)

One section per anchor: `#eca`, `#lifelike`, `#rle`, `#rule-code`, `#lenia`, `#nca`. P09–P11 add `#transfer-matrix`, `#lucas`, `#duality`, `#mean-field`, `#bayes-inference`, `#relu-compiler`. Each section gives the exact maths, pseudo-code, edge cases, and the fixture that verifies it. It **must** state these gotchas:
- Wolfram bit order: index = 4l + 2c + r; bit 7 = '111'.
- `lut[state][count]`, with counts over the Moore neighbourhood excluding the centre. wrap = torus; dead = zero padding.
- RLE: `$` takes a count; trailing dead cells may be omitted; `!` ends the body.
- Lenia: kernel normalised to sum 1 (except Life-as-Lenia); circular convolution; clip after the update; RLE prefix arithmetic.
- NCA: perception index `3c+k`; kernels already scaled in the JSON; zero padding for perception, but −∞ padding for the alive max-pool; alive is required **before and after** the update; the fire mask is per cell and shared by all channels; display uses `rgb = clip(1 − clip(α) + rgb_premultiplied)`; float32 arithmetic.

`docs/DATA_CONTRACT.md` is a human-readable table of every dataset, linking to its schema and the versioning rule above. It also includes the 20-line JS reader for cabin files.

## 5. Tests: `tests/export/`
- Export twice to `tmp_path` and get byte-identical trees; `index.json` sha256s match the files.
- Every exported JSON validates against its Pydantic model (re-load with `model_validate_json`).
- NCA weights decoded from b64 equal the `.npz` values.
- Re-running each fixture case through the Python engines reproduces it.
- `write_cabin`/`read_cabin` round-trip within the quantisation bound.

**Commits:** `feat(contract): pydantic data contract and JSON schemas`, then `feat(export): deterministic web export, fixtures, algorithm spec`
