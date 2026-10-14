# Data contract

The backend is a folder of static files, `web/public/data/`, produced by `cellauto export-web` and
committed. The frontend reads `index.json`, discovers datasets from it, and never needs a Python
change to add a control or a page. The contract is defined once, as Pydantic v2 models in
`src/cellauto/contract/models.py`; the JSON Schemas in `schema/` are generated from them, and every
exported file is validated against its model before it is written.

## Versioning rule

* `format_version` is an integer per dataset (and for `index.json`). A breaking change bumps it, and
  is only allowed together with a frontend change.
* Since P07, changes are **additive only**: new datasets, or new *optional* fields (declared `None`
  by default). Every field that a later phase fills in was declared up front (for example
  `LifeLikeRule.mean_field` is `null` until the life-like theory phase populates it), so the shape
  of the schema does not change.
* Field names are snake_case in the JSON. The one alias is `lambda_` ↔ `"lambda"`.
* The export is deterministic: sorted keys, compact separators, no timestamps, trailing newline.
  `cellauto export-web --check` fails if the committed files are stale.

## Datasets

All paths are relative to the data root. `index.json` lists **every** other file with its sha256 and
size (`DatasetRef`); use it for discovery and cache-busting.

| id | path | model (`schema/<name>.schema.json`) | content |
|---|---|---|---|
| `engines` | `engines.json` | `engines` (list of `EngineSpec`) | parameter specs (name, kind, range, default, ref) that UI controls are generated from |
| `elementary_rules` | `elementary/rules.json` | `elementary_rules` | all 256 rules: table, symmetry class, `lambda`, `surjective`, `reversible`; catalog names and Wolfram classes merged in |
| `eca_atlas` | `elementary/atlas.json` | `eca_atlas` | behaviour features, PCA coordinates and clusters for all 256 rules, with the cluster-vs-Wolfram-class contingency table (`produced_by`: `cellauto analyze eca-atlas`) |
| `lifelike_rules` | `lifelike/rules.json` | `lifelike_rules` | named B/S rules: rulestring, 18-bit code, `(2, 9)` table, `dual`, `self_dual`, `lambda`, and (once the atlas has been run) a `mean_field` block |
| `lifelike_atlas_header` / `lifelike_atlas_bin` | `lifelike/atlas.header.json`, `lifelike/atlas.bin` | `binary_table_header` | the rule-space atlas: all 131,072 B0-free rules × 11 columns in the cabin format below; **row `i` is the rule with code `2·i`** |
| `lifelike_atlas_meta` | `lifelike/atlas_summary.json` | `lifelike_atlas_meta` | atlas protocol, PCA components, cluster summaries, the 20-bin edge-of-chaos table, Spearman correlations, and the catalog rules' row indices |
| `relu_examples` | `inference/relu_examples.json` | `relu_examples` | every catalog rule compiled to an exact ReLU network, plus the histogram of hidden units over all 2^18 rules |
| `lifelike_patterns` | `lifelike/patterns.json` | `lifelike_patterns` | named patterns: RLE, size, live cells |
| `lenia_presets` | `lenia/presets.json` | `lenia_presets` | Lenia presets (Orbium family, Life as a Lenia) |
| `nca_models` | `nca/models.json` | `nca_models` | the 25 trained NCA checkpoints and their configuration |
| `nca_weights/<id>` | `nca/weights/<id>.json` | `nca_weights` | scaled kernels + float32 weights (base64) |
| `nca_target/<t>` | `nca/targets/<t>.png` | – | 28×28 RGBA target image |
| `nca_animation/<id>` | `nca/animations/<id>.gif` | – | growth animation |
| `fixture/<name>` | `fixtures/{elementary,elementary_math,lifelike,lifelike_math,rle,lenia,nca,inference}.json` | `fixture` | golden inputs/outputs (see [ALGORITHMS.md](ALGORITHMS.md)) |
| `content_index` | `content/index.json` | `content_index` | explainer pages in display order |
| `content/<slug>` | `content/<slug>.md` | – | Markdown with a `key: value` front-matter block (`title`, `page`, `order`) |
| `schema/<name>` | `schema/<name>.schema.json` | – | generated JSON Schemas |

`ref` parameters in `engines.json` (`lifelike_rules`, `lifelike_patterns`, `lenia_presets`,
`nca_models`) name a dataset id above; their `default` is an `id` inside that dataset (for Lenia the
default preset id is `o2u`, lowercase).

Generate TypeScript types from the schemas with
`npx json-schema-to-typescript schema/engines.schema.json` (and so on).

## Fixtures

Each fixture is `{engine, description, tolerance, cases}`. `tolerance` is `null` for exact
(integer) fixtures. Float32 arrays are `Tensor`s: `{shape, dtype: "float32-le", b64}`. Decode with:

```js
const f32 = (t) => new Float32Array(Uint8Array.from(atob(t.b64), (c) => c.charCodeAt(0)).buffer);
```

Bit-string grids are lists of strings (one per row). The case schemas are documented in
[ALGORITHMS.md](ALGORITHMS.md) next to each algorithm. `cellauto export-web --check` compares all
files byte-for-byte **except** the two float-heavy fixtures (`fixtures/lenia.json`,
`fixtures/nca.json`), whose tensors are compared within a tenth of their `tolerance`; BLAS and FFT
builds legitimately differ in the last bits.

## Binary tables ("cabin")

Large tabular data (the 131,072-rule atlas) is a `BinaryTableHeader` JSON plus one `.bin`. The `.bin`
is the concatenation of columns; each column is little-endian, contiguous, and starts at an
8-byte-aligned **byte** `offset` (`length` is its byte length). `rows` is the row count. A `uint16`
column with a `scale` and `zero` is quantised: `value = zero + scale · q`, with error at most
`scale / 2`; without them it is an exact integer. `row_key` explains how rows map to entities.

```js
async function loadCabin(headerUrl) {
  const header = await (await fetch(headerUrl)).json();
  const binUrl = new URL(header.bin_path, new URL(headerUrl, location.href));
  const buf = await (await fetch(binUrl)).arrayBuffer();
  const Typed = { uint8: Uint8Array, uint16: Uint16Array, float32: Float32Array };
  const cols = {};
  for (const c of header.columns) {
    const raw = new Typed[c.dtype](buf, c.offset, header.rows); // views, no copy
    if (c.scale == null) { cols[c.name] = raw; continue; }      // exact values
    const out = new Float64Array(header.rows);
    for (let i = 0; i < header.rows; i++) out[i] = c.zero + c.scale * raw[i];
    cols[c.name] = out;
  }
  return { rows: header.rows, columns: cols };
}
```

(Typed arrays assume a little-endian host, which every browser platform is.) The header's
`bin_sha256` can be used to verify the download.

## Regenerating

```bash
cellauto export-web            # write web/public/data
cellauto export-web --check    # exit 1 and list files if the committed output is stale
```

## Datasets produced by analyses

Some datasets are computed by `cellauto analyze …` into `results/*.json` (committed; the file holds
the dataset plus a `meta` block with runtime, versions and machine, which is **not** exported because
it is not deterministic). `cellauto export-web` copies the dataset part into `web/public/data/` and
skips it if the results file does not exist, so a missing analysis only removes its page, never breaks
the export. Currently: `eca_atlas` (`results/eca_atlas.json`) and the life-like atlas
(`results/lifelike_atlas/{atlas.header.json,atlas.bin,summary.json}`, whose `mean_field` block also
fills `LifeLikeRule.mean_field`). The atlas columns are `density`, `density_std`, `activity`, `damage`,
`compression`, `lambda`, `mf_rho`, `pca_x`, `pca_y` (quantised uint16, see each column's `scale`/`zero`)
and `cluster`, `is_catalog` (uint8).
