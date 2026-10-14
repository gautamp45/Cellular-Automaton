# Planning-time reference scripts

These are the throwaway prototypes that produced every number marked **verified** in the phase files. They are **evidence, not production code**:
- they are not part of the package, and they're excluded from ruff, mypy and pytest;
- don't refactor them, and don't import them from `src/`.

When a phase asks you to implement something, these scripts show one known-correct way to do it. When a test fails against a verified value, run the matching script to confirm the value and to compare intermediate results.

Run from the **repo root**. All of them were re-run successfully from this location on 2026-10-07.

| script | verifies | phases | runtime |
|---|---|---|---|
| `vectors.py` | ECA rule tables, rule 30/90/110 vectors, all pattern behaviours (glider shift, periods, methuselah populations, Gosper gun +5 per 30 gens) | P02, P03 | ~5 s |
| `eca_math.py` | 88 symmetry classes, 30 surjective / 6 reversible rules, transfer-matrix preimages vs brute force, Garden-of-Eden counts, rule 90 Lucas closed form. **Its metric section deliberately uses width 256, the GF(2) nilpotency trap**: rule 90 reads as "dead" there. | P09 | ~15 s |
| `eca_atlas.py` | the corrected behaviour-metric protocol (width 251, time-averaged) and the k-means sanity check | P09 | ~16 s |
| `life_math.py` | mean-field vs simulated density, batched rule simulation timing | P10 | ~15 s |
| `infer.py` | Bayesian rule recovery, exhaustive 2¹⁸ search, duality (by simulation) | P10, P11 | ~5 s |
| `extras.py` | ReLU compiler (Life = 4 units; 0–17 units, mean 12.5 over all rules), self-dual count 512 / 131,328 classes, mean-field fixed points and λ, Life-as-Lenia, rule 90 width 255 vs 256, hand-built Life net | P04, P10, P11, P13 | ~10 s |
| `lenia.py` | Orbium decode and glider stability | P04 | <1 s (needs `animals.json`; see below) |
| `np_nca.py` | torch-free `.pt` reader + numpy NCA; MSE@50/100/200 for all 25 checkpoints | P05 | ~60 s |
| `nca_eval.py` | regeneration pilot (5 seeds): damage-trained 35 ± 8 vs 60 ± 17 steps | P12 | ~15 s |

Notes:
- `lenia.py` needs Chan's catalogue (MIT). It's not committed, so download it first:
  `curl -sSO https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json && python3 docs/plan/reference/lenia.py animals.json && rm animals.json`
- `np_nca.py` and `nca_eval.py` read the **pre-refactor** paths `neural-cellular-automata/{models,data}/...`. After P05 moves those files they stop working, and that's expected; the production equivalents live in `src/cellauto/nca/` by then. To re-run them later, check out commit `1ff3804`.
- `nca_eval.py` uses one RNG stream for both the fire mask and the damage mask. P12 deliberately fixes that with separate streams, so don't copy that part.
