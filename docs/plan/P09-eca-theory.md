# P09: Elementary CA theory and behaviour atlas

**Tier:** Should · **Prereqs:** P07 · **File:** `src/cellauto/analysis/elementary.py` (+ `analysis/stats.py`, `analysis/embed.py`)

This phase turns 256 rule numbers into a mathematical object you can explore. Every result below was verified during planning.

## 1. Symmetry: the Klein four-group acting on rule space

```python
def reflect(rule: int) -> int      # swap left/right: out bit i ← table[(r<<2)|(c<<1)|l]
def complement(rule: int) -> int   # swap 0/1:        out bit i ← 1 - table[7 - i]
def orbit(rule: int) -> list[int]  # sorted {r, reflect(r), complement(r), reflect(complement(r))}
def equivalence_classes() -> dict[int, list[int]]   # key = min of orbit
```
Verified: **88 classes**. `orbit(30) == [30, 86, 135, 149]`, `orbit(110) == [110, 124, 137, 193]`. Tests: `reflect` and `complement` are involutions and commute (hypothesis over 0..255). Simulation identities: `step(row[::-1], reflect(r)) == step(row, r)[::-1]`, and `step(1-row, complement(r)) == 1 - step(row, r)`.

## 2. Langton's λ
`lambda_(rule) = rule_table(rule).sum() / 8` (fraction of neighbourhoods mapping to 1). Rule 30 gives 0.5 and rule 110 gives 0.625.

## 3. Surjectivity, reversibility, Garden of Eden, preimages

**Surjectivity (Hedlund; Amoroso–Patt)** uses the subset construction on the de Bruijn graph. States are pairs (a,b) ∈ {0..3}. Reading output bit y, (a,b) → (b,c) iff f(a,b,c) = y. Start from the set of all 4 states. The rule is surjective iff the empty set is unreachable.
```python
def is_surjective(rule) -> bool:
    t = rule_table(rule); start = frozenset(range(4)); seen = {start}; stack = [start]
    while stack:
        S = stack.pop()
        for y in (0, 1):
            T = frozenset(((ab << 1) & 3) | c for ab in S for c in (0, 1) if t[(ab << 1) | c] == y)
            if not T: return False
            if T not in seen: seen.add(T); stack.append(T)
    return True
```
Verified: **30 surjective rules**: `[15,30,45,51,60,75,85,86,89,90,101,102,105,106,120,135,149,150,153,154,165,166,169,170,180,195,204,210,225,240]`.

**Reversible** = injective on all cyclic configurations of length 1..10 (exhaustive). Verified: **exactly `[15, 51, 85, 170, 204, 240]`**: identity, shifts, and complements; ECA has no non-trivial reversible rules. (Could: implement the proper Amoroso–Patt pair-graph injectivity test, and check it agrees.)

**Preimage counting by transfer matrices.** For output bit y define the 4×4 matrix M_y[(a,b),(b,c)] = 1 iff f(a,b,c) = y. For a cyclic configuration y₁…y_N, the number of preimages is trace(M_{y₁} ⋯ M_{y_N}). Use int64 (or Python ints for N > 60).
```python
def preimage_count(rule, config: str | BinaryArray) -> int
def garden_of_eden_count(rule, N) -> int        # configs on a ring of length N with 0 preimages
```
Verified: the transfer-matrix count equals brute force for rules {0, 30, 90, 110, 150, 184}, N = 10, on 20 random configs. Garden-of-Eden counts at N = 12: **rule 30 → 280, rule 110 → 1971, rule 90 → 3072** (out of 4096).
**Subtlety to explain in content:** rule 90 is surjective on the infinite line, yet 3072/4096 ring configurations have no preimage. Surjectivity on ℤ does not imply surjectivity on rings. For rule 90 on even N, the GF(2) map has rank N − 2 (an image of 2^(N−2) = 1024 at N = 12, consistent with 3072 GoE).

## 4. Additive rules: exact closed form via GF(2)
Rule 90 from a single cell: cell x at time t is 1 iff (t + x) is even, |x| ≤ t, and with k = (t + x)/2, `k & (t − k) == 0` (Lucas' theorem: C(t,k) is odd iff k's bits ⊆ t's bits).
```python
def rule90_closed_form(t: int, width: int) -> BinaryArray
```
Verified equal to simulation for t ≤ 128 on width 257. Also add `gf2_transition_matrix(rule, N)` for additive rules (90, 150, 60, 102) plus `gf2_rank`. Test: rank of rule 90 at N = 12 is 10. Test the nilpotency trap: `matrix_power` mod 2 of rule 90 at N = 256, power 128, is 0.

## 5. Behaviour features → unsupervised classification

**Measurement protocol (fixed; avoids the GF(2) trap).** Width **251** (prime), T = 256, statistics over t ∈ [128, 256), 6 random initial rows with `default_rng(seed + trial)`:

| feature | definition |
|---|---|
| `damage` | flip the centre cell in a copy; mean Hamming fraction between the copies over the window (a Lyapunov-like spreading rate) |
| `compression` | zlib (level 9) size / raw size of `np.packbits(window)`; ≈1.0 incompressible, ≈0 trivial |
| `density` | mean of the window |
| `activity` | mean fraction of cells that change per step |
| `block_entropy` | Shannon entropy (bits) of length-3 spatial blocks over the window, divided by 3 |
| `period` | smallest p ≤ 64 such that the last row equals the row p steps earlier shifted by some s ∈ [−p, p], else 0 (aperiodic) |

Verified reference values for the first three (6 trials): rule 30: damage 0.448, compression 1.003, density 0.500. Rule 110: 0.186, 0.641, 0.570. Rule 54: 0.195, 0.646, 0.479. Rule 90: 0.136, 1.003, 0.502. Rule 184: 0.044, 0.092, 0.523. Rule 0: all 0 (compression ≈ 0.007). Tests use tolerance ±0.03 on these.

**Consistency property test (math sanity):** within a symmetry orbit, `damage`, `compression` and `activity` agree within ±0.05, and `density(complement(r)) ≈ 1 − density(r)` within ±0.03.

**Embedding and clustering** (`analysis/embed.py`, numpy only; no sklearn, so it stays inspectable and dependency-free):
- `standardize(X)`, `pca(X, k=2)` via SVD (sign-fix each component so its largest-|loading| entry is positive, for determinism), `kmeans(X, k, n_init=20, seed)` with k-means++ init, `silhouette(X, labels)`.
- Unit tests on synthetic Gaussian blobs: PCA recovers the axis of a known anisotropic Gaussian; k-means finds 3 well-separated blobs; silhouette > 0.7 on them.
- Pick k ∈ 3..6 by silhouette; also always report k = 4.

**Evaluation against reference labels** (only the 18 catalog rules with `wolfram_class`; never invent labels for other rules): produce a contingency table of cluster vs class. Verified with the (damage, compression) features alone: classes I and II merge into one cluster; chaotic class III rules (22, 30, 45, 90, 150) share the high-compression cluster; class IV (54, 110) sits in an intermediate cluster together with 18 and 126. Report whatever the full feature set gives, honestly. The point is "unsupervised features recover a coarse version of Wolfram's classes, and the class III/IV boundary is genuinely fuzzy", which matches the literature.

## 6. Export (additive)

- Fill `ElementaryRule.{equivalence_class, lambda, surjective, reversible}` for all 256 rules in `elementary/rules.json`.
- New dataset `elementary/atlas.json` with model `EcaAtlas`: `features` (names, descriptions, units), `rows` (256 × {rule, features…, pca_x, pca_y, cluster}), `cluster_summary`, `contingency`, `protocol` (width, T, trials).
- New fixture `fixtures/elementary_math.json`: preimage counts for 10 (rule, config) pairs, GoE counts (N = 8 for rules 30, 90, 110), and the rule 90 closed form at t = 0..16.
- `ALGORITHMS.md`: `#transfer-matrix`, `#lucas`.
- CLI: `cellauto analyze eca-atlas [--quick]` writes `results/eca_atlas.json` (with meta). `export-web` reads results from `results/`.

Runtime: verified ~16 s for all 256 rules (3 features). The full feature set should stay under 1 minute.

**Commit:** `feat(analysis): ECA symmetry, surjectivity, preimages, GF(2) closed forms, behaviour atlas`
