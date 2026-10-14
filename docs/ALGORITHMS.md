# Algorithms: the browser implementation spec

Everything the browser re-implements live is specified here, with the fixture that verifies it.
Python (`src/cellauto`) is the reference; the files in `web/public/data/fixtures/` are the golden
inputs and outputs. If this document and the Python code disagree, that is a bug in one of them:
fix it and regenerate the fixtures (`cellauto export-web`).

General conventions: cells are indexed `(row, col)` with row 0 at the top; "wrap" is a torus,
"dead" means cells outside the grid are 0. Random initial states **cannot** be reproduced across
languages (numpy's `default_rng` is not available in the browser), so every fixture carries its
explicit `initial` state; use any PRNG you like for live random starts.

Sections: [ECA](#eca) · [Life-like](#lifelike) · [RLE](#rle) · [Rule code](#rule-code) ·
[Lenia](#lenia) · [Neural CA](#nca) · [Transfer matrix](#transfer-matrix) · [Lucas](#lucas).
[Duality](#duality) · [Mean field](#mean-field). Later phases append `#bayes-inference` and
`#relu-compiler`.

---

## ECA

Fixture: `fixtures/elementary.json` (20 cases: rules 30, 54, 90, 110, 184 × wrap/dead × single/random
start, width 31, 15 generations).

**Rule table.** Rule `n` (0-255) is the 8-entry table `table[i] = (n >> i) & 1`, `i = 0..7`.
**Wolfram bit order:** the neighbourhood `(left, centre, right)` is the 3-bit index
`i = 4*left + 2*centre + right`, so bit 7 is the neighbourhood `111` and bit 0 is `000`.
Rule 30 = `[0,1,1,1,1,0,0,0]`, rule 110 = `[0,1,1,1,0,1,1,0]`, rule 90 = `[0,1,0,1,1,0,1,0]`.

**Step.** Every cell updates, including the edges:

```
for x in 0..W-1:
    l = row[x-1]   (wrap: row[(x-1) mod W];   dead: 0 if x == 0)
    r = row[x+1]   (wrap: row[(x+1) mod W];   dead: 0 if x == W-1)
    new[x] = table[4*l + 2*row[x] + r]
```

**Run.** `history[0]` is the initial row and `history[t+1] = step(history[t])`, giving
`generations + 1` rows.

**Initial states.** `single`: all zeros except cell `floor(W/2)`. `random`: each cell is 1 with
probability `density`. `bits`: the user's 0/1 string, left-aligned and zero-padded to the width.

**Edge cases.** Width 1 with wrap: the cell is its own left and right neighbour. Rule 90 on a ring of
width 2^k dies completely after 2^(k-1) steps (the update is nilpotent over GF(2)); this is correct
behaviour, not a bug.

## Life-like

Fixture: `fixtures/lifelike.json` (a glider on 16×16 and a 24×24 soup under five rules, plus
Life on a dead boundary; each case gives `initial`, `steps`, and the `expected` grid after that
many steps, as lists of bit strings).

A rule is a `(2, 9)` lookup table `lut[state][count]` (`state` ∈ {0, 1}, `count` ∈ 0..8) holding the
next state. `count` is the number of live cells among the **8 Moore neighbours, excluding the
centre**. For a rulestring `Bb.../Ss...`: `lut[0][n] = 1` iff `n` is a birth digit, `lut[1][n] = 1`
iff `n` is a survival digit.

```
for each cell (y, x):
    count = sum of grid[y+dy][x+dx] for (dy, dx) != (0, 0) in {-1,0,1}²   (wrap: indices mod H/W;
                                                                           dead: out of range = 0)
    new[y][x] = lut[grid[y][x]][count]
```

**Rulestrings.** Case-insensitive; `B3/S23`, `b3s23`, `S23/B3` (either order), `B2/S` (empty survival),
`B/S`. Digits 0-8 only. The old `23/3` form is rejected. The canonical form is `B` + sorted
birth digits + `/S` + sorted survival digits.

**B0 rules** (birth on 0 neighbours) make the whole background flip every step, so the picture
strobes. The engine computes them correctly on a finite grid; the exported rule `antilife` is one.

## RLE

Fixture: `fixtures/rle.json` (all catalog patterns plus three edge cases).

Run-length-encoded Life patterns:

* Lines starting with `#` are comments and are ignored.
* An optional header line `x = W, y = H[, rule = ...]` declares the size; the decoded grid is
  padded with dead cells to **at least** `H × W`. Without a header the size is the bounding extent
  of the body (including trailing `b` runs).
* Body tokens: `b` = dead, `o` = alive, `$` = end of row, each with an optional decimal run count
  (`3o`, `2b`, `4$`). **`$` takes a count**: `3$` ends the current row and leaves two empty rows.
  Whitespace and line breaks anywhere between tokens are ignored.
* **Trailing dead cells may be omitted** at the end of a row, and a row may be omitted entirely at
  the end of the pattern.
* `!` ends the body. A body without `!`, an unknown character, or a count not followed by a token is
  an error.

Decoding: keep `(row, col)`; `b n` advances `col` by `n`; `o n` sets `n` cells and advances;
`$ n` does `row += n; col = 0`. Fixture cells are `(row, col)` pairs of live cells.

Encoding (not required in the browser): trim the empty border, emit runs per row, drop trailing
dead runs, join rows with `$` (a blank row between two filled rows increases the `$` count), end
with `!`.

## Rule code

Every life-like rule has an 18-bit integer identity: bit `n` (0..8) is "birth on `n`" and bit `9 + n`
is "survival on `n`". Life (`B3/S23`) is `8 + 2048 + 4096 = 6152`. The code is exported as
`code` in `lifelike/rules.json`. Rules without birth on 0 neighbours (B0-free, bit 0 clear) are
the even codes, so `row = code >> 1` indexes the rule-space atlas.

## Lenia

Fixture: `fixtures/lenia.json` (Orbium on 64×64 for 5 steps, `tolerance` 1e-5; and Life as a Lenia on
16×16 for 3 steps, **exact**). Source: Chan, *Lenia: Biology of Artificial Life*, Complex Systems
28(3), 2019.

State `A` is an `H × W` array of floats in [0, 1] on a torus.

**Kernel.** For an integer offset `(dy, dx)`: `r = sqrt(dy² + dx²) / R`. With peaks
`β = (β_1 … β_B)`: if `r >= 1` then `K = 0`; otherwise `br = B·r`, `i = floor(br)` (so
`0 <= i < B`), `q = br − i`, and

```
K(dy, dx) = β_i · exp(4 − 1 / (q (1 − q)))     if 0 < q < 1
          = 0                                    otherwise (this includes the centre, q = 0)
```

then normalise so that `Σ K = 1` over all offsets (the support is the disc `r < 1`; the grid must
satisfy `2R + 1 <= min(H, W)`).

**Potential.** Circular convolution `U[y][x] = Σ K(dy,dx) · A[(y+dy) mod H][(x+dx) mod W]` (the kernel
is symmetric so convolution and correlation coincide). The Python reference uses
`ifft2(fft2(K_origin_at_0) · fft2(A)).real` in float64; direct summation over the `(2R+1)²` window is
equivalent and fine in the browser (≈ 700 multiply-adds per cell for `R = 13`).

**Growth and update.** With `μ` = `mu` and `σ` = `sigma`:

```
G(u)  = 2·exp(−(u − μ)² / (2σ²)) − 1
A_new = clip(A + G(U) / T, 0, 1)
```

**Life as a Lenia** (preset `life_as_lenia`, `growth = life_step`, `kernel_core = life_box`):

* kernel = 3×3 ones with 0 at the centre, **not normalised**; `T = 1`;
* `G(u) = +1` if `round(u) == 3`, `0` if `round(u) == 2`, otherwise `−1` (rounding absorbs FFT noise);
* with a binary `A` this is exactly B3/S23 on a torus (`A_new = clip(A + G, 0, 1)`).

**Presets.** `cells_rle` uses Chan's RLE:

* `.` = 0; `A`..`X` = 1..24; a two-character value is a prefix `p`..`y` (1..10) followed by a letter
  `A`..`X`, with value `prefix · 24 + (letter − 'A' + 1)`; the value is divided by 255 at the end;
* an optional decimal run count precedes any token; `$` ends a row (and takes a count); `!` ends the
  pattern; shorter rows are padded with 0 on the right.

Place a preset centred: `top = (H − h) // 2`, `left = (W − w) // 2`.

**Precision.** State is stored as float32; Python computes the potential in float64 and rounds the
new state to float32. Compare with the fixture's `tolerance` (1e-5), not for equality.

## NCA

Fixture: `fixtures/nca.json` (model `chest-l2-sobel` grown from the seed with `fire_rate = 1`, states
after updates 1, 2, 5, 10, and the hidden activations for the state at step 1; `tolerance` 1e-4).
Reference: Mordvintsev et al., *Growing Neural Cellular Automata*, Distill 2020.

**State.** `x` has shape `(C, H, W)` (here `C = 16`, `H = W = 28`), float32. Channels 0-2 are RGB
(premultiplied by alpha), channel 3 is alpha ("aliveness"), channels 4+ are hidden.
**Seed:** all zeros except channels `3..C−1` equal to 1 at cell `(H // 2, W // 2)`.

**Weights** (`nca/weights/<id>.json`): `kernels` is `(3, 3, 3)` and **already scaled**:
`[identity, dx, dy]`, where `dx = F / s` and `dy = Fᵀ / s` for the filter `F` (Sobel
`[[-1,0,1],[-2,0,2],[-1,0,1]] / 8`, etc.). `w1` is `(hidden, 3C)`, `b1` is `(hidden,)`, `w2` is
`(C, hidden)` (no bias), all row-major float32, base64 little-endian. `fire_rate = 0.5`,
`alive_threshold = 0.1`, `alpha_channel = 3`.

**Update.**

```
alive_pre  = alive(x)                                   # computed BEFORE the update
per[3c+k][y][x] = Σ_{i,j in 0..2} kernels[k][i][j] · x[c][y+i−1][x+j−1]      # zero outside
hidden = relu(w1 @ per + b1)                            # (hidden, H·W)
dx     = w2 @ hidden                                    # (C, H·W)
dx    *= fire_mask                                      # per cell, SHARED by all channels
new    = x + dx
alive_post = alive(new)
x_next = new · (alive_pre AND alive_post)               # dead cells are zeroed in all channels
```

* **Perception layout:** output index `3·c + k` with `k ∈ (identity, dx, dy)`: channel-major. It is a
  depthwise 3×3 **cross-correlation** (no kernel flip) with **zero padding**.
* **Alive:** `alive(x)[y][x] = max over the 3×3 neighbourhood of x[3] > 0.1`. **Out-of-bounds cells
  are ignored** (−∞ padding), not treated as 0. A cell must be alive **before and after** the update.
* **Fire mask:** each cell independently updates with probability `fire_rate`:
  `mask[y][x] = random() <= fire_rate`. With `fire_rate = 1` do not draw random numbers (the fixture
  relies on this).
* **Float32 throughout.** The alive threshold makes tiny errors discontinuous, so fixtures are short
  and the tolerance is loose; do not expect bit-exactness over hundreds of steps.

**Display:** `rgb = clip(1 − clip(alpha, 0, 1) + rgb_premultiplied, 0, 1)` per pixel (composite over
white).

**Damage** (interactive): zero all channels of every cell within the damage disc,
`(xs − cx)² + (ys − cy)² < r²`, where `xs, ys = linspace(-1, 1, size)`, `(cx, cy)` is the click mapped
to that range, and `r = damage_radius` (default 0.25). The Python evaluation draws a random disc
instead (centre ~ U(−0.5, 0.5)², r ~ U(0.1, 0.4)).

**Hidden activations** (`hidden_activations` in the fixture, `(128, 784)`): `relu(w1 @ per + b1)` for the
state at `hidden_state_step`; the browser can colour each cell by a hidden unit to show "inside the
cell".

---

## Transfer matrix

Fixture: `fixtures/elementary_math.json` (`kind: "preimage_count"`, `"garden_of_eden"`).

How many ring configurations `x` of length `N` satisfy `step(x, rule) = y`? Use the de Bruijn graph
on pairs of adjacent cells. A state is `(a, b)`, numbered `2a + b`. For an output bit `y`, define the
4×4 matrix

```
M_y[2a + b][2b + c] = 1   iff   table[4a + 2b + c] == y       (all other entries 0)
```

Reading `y_1 … y_N` around the ring, the number of preimages is the number of closed walks of length
`N` that emit that sequence:

```
preimages(y) = trace( M_{y_1} · M_{y_2} · … · M_{y_N} )          (N >= 1)
```

Entries can reach `2^N`: use `BigInt` beyond 53 bits (about 53 cells). A configuration with
`preimages = 0` is a **Garden of Eden** configuration. Facts the fixture pins down (rings of 12 cells,
4096 configurations): rule 30 has 280 Gardens of Eden, rule 110 has 1971, and rule 90 has 3072.
Rule 90 is surjective on the infinite line, yet most ring configurations have no preimage: surjectivity
on ℤ does not imply surjectivity on rings (on an even ring, rule 90 loses rank 2 over GF(2), so its image
has `2^(N-2)` configurations).

**Surjectivity on the infinite line** (Hedlund; Amoroso-Patt), exported as `surjective` in
`elementary/rules.json`: take the set `S0 = {0, 1, 2, 3}` of de Bruijn states; for each output bit `y`,
the successor set of `S` is `{ 2(ab mod 2) + c : ab in S, c in {0,1}, table[2ab + c] == y }`; the rule is
surjective iff the empty set is never reached from `S0` by repeated successors.

## Lucas

Fixture: `fixtures/elementary_math.json` (`kind: "rule90_closed_form"`, rows `t = 0..16`).

Rule 90 is additive over GF(2): `new[x] = old[x-1] xor old[x+1]`. From a single live cell, row `t`
is the `t`-th row of Pascal's triangle mod 2 (the Sierpinski triangle). Cell `x` (relative to the
centre cell `width // 2`) is live iff

```
|x| <= t,   (t + x) is even,   and with k = (t + x) / 2:   (k & (t - k)) == 0
```

by Lucas' theorem (`C(t, k)` is odd iff the binary digits of `k` are a subset of those of `t`). This
needs no simulation, but is only valid before the pattern wraps: `t <= width // 2`. The number of live
cells in row `t` is `2^popcount(t)` (Gould's sequence).

On a ring of `N = 2^k` cells rule 90 is the linear map `S + S^-1` over GF(2) and
`(S + S^-1)^(2^(k-1)) = 0`: every initial state dies. Use a prime or odd width for rule 90 experiments.

---

## Duality

Fixture: `fixtures/lifelike_math.json` (`kind: "dual"`, one case per catalog rule).

Inverting every cell (live ↔ dead) maps a life-like rule `(B, S)` to its **dual** `(B', S')`:

```
B' = { 8 − n : n ∉ S }          S' = { 8 − n : n ∉ B }
```

On a torus, `1 − step(grid, rule) == step(1 − grid, dual(rule))`. (With a dead border the identity fails
because the outside is not inverted.) Duality is an involution; a rule is **self-dual** if `dual(rule) == rule`
(Day & Night, `B3678/S34678`). Life's dual is `B0123478/S01234678`. There are exactly 512 self-dual rules
(survival determines birth), hence `(2^18 + 2^9) / 2 = 131,328` equivalence classes (Burnside). The B0-free
set is **not** closed under duality: `0 ∈ B'` iff `8 ∉ S`. In terms of the 18-bit [rule code](#rule-code), swap
and invert the two 9-bit halves: `birth' = ~reverse(survival)`, `survival' = ~reverse(birth)`.

**Langton's lambda** for a life-like rule is the probability that a uniformly random 3×3 neighbourhood
maps to 1: `λ = (Σ_{n∈B} C(8,n) + Σ_{n∈S} C(8,n)) / 512`. Life 0.2734, HighLife 0.3281, Seeds 0.0547,
Day & Night 0.5.

## Mean field

Fixture: `fixtures/lifelike_math.json` (`kind: "mean_field"`, tolerance 1e-9).

If cells were independent with density `ρ`, a cell's live-neighbour count would be `Binomial(8, ρ)` and the
next density would be

```
f(ρ) = ρ · Σ_{n∈S} C(8,n) ρⁿ (1−ρ)^(8−n)  +  (1−ρ) · Σ_{n∈B} C(8,n) ρⁿ (1−ρ)^(8−n)
```

**Fixed points** are the roots of `f(ρ) − ρ` on [0, 1]. Find sign changes of `g(ρ) = f(ρ) − ρ` on a
20,001-point grid, refine each by 60 bisection steps, keep exact zeros on the grid (ρ = 0 is always a root for
rules without B0; ρ = 1 is a root iff `8 ∈ S`), and merge roots closer than 1e-3. The derivative is a central
difference with `h = 1e-6` (one-sided at 0 and 1). **Stability:** `|f'| < 1 − 1e-3` stable, `> 1 + 1e-3`
unstable, otherwise **marginal** (near-degenerate roots such as Coral's, close to ρ = 1). Self-dual rules have a
fixed-point set symmetric under `ρ → 1 − ρ`. Life: 0 (stable), 0.1925 (unstable), 0.3702 (stable).

**Mean field versus reality.** The exported `mean_field` block of each catalog rule compares `f`'s attractor
(iterating from ρ = 0.5) with the density of a simulation on a 127×127 torus (not a power of two: Replicator is
linear over GF(2) and dies on 128²), random start at 0.5, 400 steps, mean density of the last 50 steps over
3 seeds. `correlation_gap` is the absolute difference. Mean field works for gas-like rules (Seeds) and fails where
spatial correlations dominate (Life's sparse ash, and the growing clusters of Coral and Diamoeba).
