# P04: Lenia (continuous CA)

**Tier:** Should · **Prereqs:** P03 · **File:** `src/cellauto/engines/lenia.py`

**Why it's here:** it is the conceptual bridge in the narrative. Life has discrete state, space and time with a hand-written rule. Lenia makes state, space and time continuous but keeps a hand-written rule. NCA then learns the rule. Lenia also showcases FFT convolution and smooth-kernel maths.

## Maths (Chan 2019), as implemented and verified

- State A ∈ [0,1]^{H×W} on a torus.
- Kernel. Let r = ‖x‖ / R for an offset x, and β = (β₁..β_B) the shell peaks. Then
  K(x) = β_{⌊B·r⌋} · K_c(B·r mod 1) for r < 1, else 0, normalised so that ΣK = 1.
  Kernel core (`kn=1`, exponential bump): K_c(q) = exp(4 − 1/(q(1−q))) for 0 < q < 1, else 0.
- Potential U = K ∗ A (circular convolution via FFT: `ifft2(fft2(K_centered) * fft2(A)).real`, with K centred via `ifftshift`).
- Growth (`gn=1`, Gaussian): G(u) = 2·exp(−(u−μ)² / (2σ²)) − 1.
- Update: A ← clip(A + G(U)/T, 0, 1).

## API

```python
@dataclass(frozen=True)
class LeniaParams:
    R: int = 13; T: float = 10.0; mu: float = 0.15; sigma: float = 0.015
    peaks: tuple[float, ...] = (1.0,)       # 'b' in Chan's catalogue
    kernel_core: Literal["exp_bump"] = "exp_bump"
    growth: Literal["gaussian", "life_step"] = "gaussian"

def kernel(params: LeniaParams, shape: tuple[int, int]) -> FloatArray          # spatial, sums to 1
def kernel_fft(params, shape) -> np.ndarray                                      # cached by (params, shape)
def growth(u, params) -> FloatArray
def step(A, params, kfft=None) -> FloatArray
def run(A, params, steps) -> Iterator[FloatArray]
def decode_lenia_rle(text: str) -> FloatArray
    """Chan's format: '.' = 0; 'A'..'X' = 1..24; two-char 'pA'..'yO' where the prefix p..y = 1..10 and
    value = prefix*24 + (letter - 'A' + 1); run counts; '$' row end; '!' end. Divide by 255."""
def life_as_lenia() -> tuple[LeniaParams, np.ndarray]
    """Life as a special case: kernel = 3x3 ones with 0 at the centre (NOT normalised), T = 1,
    growth 'life_step': G(u) = +1 if u == 3, 0 if u == 2, else −1. With binary A this equals B3/S23 exactly."""
```

Reference decoder and step (verified to give a stable Orbium):

```python
def decode_lenia_rle(rle):
    rows=[[]]; n=''; i=0; s=rle.rstrip('!')
    while i < len(s):
        ch=s[i]
        if ch.isdigit(): n+=ch; i+=1; continue
        k=int(n or 1); n=''
        if ch=='$': rows += [[] for _ in range(k)]; i+=1; continue
        if ch=='.': rows[-1] += [0]*k; i+=1; continue
        if 'p'<=ch<='y': v=(ord(ch)-ord('p')+1)*24+(ord(s[i+1])-ord('A')+1); i+=2
        else: v=ord(ch)-ord('A')+1; i+=1
        rows[-1] += [v]*k
    w=max(map(len,rows)); return np.array([r+[0]*(w-len(r)) for r in rows], np.float32)/255
```

## Presets: `src/cellauto/catalog/data/lenia_presets.json`

Fetch https://raw.githubusercontent.com/Chakazul/Lenia/master/Python/animals.json once (MIT license; credit it in the file header and README). Copy these entries verbatim: `O2u` (Orbium unicaudatus: R=13, T=10, m=0.15, s=0.015, b="1"), `O2ui` (ignis), and `O2b` (bicaudatus). Store `{id, name, code, params:{R,T,mu,sigma,peaks}, cells_rle, source}`. If the network is blocked, ask the user to download the file manually. **Do not invent cell data.** Add a non-pattern preset `life_as_lenia`.

## Tests: `tests/engines/test_lenia.py` (verified)

- `decode_lenia_rle(O2u.cells_rle)`: shape (20, 20), max == 1.0, sum ≈ 76.863 (±0.01).
- Orbium in 128×128 placed at (40,40), 500 steps: mass stays in [69, 74] at t = 100..500 (verified ≈ 71.1–71.5). The circular centroid moves > 20 cells (torus distance) between t=100 and t=200, which proves it's a glider (verified ≈ 0.6 cells/step).
- `kernel(...)` sums to 1 (atol 1e-6) and is zero for r ≥ 1.
- **Life as Lenia:** for 20 random binary 32×32 grids, `lenia.step(A, life_params, life_kernel)` equals `lifelike.step(A, "B3/S23", "wrap")` exactly. Use direct convolution for the integer kernel (np.roll sum), or FFT then `np.rint` (FFT round-off is ~1e-12).
- Performance: one 128² step < 10 ms (mark `slow` if flaky). Verified 0.46 ms.

**Commit:** `feat(engines): Lenia continuous CA with FFT convolution, Orbium presets, Life-as-Lenia`
