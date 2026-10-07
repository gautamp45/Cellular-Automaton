"""Planning-time checks that were run inline (Oct 2026). Run from repo root: python3 docs/plan/reference/extras.py
Verifies: ReLU compiler stats over all 2^18 life-like rules, duality counts, mean-field fixed points,
Life-as-Lenia equivalence, rule-90 nilpotency on width 2^k but not 255, hand-built 4-ReLU Life net."""
from math import comb

import numpy as np


def counts(g):
    return sum(np.roll(np.roll(g, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def life(g):
    n = counts(g)
    return ((n == 3) | ((g == 1) & (n == 2))).astype(float)


# --- ReLU compiler + duality over all 2^18 rule codes (bit n = birth n, bit 9+n = survival n)
R = 1 << 18
codes = np.arange(R)
bits = ((codes[:, None] >> np.arange(18)) & 1).astype(np.int8)
k = np.arange(18)
idx = np.where(k % 2 == 0, k // 2, 9 + (k - 1) // 2)  # v_k = k/2 = n + s/2
Y = bits[:, idx].astype(np.float64)
slopes = np.diff(Y, axis=1) / 0.5
coef = np.concatenate([slopes[:, :1], np.diff(slopes, axis=1)], axis=1)  # ReLU(v - j/2) coefficients
units = (coef != 0).sum(1)
life_code = (1 << 3) | (1 << 11) | (1 << 12)
print("Life hidden units:", units[life_code], "kinks at v =", (np.nonzero(coef[life_code])[0] / 2).tolist(),
      "coef", coef[life_code][coef[life_code] != 0].tolist())
print("units over all rules: min", units.min(), "max", units.max(), "mean", units.mean(), "hist", np.bincount(units).tolist())
rng = np.random.default_rng(0)
ok = True
for c in rng.integers(0, R, 300):
    g = (rng.random((24, 24)) < rng.random()).astype(np.uint8)
    n = counts(g)
    v = n + 0.5 * g
    out = Y[c, 0] + sum(coef[c, j] * np.maximum(v - j / 2, 0) for j in range(17))
    ok &= np.allclose(out, bits[c].reshape(2, 9)[g, n])
print("compiled ReLU nets exact on 300 random rules:", ok)
B, S = bits[:, :9], bits[:, 9:]
dcode = (np.concatenate([1 - S[:, ::-1], 1 - B[:, ::-1]], 1).astype(np.int64) << np.arange(18)).sum(1)
sd = int((dcode == codes).sum())
print("self-dual:", sd, "involution:", bool((dcode[dcode] == codes).all()), "classes:", (R + sd) // 2,
      "non-B0 closed under duality:", bool(((dcode[codes[(codes & 1) == 0]] & 1) == 0).all()))

# --- hand-built Life network: 3x3 conv (centre 0.5) -> 4 ReLU -> linear
relu = lambda z: np.maximum(z, 0)  # noqa: E731
ok = True
for d in (0.1, 0.3, 0.5, 0.7, 0.9):
    g = (rng.random((200, 200)) < d).astype(float)
    v = counts(g) + 0.5 * g
    ok &= np.array_equal(2 * relu(v - 2) - 2 * relu(v - 2.5) - 2 * relu(v - 3.5) + 2 * relu(v - 4), life(g))
print("hand-built 4-ReLU net == Life:", ok)


# --- mean-field fixed points and lambda
def f(r, Bs, Ss):
    p = [comb(8, n) * r**n * (1 - r) ** (8 - n) for n in range(9)]
    return r * sum(p[n] for n in Ss) + (1 - r) * sum(p[n] for n in Bs)


def fixed_points(Bs, Ss, grid=20001):
    xs = np.linspace(0, 1, grid)
    g = np.array([f(x, Bs, Ss) - x for x in xs])
    out = []
    for i in range(grid - 1):
        if g[i] == 0:
            out.append(xs[i])
        elif g[i] * g[i + 1] < 0:
            a, b = xs[i], xs[i + 1]
            for _ in range(60):
                m = (a + b) / 2
                a, b = (a, m) if (f(a, Bs, Ss) - a) * (f(m, Bs, Ss) - m) <= 0 else (m, b)
            out.append((a + b) / 2)
    if g[-1] == 0:
        out.append(1.0)
    res = []
    for r in out:
        h = 1e-6
        lo, hi = max(r - h, 0), min(r + h, 1)
        dv = (f(hi, Bs, Ss) - f(lo, Bs, Ss)) / (hi - lo)
        res.append((round(float(r), 4), "stable" if abs(dv) < 1 else "unstable"))
    return res


lam = lambda Bs, Ss: sum(0.5 * comb(8, n) / 256 for n in Bs) + sum(0.5 * comb(8, n) / 256 for n in Ss)  # noqa: E731
for name, Bs, Ss in [("life", [3], [2, 3]), ("highlife", [3, 6], [2, 3]), ("seeds", [2], []),
                     ("day_night", [3, 6, 7, 8], [3, 4, 6, 7, 8])]:
    print(f"{name:10s} lambda={lam(Bs, Ss):.4f} fixed points={fixed_points(Bs, Ss)}")

# --- Life as Lenia (integer Moore kernel, T=1, step growth) and rule-90 nilpotency
N = 32
K = np.zeros((N, N))
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy or dx:
            K[dy % N, dx % N] = 1
fK = np.fft.fft2(K)
ok = True
for s in range(20):
    A = (np.random.default_rng(s).random((N, N)) < 0.4).astype(float)
    U = np.rint(np.real(np.fft.ifft2(fK * np.fft.fft2(A))))
    ok &= np.array_equal(np.clip(A + np.where(U == 3, 1, np.where(U == 2, 0, -1)), 0, 1), life(A))
print("Life == Lenia(step growth):", ok)
t90 = (90 >> np.arange(8)) & 1


def run90(x, T):
    for _ in range(T):
        x = t90[(np.roll(x, 1) << 2) | (x << 1) | np.roll(x, -1)]
    return x


r3 = np.random.default_rng(3)
print("rule 90: width 256 dies after 128:", run90(r3.integers(0, 2, 256).astype(np.uint8), 128).sum() == 0,
      "| width 255 alive:", run90(r3.integers(0, 2, 255).astype(np.uint8), 128).sum() > 0)
