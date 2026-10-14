import time

import numpy as np
import pytest

from cellauto.catalog import lenia_presets
from cellauto.engines import lenia, lifelike


def _preset(code: str) -> dict:
    return next(p for p in lenia_presets() if p["code"] == code)


def _params(preset: dict) -> lenia.LeniaParams:
    p = preset["params"]
    return lenia.LeniaParams(
        R=p["R"], T=p["T"], mu=p["mu"], sigma=p["sigma"], peaks=tuple(p["peaks"])
    )


def _torus_dist(a: tuple[float, float], b: tuple[float, float], n: int) -> float:
    d = [min(abs(x - y), n - abs(x - y)) for x, y in zip(a, b, strict=True)]
    return float(np.hypot(*d))


def _centroid(A: np.ndarray) -> tuple[float, float]:
    n = A.shape[0]
    ang = 2 * np.pi * np.arange(n) / n
    cy = np.angle((A.sum(1) * np.exp(1j * ang)).sum()) % (2 * np.pi) * n / (2 * np.pi)
    cx = np.angle((A.sum(0) * np.exp(1j * ang)).sum()) % (2 * np.pi) * n / (2 * np.pi)
    return float(cy), float(cx)


def test_presets_present() -> None:
    codes = [p["code"] for p in lenia_presets()]
    assert codes == ["O2u", "O2ui", "O2b", "life_as_lenia"]
    assert _preset("O2u")["params"] == {
        "R": 13, "T": 10, "mu": 0.15, "sigma": 0.015, "peaks": [1.0],
    }  # fmt: skip


def test_decode_orbium() -> None:
    cells = lenia.decode_lenia_rle(_preset("O2u")["cells_rle"])
    assert cells.shape == (20, 20)
    assert cells.max() == pytest.approx(1.0)
    assert cells.sum() == pytest.approx(76.863, abs=0.01)


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        lenia.decode_lenia_rle("3Z!")
    with pytest.raises(ValueError):
        lenia.decode_lenia_rle("p!")


def test_kernel_normalised_and_compact() -> None:
    params = lenia.LeniaParams()
    k = lenia.kernel(params, (64, 64))
    assert k.sum() == pytest.approx(1.0, abs=1e-6)
    h = 64
    yy, xx = np.ogrid[:h, :h]
    r = np.hypot(yy - h // 2, xx - h // 2)
    assert (k[r >= params.R] == 0).all()
    assert (k[r < params.R].sum()) == pytest.approx(1.0, abs=1e-6)
    multi = lenia.kernel(lenia.LeniaParams(peaks=(1.0, 0.5, 0.25)), (64, 64))
    assert multi.sum() == pytest.approx(1.0, abs=1e-6)


def test_kernel_too_small() -> None:
    with pytest.raises(ValueError):
        lenia.kernel(lenia.LeniaParams(R=13), (20, 20))


def test_params_validation() -> None:
    for bad in ({"R": 0}, {"T": 0.0}, {"sigma": 0.0}, {"peaks": ()}, {"R": True}):
        with pytest.raises(ValueError):
            lenia.LeniaParams(**bad)  # type: ignore[arg-type]


def test_orbium_is_a_glider() -> None:
    preset = _preset("O2u")
    params = _params(preset)
    A = np.zeros((128, 128), dtype=np.float32)
    cells = lenia.decode_lenia_rle(preset["cells_rle"])
    A[40:60, 40:60] = cells
    kf = lenia.kernel_fft(params, A.shape)
    cents: dict[int, tuple[float, float]] = {}
    for t in range(1, 501):
        A = lenia.step(A, params, kf)
        if t >= 100:
            assert 69 <= A.sum() <= 74, (t, A.sum())
        if t in (100, 200):
            cents[t] = _centroid(A)
    assert _torus_dist(cents[100], cents[200], 128) > 20


def test_run_generator_matches_step() -> None:
    params = lenia.LeniaParams()
    rng = np.random.default_rng(0)
    A = rng.random((40, 40)).astype(np.float32)
    seq = list(lenia.run(A, params, 3))
    assert len(seq) == 3
    assert np.array_equal(seq[0], lenia.step(A, params))
    assert np.array_equal(seq[2], lenia.step(lenia.step(seq[0], params), params))


def test_life_as_lenia_equals_life() -> None:
    params, k3 = lenia.life_as_lenia()
    assert k3.tolist() == [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    assert params.T == 1.0
    for seed in range(20):
        A = lifelike.random_grid(32, 32, 0.4, seed)
        got = lenia.step(A.astype(np.float32), params)
        assert np.array_equal(got, lifelike.step(A, "B3/S23", "wrap")), seed


def test_step_speed() -> None:
    params = lenia.LeniaParams()
    A = np.random.default_rng(1).random((128, 128)).astype(np.float32)
    kf = lenia.kernel_fft(params, A.shape)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        lenia.step(A, params, kf)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 0.010
