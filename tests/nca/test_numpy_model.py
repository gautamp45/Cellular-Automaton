import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cellauto.nca.checkpoints import find_repo_root, load_manifest, load_model
from cellauto.nca.image import circle_mask, load_target, make_seed, pad, rgba_to_rgb
from cellauto.nca.numpy_model import NumpyNCA
from cellauto.nca.spec import NCAConfig, is_isotropic, perception_kernels

ROOT = find_repo_root(Path(__file__))
MANIFEST = load_manifest(ROOT)


def test_manifest_and_weights() -> None:
    assert len(MANIFEST) == 25
    assert [m["id"] for m in MANIFEST] == sorted(m["id"] for m in MANIFEST)
    for m in MANIFEST:
        assert (ROOT / m["target_image"]).is_file()
        assert (ROOT / m["animation"]).is_file()
        assert (ROOT / m["torch_checkpoint"]).is_file()
        with np.load(ROOT / m["weights"]) as z:
            assert z["w1"].shape == (128, 48) and z["w1"].dtype == np.float32
            assert z["b1"].shape == (128,)
            assert z["w2"].shape == (16, 128)


def test_npz_matches_torch_free_reader() -> None:
    spec = importlib.util.spec_from_file_location(
        "convert", ROOT / "scripts/convert_checkpoints.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for m in MANIFEST:
        sd = mod.load_pt_state_dict(ROOT / m["torch_checkpoint"])
        with np.load(ROOT / m["weights"]) as z:
            assert np.array_equal(z["w1"], sd["update_module.0.weight"][:, :, 0, 0])
            assert np.array_equal(z["b1"], sd["update_module.0.bias"])
            assert np.array_equal(z["w2"], sd["update_module.2.weight"][:, :, 0, 0])


def test_kernels_and_isotropy() -> None:
    k = perception_kernels("sobel")
    assert k.shape == (3, 3, 3)
    assert np.array_equal(k[1], np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) / 8)
    assert np.array_equal(k[2], k[1].T)
    assert k[0].tolist() == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    iso = {n for n in ("sobel", "scharr", "gaussian", "laplacian", "mean") if is_isotropic(n)}
    assert iso == {"gaussian", "laplacian", "mean"}
    with pytest.raises(ValueError):
        perception_kernels("nope")


def test_config_validation() -> None:
    for bad in ({"n_channels": 3}, {"filter": "x"}, {"fire_rate": 0.0}, {"hidden_channels": 0}):
        with pytest.raises(ValueError):
            NCAConfig(**bad)  # type: ignore[arg-type]


def test_fire_rate_one_is_deterministic_and_rng_free() -> None:
    model = load_model("chest-l2-sobel", ROOT)
    seed = make_seed(28)
    m1 = NumpyNCA(model.w1, model.b1, model.w2, NCAConfig(fire_rate=1.0))
    ra = list(m1.run(seed, 20))  # rng=None: must not need or consume an RNG
    rb = list(m1.run(seed, 20))
    assert all(np.array_equal(p, q) for p, q in zip(ra, rb, strict=True))
    # the model default (fire_rate 0.5) cannot run without an rng
    with pytest.raises(ValueError):
        next(iter(model.run(seed, 1)))


def test_step_from_seed_alpha_stays_local() -> None:
    m = load_model("chest-l2-sobel", ROOT)
    out = m.step(make_seed(28), fire_rate=1.0)
    assert isinstance(out, np.ndarray)
    nz = np.argwhere(out[3] != 0)
    assert nz.size > 0
    assert nz.min() >= 13 and nz.max() <= 15


def test_step_return_dx_and_hidden_shapes() -> None:
    m = load_model("chest-l2-sobel", ROOT)
    out, dx = m.step(make_seed(28), fire_rate=1.0, return_dx=True)  # type: ignore[misc]
    assert out.shape == dx.shape == (16, 28, 28)
    assert m.hidden_activations(make_seed(28)).shape == (128, 28 * 28)
    assert m.perceive(make_seed(28)).shape == (48, 28 * 28)


def test_damage_requires_separate_rng_and_masks_pairing() -> None:
    m = load_model("chest-l2-sobel", ROOT)
    seed = make_seed(28)
    with pytest.raises(ValueError):
        list(m.run(seed, 5, np.random.default_rng(0), damage_at=[3]))
    h1 = list(m.run(seed, 6, np.random.default_rng(0), [3], np.random.default_rng(7)))
    h2 = list(m.run(seed, 6, np.random.default_rng(0), [3], np.random.default_rng(7)))
    assert all(np.array_equal(p, q) for p, q in zip(h1, h2, strict=True))


def test_image_helpers() -> None:
    t = load_target(ROOT / "assets/nca/targets/chest.png", 28)
    assert t.shape == (4, 28, 28) and t.dtype == np.float32
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert rgba_to_rgb(t).shape == (3, 28, 28)
    s = make_seed(28, 16, padding=3)
    assert s.shape == (16, 34, 34)
    assert s[3:, 17, 17].tolist() == [1.0] * 13 and s[:3].sum() == 0
    assert pad(t, 2).shape == (4, 32, 32)
    mask = circle_mask(28, np.random.default_rng(0))
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert 0 < mask.sum() < 28 * 28 * 0.5
    assert np.array_equal(mask, circle_mask(28, np.random.default_rng(0)))


def _mse_after(model_id: str, steps: int = 200) -> float:
    m = load_model(model_id, ROOT)
    entry = next(e for e in MANIFEST if e["id"] == model_id)
    target = load_target(ROOT / entry["target_image"], 28)
    rng = np.random.default_rng(0)
    *_, final = m.run(make_seed(28), steps, rng)
    return float(((final[:4] - target) ** 2).mean())


@pytest.mark.slow
def test_convergence_regression() -> None:
    ids = [m["id"] for m in MANIFEST if m["filter"] in ("sobel", "scharr") and m["loss"] != "hinge"]
    assert len(ids) == 13
    for i in ids:
        assert _mse_after(i) < 0.005, i


@pytest.mark.slow
def test_isotropic_ablation_sanity() -> None:
    ids = [m["id"] for m in MANIFEST if m["filter"] in ("gaussian", "laplacian", "mean")]
    assert len(ids) == 9
    for i in ids:
        assert _mse_after(i) > 0.005, i
