import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cellauto.nca.checkpoints import find_repo_root, load_manifest, load_model  # noqa: E402
from cellauto.nca.image import make_seed  # noqa: E402
from cellauto.nca.spec import NCAConfig  # noqa: E402
from cellauto.nca.training.losses import LOSSES, get_loss  # noqa: E402
from cellauto.nca.training.model import TorchNCA  # noqa: E402
from cellauto.nca.training.train import resolve_config, train  # noqa: E402

pytestmark = pytest.mark.torch

ROOT = find_repo_root(Path(__file__))
MANIFEST = load_manifest(ROOT)


def _config(entry: dict) -> NCAConfig:
    return NCAConfig(entry["n_channels"], entry["hidden_channels"], entry["filter"],
                     entry["fire_rate"])  # fmt: skip


@pytest.mark.parametrize("entry", MANIFEST, ids=[m["id"] for m in MANIFEST])
def test_all_checkpoints_load_strictly(entry: dict) -> None:
    model = TorchNCA.load(ROOT / entry["torch_checkpoint"], _config(entry))
    assert set(model.state_dict()) == {
        "update_module.0.weight", "update_module.0.bias", "update_module.2.weight",
    }  # the kernel buffer must not be persisted  # fmt: skip
    w = model.numpy_weights()
    with np.load(ROOT / entry["weights"]) as z:
        for k in ("w1", "b1", "w2"):
            assert np.array_equal(w[k], z[k])
    npz_model = TorchNCA.load(ROOT / entry["weights"], _config(entry))
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), npz_model.state_dict().values(), strict=True)
    )


def test_torch_matches_numpy_ten_steps() -> None:
    entry = next(m for m in MANIFEST if m["id"] == "chest-l2-sobel")
    cfg = NCAConfig(entry["n_channels"], entry["hidden_channels"], entry["filter"], 1.0)
    tmodel = TorchNCA.load(ROOT / entry["torch_checkpoint"], cfg)
    nmodel = load_model("chest-l2-sobel", ROOT)
    x_np = make_seed(28)
    x_t = torch.from_numpy(x_np)[None]
    with torch.no_grad():
        for _ in range(10):
            x_t = tmodel(x_t)
            nxt = nmodel.step(x_np, fire_rate=1.0)
            assert isinstance(nxt, np.ndarray)
            x_np = nxt
    assert np.allclose(x_t[0].numpy(), x_np, atol=1e-4)


@pytest.mark.parametrize("name", ["sobel", "scharr", "gaussian", "laplacian", "mean"])
def test_kernel_matches_original_construction(name: str) -> None:
    f = {
        "sobel": ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 8.0),
        "scharr": ([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 16.0),
        "gaussian": ([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 16.0),
        "laplacian": ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 8.0),
        "mean": ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 9.0),
    }[name]
    filt = torch.tensor(f[0])
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    original = torch.stack([identity, filt / f[1], filt.t() / f[1]], dim=0)
    original = original.repeat((16, 1, 1))[:, None, ...]
    assert torch.equal(TorchNCA(NCAConfig(filter=name)).kernel, original)


def test_fire_rate_is_honoured_and_generator_controls_it() -> None:
    entry = next(m for m in MANIFEST if m["id"] == "chest-l2-sobel")
    x = torch.from_numpy(make_seed(28))[None]
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(3)
        m = TorchNCA.load(ROOT / entry["torch_checkpoint"], NCAConfig(fire_rate=0.3), generator=g)
        with torch.no_grad():
            outs.append(m(m(x)))
    assert torch.equal(outs[0], outs[1])
    m0 = TorchNCA.load(ROOT / entry["torch_checkpoint"], NCAConfig(fire_rate=1e-9))
    with torch.no_grad():
        assert torch.equal(m0(x), x)  # (almost) nothing fires, so the state is unchanged


def test_losses() -> None:
    assert set(LOSSES) == {"l1", "l2", "manhattan", "hinge"}
    target = torch.zeros(2, 4, 3, 3)
    cs = torch.ones(2, 16, 3, 3)
    assert get_loss("L2")(target, cs)[1].item() == 1.0
    assert get_loss("Manhattan")(target, cs)[0].tolist() == [36.0, 36.0]
    assert get_loss("l1")(target, cs)[1].item() == 1.0
    assert get_loss("HINGE")(target, cs)[1].item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        get_loss("mse")


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        resolve_config({"id": "x"})
    with pytest.raises(ValueError):
        resolve_config({"id": "x", "target_path": "t.png", "bogus": 1})
    with pytest.raises(ValueError):
        resolve_config({"id": "x", "target_path": "t.png", "loss": "mse"})
    cfg = resolve_config({"id": "x", "target_path": "t.png", "loss": "L2"})
    assert cfg["fire_rate"] == 0.5 and cfg["device"] is None and cfg["target"] == "t"


def _smoke_cfg(tmp_path: Path, **kw) -> dict:
    return {
        "id": "smoke", "target_path": str(ROOT / "assets/nca/targets/chest.png"),
        "device": "cpu", "iterations": 2, "pool_size": 8, "batch_size": 2, "damage": True,
        "out_dir": str(tmp_path / "models"), "runs_dir": str(tmp_path / "runs"),
        "animation_dir": str(tmp_path / "gifs"), **kw,
    }  # fmt: skip


def test_smoke_train_writes_checkpoint_npz_and_logs(tmp_path: Path) -> None:
    pt = train(_smoke_cfg(tmp_path))
    assert pt == tmp_path / "models" / "smoke.pt"
    assert pt.is_file() and (tmp_path / "models" / "smoke.npz").is_file()
    assert (tmp_path / "gifs" / "smoke.gif").is_file()
    rows = (tmp_path / "runs" / "smoke" / "losses.csv").read_text().splitlines()
    assert rows[0] == "iteration,loss" and len(rows) == 3
    meta = json.loads((tmp_path / "runs" / "smoke" / "config.json").read_text())
    assert meta["config"]["seed"] == 0 and "torch" in meta["meta"]
    model = TorchNCA.load(pt)  # strict load of what we just wrote
    assert model.numpy_weights()["w1"].shape == (128, 48)


def test_training_is_seeded(tmp_path: Path) -> None:
    a = train(_smoke_cfg(tmp_path / "a", seed=5, animation=False))
    b = train(_smoke_cfg(tmp_path / "b", seed=5, animation=False))
    c = train(_smoke_cfg(tmp_path / "c", seed=6, animation=False))
    sa, sb, sc = (torch.load(p, weights_only=True) for p in (a, b, c))
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def test_optional_flags_and_padding_and_no_pool(tmp_path: Path) -> None:
    cfg = _smoke_cfg(tmp_path, grad_norm=True, lr_schedule={"milestones": [1], "gamma": 0.5},
                     padding=4, batch_size=1, animation=False)  # fmt: skip
    pt = train(cfg)  # batch_size 1 also exercises the "fewer than 3 samples to damage" path
    assert pt.is_file()
    pt2 = train(_smoke_cfg(tmp_path / "np", use_pool=False, animation=False))
    assert pt2.is_file()


def test_manifest_is_updated_when_present(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(ROOT / "models/nca/manifest.json", models / "manifest.json")
    train(_smoke_cfg(tmp_path, animation=False))
    entries = json.loads((models / "manifest.json").read_text())
    assert [e["id"] for e in entries] == sorted(e["id"] for e in entries)
    smoke = next(e for e in entries if e["id"] == "smoke")
    assert smoke["trained_with_damage"] is True and smoke["animation"] is None
    assert len(entries) == 26
