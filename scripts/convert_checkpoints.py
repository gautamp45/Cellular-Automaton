"""Convert the trusted ``models/nca/*.pt`` checkpoints to torch-free ``.npz`` weights.

    python scripts/convert_checkpoints.py [--manifest]

Writes ``models/nca/<id>.npz`` (``w1 (128, 48)``, ``b1 (128,)``, ``w2 (16, 128)`` float32) and,
with ``--manifest``, ``models/nca/manifest.json``. If torch is importable, the numpy reader is
cross-checked against ``torch.load``.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "nca"
LOSSES = ("hinge", "l1", "l2", "manhattan")
FILTERS = ("sobel", "scharr", "gaussian", "laplacian", "mean")
PROVENANCE = (
    "trained 2023 with neural-cellular-automata/src/train.py "
    "(1000 iterations, lr 2e-3, batch 8, pool 1024)"
)


def load_pt_state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """One-off conversion of TRUSTED repo files only. Never use on untrusted input (pickle)."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *_):  # type: ignore[no-untyped-def]
        return np.lib.stride_tricks.as_strided(
            storage[offset:], shape=size, strides=[s * 4 for s in stride]
        ).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):  # type: ignore[no-untyped-def]
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "OrderedDict":
                return collections.OrderedDict
            return lambda *a, **k: None

        def persistent_load(self, pid):  # type: ignore[no-untyped-def]
            # ('storage', type, key, location, numel)
            return np.frombuffer(z.read(f"{root}/data/{pid[2]}"), dtype="<f4")

    return dict(Unpickler(z.open(f"{root}/data.pkl")).load())


def parse_id(model_id: str) -> dict[str, object]:
    """``chest-l2-sobel-damage`` -> target/loss/filter/damage."""
    parts = model_id.split("-")
    damage = parts[-1] == "damage"
    if damage:
        parts = parts[:-1]
    target, loss, filt = parts
    assert loss in LOSSES and filt in FILTERS, model_id
    return {"target": target, "loss": loss, "filter": filt, "trained_with_damage": damage}


def convert(pt: Path, check_torch: bool) -> dict[str, np.ndarray]:
    sd = load_pt_state_dict(pt)
    out = {
        "w1": sd["update_module.0.weight"][:, :, 0, 0].astype(np.float32),
        "b1": sd["update_module.0.bias"].astype(np.float32),
        "w2": sd["update_module.2.weight"][:, :, 0, 0].astype(np.float32),
    }
    assert out["w1"].shape == (128, 48) and out["b1"].shape == (128,), pt
    assert out["w2"].shape == (16, 128), pt
    if check_torch:
        import torch

        ref = torch.load(pt, map_location="cpu", weights_only=True)
        for key, arr in sd.items():
            np.testing.assert_array_equal(arr, ref[key].numpy(), err_msg=f"{pt}:{key}")
    return out


def manifest_entry(model_id: str) -> dict[str, object]:
    info = parse_id(model_id)
    return {
        "id": model_id,
        "target": info["target"],
        "target_image": f"assets/nca/targets/{info['target']}.png",
        "loss": info["loss"],
        "filter": info["filter"],
        "trained_with_damage": info["trained_with_damage"],
        "n_channels": 16,
        "hidden_channels": 128,
        "fire_rate": 0.5,
        "grid_size": 28,
        "padding": 0,
        "weights": f"models/nca/{model_id}.npz",
        "torch_checkpoint": f"models/nca/{model_id}.pt",
        "animation": f"assets/nca/animations/{model_id}.gif",
        "provenance": PROVENANCE,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="store_true", help="also write models/nca/manifest.json")
    args = ap.parse_args()
    try:
        import torch  # noqa: F401

        check_torch = True
    except ImportError:
        check_torch = False
    ids = []
    for pt in sorted(MODELS.glob("*.pt")):
        weights = convert(pt, check_torch)
        np.savez(pt.with_suffix(".npz"), **weights)
        ids.append(pt.stem)
    print(f"converted {len(ids)} checkpoints (torch cross-check: {check_torch})")
    if args.manifest:
        manifest = [manifest_entry(i) for i in sorted(ids)]
        (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote manifest with {len(manifest)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
