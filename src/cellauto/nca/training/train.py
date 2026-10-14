"""Train a Growing NCA (sample pool, worst-sample reseed, optional damage).

The loop is the 2023 one; see ``docs/plan/P08-nca-training.md`` for the list of deliberate
differences. Run from the repository root: all config paths are relative to the CWD.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cellauto.nca.checkpoints import upsert_manifest_entry
from cellauto.nca.image import circle_mask, load_target, make_seed, pad
from cellauto.nca.spec import FILTERS, NCAConfig
from cellauto.nca.training.losses import get_loss
from cellauto.nca.training.model import TorchNCA

DEFAULTS: dict[str, Any] = {
    "device": None,  # null = auto: cuda -> mps -> cpu
    "iterations": 1000,
    "learning_rate": 2e-3,
    "img_size": 28,
    "padding": 0,
    "n_channels": 16,
    "hidden_channels": 128,
    "batch_size": 8,
    "pool_size": 1024,
    "filter": "sobel",
    "loss": "l2",
    "damage": False,
    "fire_rate": 0.5,
    "seed": 0,
    "use_pool": True,  # False = "growing" regime: every iteration restarts from the seed
    "grad_norm": False,  # Distill's per-parameter gradient normalisation g / (|g| + 1e-8)
    "lr_schedule": None,  # {"milestones": [...], "gamma": 0.1}
    "out_dir": "models/nca",
    "runs_dir": "runs",
    "animation_dir": "assets/nca/animations",
    "update_manifest": True,
    "animation": True,  # also write a 200-step growth GIF
}
REQUIRED = ("id", "target_path")


def setup_device(device: str | None = None) -> torch.device:
    """``None`` = auto-detect: cuda, then mps, then cpu."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED if k not in config]
    if missing:
        raise ValueError(f"config is missing required keys {missing}")
    unknown = sorted(set(config) - set(DEFAULTS) - set(REQUIRED) - {"target"})
    if unknown:
        raise ValueError(f"unknown config keys {unknown}")
    cfg = {**DEFAULTS, **config}
    if cfg["filter"] not in FILTERS:
        raise ValueError(f"unknown filter {cfg['filter']!r}; expected one of {sorted(FILTERS)}")
    get_loss(cfg["loss"])  # validates
    if cfg["batch_size"] > cfg["pool_size"]:
        raise ValueError("batch_size must not exceed pool_size")
    cfg.setdefault("target", Path(cfg["target_path"]).stem)
    return cfg


def _grad_normalise(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = p.grad / (p.grad.norm() + 1e-8)


def train(config: dict[str, Any]) -> Path:
    """Train and return the path of the saved ``<out_dir>/<id>.pt``."""
    cfg = resolve_config(config)
    model_id: str = cfg["id"]
    device = setup_device(cfg["device"])
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    started = time.time()

    # target and seed (padded); batch of targets
    size, padding, n_channels = cfg["img_size"], cfg["padding"], cfg["n_channels"]
    target = torch.from_numpy(pad(load_target(cfg["target_path"], size), padding))[None].to(device)
    target_batch = target.repeat(cfg["batch_size"], 1, 1, 1)
    seed_state = torch.from_numpy(make_seed(size, n_channels, padding))[None].to(device)
    spatial = seed_state.shape[-1]  # padded size, also used for damage masks (N7)
    pool = seed_state.clone().repeat(cfg["pool_size"], 1, 1, 1)

    nca_config = NCAConfig(n_channels, cfg["hidden_channels"], cfg["filter"], cfg["fire_rate"])
    model = TorchNCA(nca_config, device=device, generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = None
    if cfg["lr_schedule"]:
        sched = cfg["lr_schedule"]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(sched["milestones"]), gamma=float(sched["gamma"])
        )
    loss_fn = get_loss(cfg["loss"])

    losses: list[float] = []
    for _ in range(cfg["iterations"]):
        if cfg["use_pool"]:
            batch_idxs = rng.choice(cfg["pool_size"], cfg["batch_size"], replace=False).tolist()
            cs = pool[batch_idxs]
        else:
            batch_idxs = []
            cs = seed_state.repeat(cfg["batch_size"], 1, 1, 1)

        for _ in range(int(rng.integers(64, 96))):
            cs = model(cs)

        loss_batch, loss = loss_fn(target_batch, cs)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if cfg["grad_norm"]:
            _grad_normalise(model)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if cfg["use_pool"]:
            argmax_batch = loss_batch.argmax().item()
            argmax_pool = batch_idxs[argmax_batch]
            remaining_batch = [i for i in range(cfg["batch_size"]) if i != argmax_batch]
            remaining_pool = [i for i in batch_idxs if i != argmax_pool]
            pool[argmax_pool] = seed_state[0].clone()
            pool[remaining_pool] = cs[remaining_batch].detach()

            if cfg["damage"]:
                # damage the (up to) 3 best samples of the batch
                best_idxs_batch = np.argsort(loss_batch.detach().cpu().numpy())[:3]
                for i in best_idxs_batch:
                    mask = torch.from_numpy(circle_mask(spatial, rng)).to(device)
                    pool[batch_idxs[int(i)]] *= 1.0 - mask

    # outputs: checkpoint (+ torch-free npz), run log, optional manifest entry and animation
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{model_id}.pt"
    torch.save(model.state_dict(), pt_path)
    np.savez(out_dir / f"{model_id}.npz", **model.numpy_weights())

    if cfg["animation"]:
        from cellauto.nca.animate import growth_gif
        from cellauto.nca.numpy_model import NumpyNCA

        growth_gif(
            NumpyNCA(**model.numpy_weights(), config=nca_config),
            Path(cfg["animation_dir"]) / f"{model_id}.gif",
            grid_size=size, padding=padding, seed=seed,
        )  # fmt: skip

    run_dir = Path(cfg["runs_dir"]) / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "losses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "loss"])
        w.writerows(enumerate(losses))
    meta = {
        "runtime_s": round(time.time() - started, 2),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "final_loss": losses[-1] if losses else None,
    }
    (run_dir / "config.json").write_text(
        json.dumps({"config": cfg, "meta": meta}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if cfg["update_manifest"] and (out_dir / "manifest.json").is_file():
        upsert_manifest_entry(out_dir / "manifest.json", _manifest_entry(cfg, out_dir))
    return pt_path


def _manifest_entry(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    model_id = cfg["id"]
    gif = Path(cfg["animation_dir"]) / f"{model_id}.gif"
    return {
        "id": model_id,
        "target": cfg["target"],
        "target_image": Path(cfg["target_path"]).as_posix(),
        "loss": cfg["loss"].lower(),
        "filter": cfg["filter"],
        "trained_with_damage": bool(cfg["damage"]),
        "n_channels": cfg["n_channels"],
        "hidden_channels": cfg["hidden_channels"],
        "fire_rate": cfg["fire_rate"],
        "grid_size": cfg["img_size"],
        "padding": cfg["padding"],
        "weights": (out_dir / f"{model_id}.npz").as_posix(),
        "torch_checkpoint": (out_dir / f"{model_id}.pt").as_posix(),
        "animation": gif.as_posix() if gif.is_file() else None,
        "provenance": (
            f"trained with cellauto nca train (seed {cfg['seed']}, {cfg['iterations']} iterations, "
            f"lr {cfg['learning_rate']}, batch {cfg['batch_size']}, pool {cfg['pool_size']}, "
            f"use_pool {cfg['use_pool']}, grad_norm {cfg['grad_norm']}, "
            f"lr_schedule {cfg['lr_schedule']})"
        ),
    }
