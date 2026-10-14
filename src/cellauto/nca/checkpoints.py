"""Locate and load the repo's trained NCA checkpoints (repo data, not shipped in the wheel)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cellauto.nca.numpy_model import NumpyNCA
from cellauto.nca.spec import NCAConfig

MANIFEST = Path("models") / "nca" / "manifest.json"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk up from ``start`` (default: CWD, then this file) to the dir holding the manifest."""
    candidates = [Path(start) if start is not None else Path.cwd()]
    if start is None:
        candidates.append(Path(__file__).resolve())
    for origin in candidates:
        for d in [origin, *origin.parents]:
            if (d / MANIFEST).is_file():
                return d
    raise FileNotFoundError(
        f"could not find {MANIFEST} above {candidates[0]}; run from inside the repository "
        "checkout (the trained models are repo data and are not shipped in the wheel)"
    )


def load_manifest(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = find_repo_root(root)
    entries: list[dict[str, Any]] = json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    return entries


def load_model(model_id: str, root: str | Path | None = None) -> NumpyNCA:
    base = find_repo_root(root)
    for entry in load_manifest(base):
        if entry["id"] == model_id:
            config = NCAConfig(
                n_channels=entry["n_channels"],
                hidden_channels=entry["hidden_channels"],
                filter=entry["filter"],
                fire_rate=entry["fire_rate"],
            )
            return NumpyNCA.from_npz(base / entry["weights"], config)
    raise KeyError(f"unknown NCA model id {model_id!r}")


def upsert_manifest_entry(manifest_path: str | Path, entry: dict[str, Any]) -> None:
    """Add ``entry`` to the manifest, replacing any entry with the same id; keeps it sorted."""
    path = Path(manifest_path)
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    entries = [e for e in entries if e["id"] != entry["id"]] + [entry]
    entries.sort(key=lambda e: e["id"])
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
