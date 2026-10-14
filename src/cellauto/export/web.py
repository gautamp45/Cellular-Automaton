"""Build, write and verify the static data backend (``web/public/data``).

``build_export`` is pure: it returns ``{relative path: bytes}`` for every file, including
``index.json`` (which lists the sha256 and size of every other file). ``export_web`` writes that
tree; ``check_export`` compares it with what is on disk. Later phases register more builders in
``_BUILDERS``; the index is always computed last, from the files themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cellauto import __version__
from cellauto.analysis import elementary as eca
from cellauto.analysis import lifelike as lla
from cellauto.catalog import elementary_rules, lenia_presets, lifelike_rules, parse_rle, patterns
from cellauto.contract.models import DatasetRef, Index, NcaWeights
from cellauto.contract.schema import adapter, dumps, json_schemas, schema_path
from cellauto.engines.elementary import rule_table
from cellauto.engines.lifelike import LifeRule
from cellauto.export.binary import from_tensor, to_tensor
from cellauto.export.content import load_pages
from cellauto.export.engines import engine_specs
from cellauto.export.fixtures import all_fixtures
from cellauto.inference import relu_compiler
from cellauto.nca.checkpoints import find_repo_root, load_manifest
from cellauto.nca.spec import ALIVE_THRESHOLD, ALPHA_CHANNEL, is_isotropic, perception_kernels

PRODUCED_BY = "cellauto export-web"
# Float-heavy fixtures can differ in the last bits across numpy/BLAS builds, so `--check` compares
# their tensors within the fixture's own tolerance instead of byte-for-byte.
TOLERANT_PATHS = frozenset(
    {
        "fixtures/lenia.json",
        "fixtures/nca.json",
        "fixtures/lifelike_math.json",
        "fixtures/inference.json",
    }
)


@dataclass
class _Entry:
    data: bytes
    id: str
    title: str
    description: str
    produced_by: str = PRODUCED_BY
    schema: str | None = None  # key into contract.schema.TOP_LEVEL
    format_version: int = 1


class _Builder:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.entries: dict[str, _Entry] = {}

    def _put(self, path: str, entry: _Entry) -> None:
        if path in self.entries:
            raise ValueError(f"duplicate export path {path}")
        self.entries[path] = entry

    def add_json(
        self,
        path: str,
        id_: str,
        schema: str,
        value: Any,
        title: str,
        description: str,
        produced_by: str = PRODUCED_BY,
    ) -> None:
        """Validate ``value`` against the contract model, then store canonical JSON."""
        ad = adapter(schema)
        validated = ad.validate_python(value)
        data = dumps(ad.dump_python(validated, mode="json", by_alias=True)).encode("utf-8")
        self._put(path, _Entry(data, id_, title, description, produced_by, schema=schema))

    def add_file(self, path: str, id_: str, data: bytes, title: str, description: str) -> None:
        self._put(path, _Entry(data, id_, title, description))


# --- builders ---------------------------------------------------------------------------------


def _engines(b: _Builder) -> None:
    b.add_json(
        "engines.json", "engines", "engines", engine_specs(),
        "Engine parameter specs", "Parameters, ranges and defaults for building UI controls.",
    )  # fmt: skip


def _elementary(b: _Builder) -> None:
    catalog = {r["rule"]: r for r in elementary_rules()}
    reversible = set(eca.reversible_rules())
    rules = []
    for n in range(256):
        c = catalog.get(n)
        rules.append(
            {
                "rule": n,
                "name": c.get("name") if c else None,
                "description": c["description"] if c else f"Elementary rule {n}.",
                "wolfram_class": c.get("wolfram_class") if c else None,
                "table": rule_table(n).tolist(),
                "equivalence_class": eca.orbit(n),
                "lambda": eca.lambda_(n),
                "surjective": eca.is_surjective(n),
                "reversible": n in reversible,
            }
        )
    b.add_json(
        "elementary/rules.json", "elementary_rules", "elementary_rules", rules,
        "Elementary rules", "All 256 elementary rules with lookup tables, symmetry class, "
        "lambda, surjectivity, reversibility and catalog annotations.",
    )  # fmt: skip


def _eca_atlas(b: _Builder) -> None:
    path = b.repo_root / "results" / "eca_atlas.json"
    if not path.is_file():
        return
    b.add_json(
        "elementary/atlas.json", "eca_atlas", "eca_atlas", json.loads(path.read_text())["atlas"],
        "Elementary behaviour atlas",
        "Behaviour features, PCA embedding and clusters for all 256 rules.",
        produced_by="cellauto analyze eca-atlas",
    )  # fmt: skip


def _lifelike(b: _Builder) -> None:
    results = b.repo_root / "results" / "lifelike_atlas" / "summary.json"
    simulated = json.loads(results.read_text())["mean_field"] if results.is_file() else {}
    rules = []
    for r in lifelike_rules():
        parsed = LifeRule.parse(r["rulestring"])
        entry: dict[str, Any] = {
            "id": r["id"],
            "name": r["name"],
            "rulestring": parsed.rulestring,
            "code": parsed.code,
            "table": parsed.table().tolist(),
            "description": r["description"],
            "tags": r["tags"],
            "dual": lla.dual(parsed).rulestring,
            "self_dual": lla.is_self_dual(parsed),
            "lambda": lla.lambda_(parsed),
        }
        if r["id"] in simulated:
            sim = simulated[r["id"]]
            entry["mean_field"] = {
                "fixed_points": [
                    {"rho": round(fp.rho, 6), "derivative": round(fp.derivative, 6),
                     "stability": fp.stability}
                    for fp in lla.fixed_points(parsed)
                ],
                "simulated_density": sim["simulated_density"],
                "correlation_gap": sim["correlation_gap"],
            }  # fmt: skip
        rules.append(entry)
    b.add_json(
        "lifelike/rules.json", "lifelike_rules", "lifelike_rules", rules,
        "Life-like rules", "Named B/S rules: 18-bit code, lookup table, dual rule, lambda and "
        "mean-field analysis.",
    )  # fmt: skip
    pats = []
    for p in patterns():
        grid = parse_rle(p["rle"])
        pats.append(
            {
                "id": p["id"],
                "name": p["name"],
                "category": p["category"],
                "rule": p["rule"],
                "rle": p["rle"],
                "width": int(grid.shape[1]),
                "height": int(grid.shape[0]),
                "cells": [[int(r), int(c)] for r, c in np.argwhere(grid)],
                "period": p.get("period"),
                "velocity": p.get("velocity"),
                "description": p["description"],
            }
        )
    b.add_json(
        "lifelike/patterns.json", "lifelike_patterns", "lifelike_patterns", pats,
        "Life patterns", "Named patterns as RLE plus decoded live cells.",
    )  # fmt: skip


def _lifelike_atlas(b: _Builder) -> None:
    root = b.repo_root / "results" / "lifelike_atlas"
    if not (root / "summary.json").is_file():
        return
    by = "cellauto analyze lifelike-atlas"
    header = json.loads((root / "atlas.header.json").read_text())
    b.add_json(
        "lifelike/atlas.header.json", "lifelike_atlas_header", "binary_table_header", header,
        "Rule-space atlas: table header", "Column layout of lifelike/atlas.bin (cabin format).",
        produced_by=by,
    )  # fmt: skip
    b._put(
        "lifelike/atlas.bin",
        _Entry(
            (root / "atlas.bin").read_bytes(),
            "lifelike_atlas_bin",
            "Rule-space atlas: table",
            "131,072 B0-free life-like rules x 11 columns, little-endian columns (cabin format).",
            by,
        ),
    )
    b.add_json(
        "lifelike/atlas_summary.json", "lifelike_atlas_meta", "lifelike_atlas_meta",
        json.loads((root / "summary.json").read_text())["summary"],
        "Rule-space atlas: summary", "Protocol, clusters, edge-of-chaos table and catalog rows.",
        produced_by=by,
    )  # fmt: skip


def _inference(b: _Builder) -> None:
    examples = []
    for r in lifelike_rules():
        rule = LifeRule.parse(r["rulestring"])
        net = relu_compiler.compile_rule(rule)
        examples.append(
            {
                "id": r["id"],
                "rulestring": rule.rulestring,
                "code": rule.code,
                "hidden_units": net.hidden_units,
                "net": {
                    "centre_weight": net.centre_weight,
                    "biases": net.biases.tolist(),
                    "out_weights": net.out_weights.tolist(),
                    "out_bias": net.out_bias,
                },
            }
        )
    histogram = relu_compiler.hidden_unit_histogram()
    b.add_json(
        "inference/relu_examples.json", "relu_examples", "relu_examples",
        {
            "format_version": 1,
            "construction": relu_compiler.CONSTRUCTION,
            "examples": examples,
            "histogram": histogram,
            "mean_hidden_units": sum(u * n for u, n in enumerate(histogram)) / sum(histogram),
        },
        "ReLU-compiled rules",
        "Every life-like rule as an exact Conv3x3 -> ReLU -> linear network, plus the "
        "distribution of hidden units over all 2^18 rules.",
    )  # fmt: skip


def _inference_study(b: _Builder) -> None:
    path = b.repo_root / "results" / "inference_study.json"
    if not path.is_file():
        return
    b.add_json(
        "inference/study.json", "inference_study", "inference_study",
        json.loads(path.read_text())["study"],
        "Rule-inference study",
        "Gradient descent vs exhaustive search: success versus rollout length and noise.",
        produced_by="cellauto analyze inference-study",
    )  # fmt: skip


def _lenia(b: _Builder) -> None:
    presets = []
    for p in lenia_presets():
        presets.append(
            {
                "id": p["id"],
                "name": p["name"],
                "R": p["params"]["R"],
                "T": float(p["params"]["T"]),
                "mu": p["params"]["mu"],
                "sigma": p["params"]["sigma"],
                "peaks": p["params"]["peaks"],
                "growth": p["growth"],
                "kernel_core": p["kernel_core"],
                "cells_rle": p["cells_rle"],
                "width": p["width"],
                "height": p["height"],
                "source": p["source"],
            }
        )
    b.add_json(
        "lenia/presets.json", "lenia_presets", "lenia_presets", presets,
        "Lenia presets", "Creatures from Chakazul/Lenia (MIT) plus Life expressed as a Lenia.",
    )  # fmt: skip


def _nca(b: _Builder) -> None:
    root = b.repo_root
    models = []
    seen_targets: set[str] = set()
    for m in load_manifest(root):
        mid = m["id"]
        has_gif = bool(m.get("animation")) and (root / m["animation"]).is_file()
        models.append(
            {
                "id": mid,
                "target": m["target"],
                "loss": m["loss"],
                "filter": m["filter"],
                "trained_with_damage": m["trained_with_damage"],
                "isotropic": is_isotropic(m["filter"]),
                "n_channels": m["n_channels"],
                "hidden_channels": m["hidden_channels"],
                "fire_rate": m["fire_rate"],
                "grid_size": m["grid_size"],
                "padding": m["padding"],
                "weights_path": f"nca/weights/{mid}.json",
                "target_path": f"nca/targets/{m['target']}.png",
                "animation_path": f"nca/animations/{mid}.gif" if has_gif else None,
            }
        )
        with np.load(root / m["weights"]) as z:
            weights = NcaWeights(
                format_version=1,
                id=mid,
                n_channels=m["n_channels"],
                hidden_channels=m["hidden_channels"],
                fire_rate=m["fire_rate"],
                alive_threshold=ALIVE_THRESHOLD,
                alpha_channel=ALPHA_CHANNEL,
                grid_size=m["grid_size"],
                padding=m["padding"],
                filter=m["filter"],
                kernels=perception_kernels(m["filter"]).tolist(),
                perception_layout="index = 3*channel + k, k in (identity, dx, dy); depthwise "
                "3x3 cross-correlation with zero padding",
                w1=to_tensor(z["w1"]),
                b1=to_tensor(z["b1"]),
                w2=to_tensor(z["w2"]),
            )
        b.add_json(
            f"nca/weights/{mid}.json", f"nca_weights/{mid}", "nca_weights", weights,
            f"NCA weights: {mid}", "Float32 weights and scaled perception kernels (base64).",
        )  # fmt: skip
        if has_gif:
            b.add_file(
                f"nca/animations/{mid}.gif", f"nca_animation/{mid}",
                (root / m["animation"]).read_bytes(),
                f"NCA growth animation: {mid}", "GIF of the model growing from a seed.",
            )  # fmt: skip
        if m["target"] not in seen_targets:
            seen_targets.add(m["target"])
            b.add_file(
                f"nca/targets/{m['target']}.png", f"nca_target/{m['target']}",
                (root / m["target_image"]).read_bytes(),
                f"NCA target: {m['target']}", "28x28 RGBA target image.",
            )  # fmt: skip
    b.add_json(
        "nca/models.json", "nca_models", "nca_models", models,
        "NCA models", "Trained neural CA checkpoints with their training configuration.",
    )  # fmt: skip


def _fixtures(b: _Builder) -> None:
    for name, fx in all_fixtures(b.repo_root).items():
        b.add_json(
            f"fixtures/{name}.json", f"fixture/{name}", "fixture", fx,
            f"Golden fixture: {name}", "Inputs and expected outputs for browser parity tests.",
        )  # fmt: skip


def _content(b: _Builder) -> None:
    pages = load_pages(b.repo_root / "content")
    b.add_json(
        "content/index.json", "content_index", "content_index", [p for p, _ in pages],
        "Content index", "Explainer pages in display order.",
    )  # fmt: skip
    for page, raw in pages:
        b.add_file(
            page.path, f"content/{page.slug}", raw, f"Content: {page.title}",
            "Markdown with a key: value front-matter block.",
        )  # fmt: skip


def _schemas(b: _Builder) -> None:
    for name, schema in json_schemas().items():
        b._put(
            schema_path(name),
            _Entry(
                dumps(schema).encode("utf-8"),
                f"schema/{name}",
                f"JSON Schema: {name}",
                "Generated from the Pydantic contract.",
            ),
        )


_BUILDERS: list[Callable[[_Builder], None]] = [
    _engines,
    _elementary,
    _eca_atlas,
    _lifelike,
    _lifelike_atlas,
    _inference,
    _inference_study,
    _lenia,
    _nca,
    _fixtures,
    _content,
    _schemas,
]


# --- public API -------------------------------------------------------------------------------


def build_export(repo_root: Path | None = None) -> dict[str, bytes]:
    """The complete output tree as ``{relative posix path: bytes}``."""
    root = find_repo_root(repo_root) if repo_root is None else Path(repo_root)
    b = _Builder(root)
    for build in _BUILDERS:
        build(b)
    refs = []
    for path in sorted(b.entries):
        e = b.entries[path]
        refs.append(
            DatasetRef(
                id=e.id,
                path=path,
                schema_path=schema_path(e.schema) if e.schema else None,
                format_version=e.format_version,
                sha256=hashlib.sha256(e.data).hexdigest(),
                bytes=len(e.data),
                title=e.title,
                description=e.description,
                produced_by=e.produced_by,
            )
        )
    index = Index(format_version=1, package_version=__version__, datasets=refs)
    files = {p: e.data for p, e in b.entries.items()}
    files["index.json"] = dumps(index.model_dump(mode="json", by_alias=True)).encode("utf-8")
    return dict(sorted(files.items()))


def _disk_files(out_dir: Path) -> dict[str, bytes]:
    if not out_dir.is_dir():
        return {}
    return {
        p.relative_to(out_dir).as_posix(): p.read_bytes()
        for p in sorted(out_dir.rglob("*"))
        if p.is_file()
    }


def export_web(out_dir: str | Path, repo_root: str | Path | None = None) -> Index:
    """Write the export to ``out_dir``, removing files that are no longer part of it."""
    out = Path(out_dir)
    files = build_export(None if repo_root is None else Path(repo_root))
    if out.is_dir() and any(out.iterdir()) and not (out / "index.json").is_file():
        raise ValueError(f"{out} is not empty and has no index.json; refusing to overwrite it")
    for path, data in files.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
    for stale in set(_disk_files(out)) - set(files):
        (out / stale).unlink()
    for d in sorted((p for p in out.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return Index.model_validate_json(files["index.json"])


def _tensor_like(x: Any) -> bool:
    return isinstance(x, dict) and set(x) == {"shape", "dtype", "b64"}


def _close(a: Any, b: Any, atol: float) -> bool:
    if _tensor_like(a) and _tensor_like(b):
        if a["shape"] != b["shape"]:
            return False
        return bool(np.allclose(from_tensor(a), from_tensor(b), rtol=0.0, atol=atol))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], atol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, atol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= atol
    return bool(a == b)


def _index_without(raw: bytes, paths: frozenset[str]) -> Any:
    doc = json.loads(raw)
    doc["datasets"] = [d for d in doc["datasets"] if d["path"] not in paths]
    return doc


def check_export(out_dir: str | Path, repo_root: str | Path | None = None) -> list[str]:
    """Paths that differ between a fresh export and ``out_dir`` (empty list = up to date).

    Everything is compared byte-for-byte except the float-sensitive fixtures in
    ``TOLERANT_PATHS``, whose tensors are compared within the fixture's tolerance (and whose
    index entries are ignored).
    """
    out = Path(out_dir)
    fresh = build_export(None if repo_root is None else Path(repo_root))
    disk = _disk_files(out)
    diffs: list[str] = []
    for path in sorted(set(fresh) | set(disk)):
        if path not in disk:
            diffs.append(f"missing: {path}")
        elif path not in fresh:
            diffs.append(f"stale: {path}")
        elif path == "index.json":
            if _index_without(fresh[path], TOLERANT_PATHS) != _index_without(
                disk[path], TOLERANT_PATHS
            ):
                diffs.append(f"changed: {path}")
        elif path in TOLERANT_PATHS:
            a, b = json.loads(fresh[path]), json.loads(disk[path])
            if not _close(a, b, float(a["tolerance"]) / 10):
                diffs.append(f"changed: {path}")
        elif fresh[path] != disk[path]:
            diffs.append(f"changed: {path}")
    return diffs
