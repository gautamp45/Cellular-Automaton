"""JSON Schema generation for the contract (the frontend derives its TS types from these)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from cellauto.contract import models as m

# schema file stem -> type of the top-level JSON value of a dataset. Later phases append here.
TOP_LEVEL: dict[str, Any] = {
    "index": m.Index,
    "engines": list[m.EngineSpec],
    "elementary_rules": list[m.ElementaryRule],
    "lifelike_rules": list[m.LifeLikeRule],
    "lifelike_patterns": list[m.Pattern],
    "lenia_presets": list[m.LeniaPreset],
    "nca_models": list[m.NcaModel],
    "nca_weights": m.NcaWeights,
    "binary_table_header": m.BinaryTableHeader,
    "content_index": list[m.ContentPage],
    "fixture": m.Fixture,
    "eca_atlas": m.EcaAtlas,
    "lifelike_atlas_meta": m.LifelikeAtlasMeta,
    "relu_examples": m.ReluExamples,
    "inference_study": m.InferenceStudy,
}


def schema_path(name: str) -> str:
    """Path of the schema file inside the data root."""
    return f"schema/{name}.schema.json"


def adapter(name: str) -> TypeAdapter[Any]:
    return TypeAdapter(TOP_LEVEL[name])


def json_schemas() -> dict[str, dict[str, Any]]:
    """``{name: JSON Schema}`` for every top-level model (alias names, e.g. ``"lambda"``)."""
    return {name: adapter(name).json_schema(by_alias=True) for name in sorted(TOP_LEVEL)}


def dumps(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, UTF-8, trailing newline."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def write_schemas(out_dir: str | Path) -> list[Path]:
    """Write ``schema/<name>.schema.json`` for every top-level model under ``out_dir``."""
    root = Path(out_dir)
    written = []
    for name, schema in json_schemas().items():
        path = root / schema_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(schema), encoding="utf-8")
        written.append(path)
    return written
