"""Static catalogs shipped with the package (``catalog/data/*.json``)."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from cellauto.engines.lifelike import LifeRule


@cache
def _read(name: str) -> str:
    return (files("cellauto.catalog") / "data" / name).read_text(encoding="utf-8")


def _load(name: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(_read(name))
    return data


def elementary_rules() -> list[dict[str, Any]]:
    return _load("elementary_rules.json")


def lifelike_rules() -> list[dict[str, Any]]:
    return _load("lifelike_rules.json")


def patterns() -> list[dict[str, Any]]:
    return _load("patterns.json")


def lifelike_rule(rule_id: str) -> LifeRule:
    for entry in lifelike_rules():
        if entry["id"] == rule_id:
            return LifeRule.parse(entry["rulestring"])
    raise KeyError(f"unknown life-like rule id {rule_id!r}")


def lenia_presets() -> list[dict[str, Any]]:
    """Lenia presets (credit and licence are in the file's ``license`` field)."""
    doc = json.loads(_read("lenia_presets.json"))
    presets: list[dict[str, Any]] = doc["presets"]
    return presets
