"""Pattern lookup helpers."""

from __future__ import annotations

from cellauto._types import BinaryArray
from cellauto.catalog.rle import parse_rle
from cellauto.catalog.rules import patterns


def get_pattern(pattern_id: str) -> BinaryArray:
    for entry in patterns():
        if entry["id"] == pattern_id:
            return parse_rle(entry["rle"])
    raise KeyError(f"unknown pattern id {pattern_id!r}")
