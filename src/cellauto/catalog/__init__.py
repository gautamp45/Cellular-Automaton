"""Static catalogs (rules, patterns) and RLE helpers."""

from cellauto.catalog.patterns import get_pattern
from cellauto.catalog.rle import parse_rle, place, to_rle
from cellauto.catalog.rules import (
    elementary_rules,
    lenia_presets,
    lifelike_rule,
    lifelike_rules,
    patterns,
)

__all__ = [
    "elementary_rules",
    "get_pattern",
    "lenia_presets",
    "lifelike_rule",
    "lifelike_rules",
    "parse_rle",
    "patterns",
    "place",
    "to_rle",
]
