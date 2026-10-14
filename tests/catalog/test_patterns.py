import numpy as np
import pytest

from cellauto.catalog import (
    elementary_rules,
    get_pattern,
    lifelike_rule,
    lifelike_rules,
    parse_rle,
    patterns,
    place,
    to_rle,
)
from cellauto.engines.lifelike import LifeRule, run, step


def _pop_history(pattern_id: str, size: int, steps: int, at: tuple[int, int] | None = None):
    g = place(get_pattern(pattern_id), (size, size), at)
    return g, run(g, "B3/S23", steps, "dead")


SHAPES = {
    "block": ((2, 2), 4),
    "beehive": ((3, 4), 6),
    "blinker": ((1, 3), 3),
    "toad": ((2, 4), 6),
    "beacon": ((4, 4), 8),
    "pulsar": ((13, 13), 48),
    "glider": ((3, 3), 5),
    "lwss": ((4, 5), 9),
    "r_pentomino": ((3, 3), 5),
    "acorn": ((3, 7), 7),
    "diehard": ((3, 8), 7),
    "gosper_glider_gun": ((9, 36), 36),
}


@pytest.mark.parametrize("pid", sorted(SHAPES))
def test_shape_and_population(pid: str) -> None:
    shape, pop = SHAPES[pid]
    p = get_pattern(pid)
    assert p.shape == shape
    assert p.sum() == pop


@pytest.mark.parametrize(
    ("pid", "period"),
    [("blinker", 2), ("toad", 2), ("beacon", 2), ("pulsar", 3)],
)
def test_oscillators(pid: str, period: int) -> None:
    g, h = _pop_history(pid, 80, period)
    assert np.array_equal(h[period], g)
    assert not np.array_equal(h[1], g)


@pytest.mark.parametrize("pid", ["block", "beehive"])
def test_still_lifes(pid: str) -> None:
    g = place(get_pattern(pid), (80, 80))
    assert np.array_equal(step(g, "B3/S23", "dead"), g)


@pytest.mark.parametrize(("pid", "shift"), [("glider", (1, 1)), ("lwss", (0, -2))])
def test_spaceships(pid: str, shift: tuple[int, int]) -> None:
    g, h = _pop_history(pid, 80, 4)
    assert np.array_equal(h[4], np.roll(g, shift, axis=(0, 1)))


def test_methuselah_populations() -> None:
    _, h = _pop_history("r_pentomino", 80, 130)
    assert h[30].sum() == 27
    assert h[130].sum() == 178
    _, h = _pop_history("acorn", 80, 30)
    assert h[30].sum() == 57


def test_diehard_dies() -> None:
    _, h = _pop_history("diehard", 120, 130)
    assert h[129].sum() == 2
    assert h[130].sum() == 0


def test_gosper_gun_population() -> None:
    _, h = _pop_history("gosper_glider_gun", 120, 120, at=(2, 2))
    assert [int(h[t].sum()) for t in (0, 30, 60, 90, 120)] == [36, 41, 46, 51, 56]


def test_rle_roundtrip_all_patterns() -> None:
    for entry in patterns():
        p = parse_rle(entry["rle"])
        assert np.array_equal(parse_rle(to_rle(p)), p), entry["id"]


def test_rle_header_and_comments() -> None:
    text = "#C a glider\n#N Glider\nx = 3, y = 3, rule = B3/S23\nbob$2bo$3o!\n"
    assert np.array_equal(parse_rle(text), parse_rle("bo$2bo$3o!"))
    padded = parse_rle("x = 5, y = 4\n2o$2o!")
    assert padded.shape == (4, 5)


def test_rle_rejects() -> None:
    with pytest.raises(ValueError):
        parse_rle("3x!")
    with pytest.raises(ValueError):
        parse_rle("3o")
    with pytest.raises(ValueError):
        place(get_pattern("pulsar"), (5, 5))


def test_place_at_and_centre() -> None:
    p = get_pattern("block")
    assert place(p, (6, 6))[2:4, 2:4].all()
    assert place(p, (6, 6), (0, 0))[:2, :2].all()
    with pytest.raises(ValueError):
        place(p, (6, 6), (5, 5))


def test_lifelike_catalog() -> None:
    rules = lifelike_rules()
    assert len(rules) == 27
    assert len({r["id"] for r in rules}) == 27
    for r in rules:
        parsed = LifeRule.parse(r["rulestring"])
        assert parsed.rulestring == r["rulestring"]
        assert LifeRule.from_code(parsed.code) == parsed
    assert lifelike_rule("life") == LifeRule.parse("B3/S23")
    anti = next(r for r in rules if r["id"] == "antilife")
    assert anti["tags"] == ["b0", "dual-of-life"]
    with pytest.raises(KeyError):
        lifelike_rule("nope")


def test_elementary_catalog() -> None:
    rules = elementary_rules()
    assert {r["rule"]: r["wolfram_class"] for r in rules} == {
        0: 1, 8: 1, 32: 1, 40: 1, 1: 2, 4: 2, 108: 2, 184: 2,
        18: 3, 22: 3, 30: 3, 45: 3, 90: 3, 126: 3, 150: 3,
        54: 4, 106: 4, 110: 4,
    }  # fmt: skip
