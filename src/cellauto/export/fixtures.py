"""Golden fixtures: inputs and expected outputs the browser implementation must reproduce."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cellauto.analysis import elementary as eca
from cellauto.catalog import get_pattern, lenia_presets, parse_rle, patterns, place
from cellauto.contract.models import Fixture
from cellauto.engines import elementary, lenia, lifelike
from cellauto.export.binary import to_tensor
from cellauto.nca.checkpoints import load_manifest, load_model
from cellauto.nca.image import make_seed


def _rows(grid: np.ndarray) -> list[str]:
    return [elementary.to_bits(r) for r in grid]


def elementary_fixture() -> Fixture:
    cases = []
    for rule in (30, 54, 90, 110, 184):
        for boundary in ("wrap", "dead"):
            for init in ("single", "random"):
                start = (
                    elementary.single_cell(31)
                    if init == "single"
                    else elementary.random_row(31, seed=7)
                )
                rows = elementary.run(start, rule, 15, boundary)
                cases.append(
                    {
                        "rule": rule,
                        "boundary": boundary,
                        "init": init,
                        "initial": elementary.to_bits(start),
                        "rows": _rows(rows),
                    }
                )
    return Fixture(
        engine="elementary",
        description="Space-time diagrams (16 rows each, bit strings, row 0 = initial).",
        tolerance=None,
        cases=cases,
    )


def lifelike_fixture() -> Fixture:
    def case(name: str, rule: str, boundary: str, grid: np.ndarray, steps: int) -> dict[str, Any]:
        r = lifelike.LifeRule.parse(rule)
        final = lifelike.run(grid, r, steps, boundary)[-1]  # type: ignore[arg-type]
        return {
            "name": name,
            "rule": r.rulestring,
            "boundary": boundary,
            "steps": steps,
            "initial": _rows(grid),
            "expected": _rows(final),
        }

    soup = lifelike.random_grid(24, 24, 0.35, seed=0)
    cases = [case("glider", "B3/S23", "wrap", place(get_pattern("glider"), (16, 16)), 8)]
    for rule in ("B3/S23", "B36/S23", "B2/S", "B3678/S34678", "B1357/S1357"):
        cases.append(case(f"soup {rule} wrap", rule, "wrap", soup, 10))
    cases.append(case("soup B3/S23 dead", "B3/S23", "dead", soup, 10))
    return Fixture(
        engine="lifelike",
        description="Final grid after `steps` generations; grids are lists of bit strings.",
        tolerance=None,
        cases=cases,
    )


RLE_EDGE_CASES = {
    "header_pads_to_declared_size": "x = 5, y = 4, rule = B3/S23\n2o$2o!",
    "comments_and_header": "#C a comment\n#N Glider\nx = 3, y = 3, rule = B3/S23\nbo$2bo$3o!",
    "multiline_body_with_blank_rows": "ob$\n2o2$\nb2o!",
}


def rle_fixture() -> Fixture:
    def case(name: str, rle: str) -> dict[str, Any]:
        grid = parse_rle(rle)
        cells = [[int(r), int(c)] for r, c in np.argwhere(grid)]
        return {"name": name, "rle": rle, "width": grid.shape[1], "height": grid.shape[0],
                "cells": cells}  # fmt: skip

    cases = [case(p["id"], p["rle"]) for p in patterns()]
    cases += [case(name, rle) for name, rle in RLE_EDGE_CASES.items()]
    return Fixture(
        engine="rle",
        description="RLE text -> bounding size and live cells as (row, col).",
        tolerance=None,
        cases=cases,
    )


def lenia_fixture() -> Fixture:
    preset = next(p for p in lenia_presets() if p["id"] == "o2u")
    pr = preset["params"]
    params = lenia.LeniaParams(
        R=pr["R"], T=pr["T"], mu=pr["mu"], sigma=pr["sigma"], peaks=tuple(pr["peaks"])
    )
    cells = lenia.decode_lenia_rle(preset["cells_rle"])
    state = np.zeros((64, 64), dtype=np.float32)
    r0, c0 = (64 - cells.shape[0]) // 2, (64 - cells.shape[1]) // 2
    state[r0 : r0 + cells.shape[0], c0 : c0 + cells.shape[1]] = cells
    states = [to_tensor(s).model_dump() for s in lenia.run(state, params, 5)]
    orbium = {
        "name": "orbium",
        "params": {"R": params.R, "T": params.T, "mu": params.mu, "sigma": params.sigma,
                   "peaks": list(params.peaks), "growth": params.growth,
                   "kernel_core": params.kernel_core},
        "origin": [r0, c0],
        "initial": to_tensor(state).model_dump(),
        "steps": [1, 2, 3, 4, 5],
        "states": states,
    }  # fmt: skip
    life_params, _ = lenia.life_as_lenia()
    grid = lifelike.random_grid(16, 16, 0.5, seed=0)
    expected = lifelike.run(grid, "B3/S23", 3, "wrap")[-1]
    life = {
        "name": "life_as_lenia",
        "params": {"growth": life_params.growth, "kernel_core": life_params.kernel_core,
                   "T": life_params.T},
        "exact": True,
        "steps": 3,
        "initial": _rows(grid),
        "expected": _rows(expected),
    }  # fmt: skip
    return Fixture(
        engine="lenia",
        description="Orbium on a 64x64 torus (float32 states, compare with `tolerance`), and "
        "Life as a Lenia on a 16x16 torus (exact).",
        tolerance=1e-5,
        cases=[orbium, life],
    )


def nca_fixture(repo_root: Path) -> Fixture:
    model_id = "chest-l2-sobel"
    entry = next(m for m in load_manifest(repo_root) if m["id"] == model_id)
    model = load_model(model_id, repo_root)
    x = make_seed(entry["grid_size"], entry["n_channels"], entry["padding"])
    wanted = [1, 2, 5, 10]
    states: dict[int, np.ndarray] = {}
    for t in range(1, max(wanted) + 1):
        nxt = model.step(x, fire_rate=1.0)
        assert isinstance(nxt, np.ndarray)
        x = nxt
        if t in wanted:
            states[t] = x
    case = {
        "name": model_id,
        "model_id": model_id,
        "fire_rate": 1.0,
        "steps": wanted,
        "states": [to_tensor(states[t]).model_dump() for t in wanted],
        "hidden_state_step": 1,
        "hidden_activations": to_tensor(model.hidden_activations(states[1])).model_dump(),
    }
    return Fixture(
        engine="nca",
        description="Grown from the seed with fire_rate 1 (no randomness). `states[i]` is the "
        "(16, 28, 28) state after update `steps[i]`; `hidden_activations` is (128, 784) "
        "computed from the state at step `hidden_state_step`.",
        tolerance=1e-4,
        cases=[case],
    )


def elementary_math_fixture() -> Fixture:
    cases: list[dict[str, Any]] = []
    for rule in (30, 90, 110, 150, 184):
        for seed in (1, 2):
            bits = elementary.to_bits(elementary.random_row(10, seed=seed))
            cases.append({"kind": "preimage_count", "rule": rule, "config": bits,
                          "count": eca.preimage_count(rule, bits)})  # fmt: skip
    for rule in (30, 90, 110):
        cases.append({"kind": "garden_of_eden", "rule": rule, "n": 8,
                      "count": eca.garden_of_eden_count(rule, 8)})  # fmt: skip
    width = 41
    cases.append(
        {
            "kind": "rule90_closed_form",
            "width": width,
            "rows": [elementary.to_bits(eca.rule90_closed_form(t, width)) for t in range(17)],
        }
    )
    return Fixture(
        engine="elementary_math",
        description="Preimage counts of ring configurations (transfer matrices), Garden-of-Eden "
        "counts, and rule 90 rows t = 0..16 from the Lucas closed form.",
        tolerance=None,
        cases=cases,
    )


def all_fixtures(repo_root: Path) -> dict[str, Fixture]:
    return {
        "elementary": elementary_fixture(),
        "elementary_math": elementary_math_fixture(),
        "lifelike": lifelike_fixture(),
        "rle": rle_fixture(),
        "lenia": lenia_fixture(),
        "nca": nca_fixture(repo_root),
    }
