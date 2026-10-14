"""``engines.json``: parameter specs from which the frontend generates its controls."""

from __future__ import annotations

from cellauto.contract.models import EngineSpec, ParamSpec

_SEED_MAX = 2**31 - 1


def _seed() -> ParamSpec:
    return ParamSpec(
        name="seed", label="Seed", description="Random seed for the initial state.",
        kind="int", default=0, min=0, max=_SEED_MAX, step=1,
    )  # fmt: skip


def _boundary() -> ParamSpec:
    return ParamSpec(
        name="boundary", label="Boundary",
        description="wrap: cells at the edge see the opposite edge; dead: outside cells are 0.",
        kind="enum", default="wrap", options=["wrap", "dead"],
    )  # fmt: skip


def _density(default: float) -> ParamSpec:
    return ParamSpec(
        name="density", label="Density",
        description="Probability that a cell is alive in a random initial state.",
        kind="float", default=default, min=0.0, max=1.0, step=0.01,
    )  # fmt: skip


def engine_specs() -> list[EngineSpec]:
    elementary = EngineSpec(
        id="elementary", name="Elementary CA",
        summary="One-dimensional, two-state, radius-1 automata: one of 256 rules (Wolfram).",
        state="binary", dimensions=1, algorithm_anchor="#eca",
        fixtures="fixtures/elementary.json", content_slug="elementary",
        params=[
            ParamSpec(name="rule", label="Rule", description="Wolfram rule number.",
                      kind="int", default=30, min=0, max=255, step=1),
            ParamSpec(name="width", label="Width", description="Number of cells.",
                      kind="int", default=256, min=16, max=2048, step=1, unit="cells"),
            ParamSpec(name="generations", label="Generations",
                      description="Number of rows (time steps) to compute.",
                      kind="int", default=256, min=1, max=2048, step=1),
            _boundary(),
            ParamSpec(name="init", label="Initial state",
                      description="single: one live cell at width // 2; random: Bernoulli "
                      "(density); bits: the `bits` string, left-aligned and zero-padded.",
                      kind="enum", default="single", options=["single", "random", "bits"]),
            ParamSpec(name="bits", label="Bits",
                      description="Initial cells as a 0/1 string (used when init = bits).",
                      kind="bitstring", default=""),
            _density(0.5),
            _seed(),
        ],
    )  # fmt: skip
    lifelike = EngineSpec(
        id="lifelike", name="Life-like CA",
        summary="Two-dimensional outer-totalistic automata on the Moore neighbourhood "
        "(B/S rules such as Conway's Life).",
        state="binary", dimensions=2, algorithm_anchor="#lifelike",
        fixtures="fixtures/lifelike.json", content_slug="lifelike",
        params=[
            ParamSpec(name="rule", label="Rule", description="Birth/survival rulestring.",
                      kind="rulestring", default="B3/S23", ref="lifelike_rules"),
            ParamSpec(name="width", label="Width", description="Grid width.",
                      kind="int", default=128, min=16, max=2048, step=1, unit="cells"),
            ParamSpec(name="height", label="Height", description="Grid height.",
                      kind="int", default=128, min=16, max=2048, step=1, unit="cells"),
            _boundary(),
            ParamSpec(name="init", label="Initial state",
                      description="random: Bernoulli (density); pattern: the chosen pattern, "
                      "centred; empty: all dead (draw cells by hand).",
                      kind="enum", default="random", options=["random", "pattern", "empty"]),
            _density(0.35),
            ParamSpec(name="pattern", label="Pattern", description="Pattern to place.",
                      kind="ref", default="glider", ref="lifelike_patterns"),
            _seed(),
        ],
    )  # fmt: skip
    lenia = EngineSpec(
        id="lenia", name="Lenia",
        summary="Continuous-state automaton with a smooth ring kernel and a Gaussian growth "
        "function (Chan 2019); Life is a special case.",
        state="continuous", dimensions=2, algorithm_anchor="#lenia",
        fixtures="fixtures/lenia.json", content_slug="lenia",
        params=[
            ParamSpec(name="preset", label="Preset", description="Named creature or rule.",
                      kind="ref", default="o2u", ref="lenia_presets"),
            ParamSpec(name="R", label="Radius", description="Kernel radius in cells.",
                      kind="int", default=13, min=3, max=40, step=1, unit="cells"),
            ParamSpec(name="T", label="Time resolution",
                      description="Steps per unit time; each step adds growth / T.",
                      kind="float", default=10.0, min=1.0, max=50.0, step=0.5),
            ParamSpec(name="mu", label="Growth centre (mu)",
                      description="Potential value with maximal growth.",
                      kind="float", default=0.15, min=0.01, max=0.6, step=0.001),
            ParamSpec(name="sigma", label="Growth width (sigma)",
                      description="Width of the Gaussian growth function.",
                      kind="float", default=0.015, min=0.001, max=0.2, step=0.0005),
            ParamSpec(name="peaks", label="Kernel peaks",
                      description="Relative heights of the kernel's concentric shells.",
                      kind="float_list", default=[1.0]),
            ParamSpec(name="size", label="Grid size", description="Side of the square grid.",
                      kind="int", default=128, min=32, max=512, step=1, unit="cells"),
        ],
    )  # fmt: skip
    nca = EngineSpec(
        id="nca", name="Neural CA",
        summary="Cells share a small neural network that reads a fixed 3x3 perception of "
        "their neighbours; trained so that a single seed grows into an image.",
        state="multichannel", dimensions=2, algorithm_anchor="#nca",
        fixtures="fixtures/nca.json", content_slug="nca",
        params=[
            ParamSpec(name="model", label="Model", description="Trained checkpoint.",
                      kind="ref", default="chest-l2-sobel-damage", ref="nca_models"),
            ParamSpec(name="fire_rate", label="Fire rate",
                      description="Probability that a cell updates on a given step.",
                      kind="float", default=0.5, min=0.0, max=1.0, step=0.05),
            ParamSpec(name="damage_radius", label="Damage radius",
                      description="Radius of the disc erased when the grid is clicked.",
                      kind="float", default=0.25, min=0.1, max=0.4, step=0.01,
                      unit="grid units on [-1, 1]"),
            _seed(),
        ],
    )  # fmt: skip
    return [elementary, lifelike, lenia, nca]
