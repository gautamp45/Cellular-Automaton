"""Command-line interface: ``cellauto <command> ...`` (also ``python -m cellauto``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from cellauto import render
from cellauto.catalog import (
    elementary_rules,
    get_pattern,
    lenia_presets,
    lifelike_rules,
    patterns,
    place,
)
from cellauto.engines import elementary, lenia, lifelike


class CliError(Exception):
    """A user error: printed to stderr, exit code 2."""


def _size(text: str) -> tuple[int, int]:
    try:
        h, w = (int(p) for p in text.lower().split("x"))
    except ValueError:
        raise CliError(f"--size must look like 64x64, got {text!r}") from None
    if h < 1 or w < 1:
        raise CliError(f"--size must be positive, got {text!r}")
    return h, w


# --- commands ---------------------------------------------------------------------------------


def _cmd_elementary(a: argparse.Namespace) -> int:
    start = (
        elementary.single_cell(a.width)
        if a.init == "center"
        else elementary.random_row(a.width, a.density, a.seed)
    )
    history = elementary.run(start, a.rule, a.steps, a.boundary)
    render.save_png(render.binary_to_image(history, a.scale), a.out)
    print(f"wrote {a.out}")
    return 0


def _life_rule(a: argparse.Namespace) -> lifelike.LifeRule:
    if a.preset:
        for entry in lifelike_rules():
            if entry["id"] == a.preset:
                return lifelike.LifeRule.parse(entry["rulestring"])
        raise CliError(f"unknown preset {a.preset!r}; try `cellauto list lifelike`")
    return lifelike.LifeRule.parse(a.rule or "B3/S23")


def _cmd_life(a: argparse.Namespace) -> int:
    rule = _life_rule(a)
    h, w = _size(a.size)
    if a.pattern:
        try:
            grid = place(get_pattern(a.pattern), (h, w))
        except KeyError:
            raise CliError(f"unknown pattern {a.pattern!r}; try `cellauto list patterns`") from None
    else:
        grid = lifelike.random_grid(h, w, 0.3 if a.random is None else a.random, a.seed)
    history = lifelike.run(grid, rule, a.steps, a.boundary)
    frames = (render.binary_to_image(g, a.scale) for g in history)
    render.save_gif(frames, a.out, a.fps)
    print(f"wrote {a.out} ({rule.rulestring}, {a.steps + 1} frames)")
    return 0


def _cmd_lenia(a: argparse.Namespace) -> int:
    preset = next((p for p in lenia_presets() if p["id"] == a.preset.lower()), None)
    if preset is None:
        raise CliError(f"unknown preset {a.preset!r}; try `cellauto list lenia`")
    if preset["cells_rle"] is None:
        raise CliError(f"preset {a.preset!r} has no pattern to simulate")
    p = preset["params"]
    params = lenia.LeniaParams(
        R=p["R"], T=p["T"], mu=p["mu"], sigma=p["sigma"], peaks=tuple(p["peaks"])
    )
    h, w = _size(a.size)
    cells = lenia.decode_lenia_rle(preset["cells_rle"])
    if cells.shape[0] > h or cells.shape[1] > w:
        raise CliError(f"--size {a.size} is too small for the {cells.shape} pattern")
    state = np.zeros((h, w), dtype=np.float32)
    r0, c0 = (h - cells.shape[0]) // 2, (w - cells.shape[1]) // 2
    state[r0 : r0 + cells.shape[0], c0 : c0 + cells.shape[1]] = cells
    frames = [render.scalar_to_image(state, a.scale)]
    frames += (render.scalar_to_image(s, a.scale) for s in lenia.run(state, params, a.steps))
    render.save_gif(frames, a.out, a.fps)
    print(f"wrote {a.out} ({preset['name']}, {a.steps + 1} frames)")
    return 0


def _cmd_nca_run(a: argparse.Namespace) -> int:
    from cellauto.nca.animate import growth_gif
    from cellauto.nca.checkpoints import load_manifest, load_model

    entry = next((m for m in load_manifest() if m["id"] == a.model), None)
    if entry is None:
        raise CliError(f"unknown model {a.model!r}; try `cellauto list models`")
    model = load_model(a.model)
    fire = model.config.fire_rate if a.fire_rate is None else a.fire_rate
    if not 0.0 < fire <= 1.0:
        raise CliError("--fire-rate must be in (0, 1]")
    model = type(model)(model.w1, model.b1, model.w2, replace(model.config, fire_rate=fire))
    n = growth_gif(
        model, a.out, grid_size=entry["grid_size"], padding=entry["padding"], steps=a.steps,
        seed=a.seed, damage_at=a.damage_at or (), scale=a.scale, fps=a.fps,
    )  # fmt: skip
    print(f"wrote {a.out} ({a.model}, {n} frames)")
    return 0


def _cmd_nca_train(a: argparse.Namespace) -> int:
    try:
        import yaml

        from cellauto.nca.training.train import train
    except ImportError:
        raise CliError('training needs PyTorch and PyYAML: pip install -e ".[train]"') from None
    with open(a.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise CliError(f"{a.config} must contain a mapping")
    path = train(config)
    print(f"saved {path}")
    return 0


def _cmd_export_web(a: argparse.Namespace) -> int:
    from cellauto.export import check_export, export_web
    from cellauto.nca.checkpoints import find_repo_root

    root = find_repo_root()
    out = Path(a.out) if a.out else root / "web" / "public" / "data"
    if a.check:
        diffs = check_export(out, root)
        for d in diffs:
            print(d)
        if diffs:
            print(f"{len(diffs)} file(s) differ; run `cellauto export-web` to regenerate",
                  file=sys.stderr)  # fmt: skip
            return 1
        print(f"{out} is up to date")
        return 0
    index = export_web(out, root)
    print(f"wrote {len(index.datasets) + 1} files to {out}")
    return 0


def _cmd_list(a: argparse.Namespace) -> int:
    if a.what == "elementary":
        for r in elementary_rules():
            print(f"{r['rule']:>3}  class {r['wolfram_class']}  {r.get('name') or ''}".rstrip())
    elif a.what == "lifelike":
        for r in lifelike_rules():
            print(f"{r['id']:<22}{r['rulestring']:<22}{r['name']}")
    elif a.what == "patterns":
        for r in patterns():
            print(f"{r['id']:<20}{r['category']:<12}{r['name']}")
    elif a.what == "lenia":
        for r in lenia_presets():
            print(f"{r['id']:<16}{r['name']}")
    else:  # models
        from cellauto.nca.checkpoints import load_manifest

        for m in load_manifest():
            print(m["id"])
    return 0


# --- parser -----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cellauto", description="Cellular automata toolkit.")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(
        name: str, help_: str, fn: Callable[[argparse.Namespace], int]
    ) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(func=fn)
        return sp

    e = add("elementary", "render a 1D elementary CA space-time diagram (PNG)", _cmd_elementary)
    e.add_argument("--rule", type=int, required=True)
    e.add_argument("--width", type=int, default=201)
    e.add_argument("--steps", type=int, default=100)
    e.add_argument("--init", choices=["center", "random"], default="center")
    e.add_argument("--density", type=float, default=0.5)
    e.add_argument("--seed", type=int, default=None)
    e.add_argument("--boundary", choices=["wrap", "dead"], default="wrap")
    e.add_argument("--scale", type=int, default=4)
    e.add_argument("--out", required=True)

    li = add("life", "render a life-like 2D CA (GIF)", _cmd_life)
    grp = li.add_mutually_exclusive_group()
    grp.add_argument("--rule", help="B/S rulestring, e.g. B3/S23")
    grp.add_argument("--preset", help="rule id from `cellauto list lifelike`")
    init = li.add_mutually_exclusive_group()
    init.add_argument("--pattern", help="pattern id from `cellauto list patterns`")
    init.add_argument("--random", type=float, help="random soup density (default 0.3)")
    li.add_argument("--seed", type=int, default=None)
    li.add_argument("--size", default="64x64", help="HxW")
    li.add_argument("--steps", type=int, default=200)
    li.add_argument("--boundary", choices=["wrap", "dead"], default="wrap")
    li.add_argument("--fps", type=int, default=20)
    li.add_argument("--scale", type=int, default=6)
    li.add_argument("--out", required=True)

    le = add("lenia", "render a Lenia preset (GIF)", _cmd_lenia)
    le.add_argument("--preset", default="O2u")
    le.add_argument("--size", default="128x128", help="HxW")
    le.add_argument("--steps", type=int, default=300)
    le.add_argument("--fps", type=int, default=20)
    le.add_argument("--scale", type=int, default=3)
    le.add_argument("--out", required=True)

    nca = add("nca", "neural CA commands", lambda _a: 0)
    nsub = nca.add_subparsers(dest="nca_command", required=True, metavar="<nca-command>")
    nr = nsub.add_parser("run", help="grow a trained NCA from the seed (GIF)")
    nr.set_defaults(func=_cmd_nca_run)
    nr.add_argument("--model", required=True)
    nr.add_argument("--steps", type=int, default=200)
    nr.add_argument("--fire-rate", type=float, default=None)
    nr.add_argument("--seed", type=int, default=0)
    nr.add_argument("--damage-at", type=int, nargs="*", help="steps after which to damage")
    nr.add_argument("--fps", type=int, default=20)
    nr.add_argument("--scale", type=int, default=8)
    nr.add_argument("--out", required=True)
    nt = nsub.add_parser("train", help="train an NCA from a YAML config (needs the [train] extra)")
    nt.set_defaults(func=_cmd_nca_train)
    nt.add_argument("--config", required=True, help="YAML config; paths are relative to the CWD")

    ex = add("export-web", "write (or --check) the static data backend", _cmd_export_web)
    ex.add_argument("--out", default=None, help="output dir (default: <repo>/web/public/data)")
    ex.add_argument("--check", action="store_true", help="exit 1 if the output is stale")

    ls = add("list", "list catalog entries", _cmd_list)
    ls.add_argument("what", choices=["elementary", "lifelike", "patterns", "lenia", "models"])
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits with 0 for --help and 2 for usage errors
        return exc.code if isinstance(exc.code, int) else 2
    try:
        return int(args.func(args))
    except (CliError, ValueError, FileNotFoundError, KeyError) as exc:
        print(
            f"error: {exc.args[0] if isinstance(exc, KeyError) and exc.args else exc}",
            file=sys.stderr,
        )
        return 2
