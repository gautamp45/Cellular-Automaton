"""The complete rule-space atlas: simulate all 131,072 B0-free life-like rules and map them.

B0 rules make empty space come alive every step, so the picture strobes and a finite-grid
emulation would complicate every feature; the atlas excludes them. B0-free rules are exactly the
even 18-bit codes, so atlas row ``i`` is the rule with code ``2 * i``.

Protocol (fixed): a 61x61 torus (prime side), initial density 0.5, 200 steps, the SAME initial
grid for every rule (a controlled comparison) and a copy with the centre cell flipped for damage
spreading. Windowed statistics use states ``t = 150 .. 199``; damage and compression use the
state at ``t = 200``.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cellauto.analysis.embed import kmeans, pca, silhouette, standardize
from cellauto.analysis.lifelike import lambda_codes, mean_field_codes
from cellauto.analysis.stats import spearman
from cellauto.engines.lifelike import tables_from_codes

N_ROWS = 2**17
RAW_FEATURES = ("density", "density_std", "activity", "damage", "compression")


@dataclass(frozen=True)
class AtlasProtocol:
    grid: int = 61
    steps: int = 200
    window_start: int = 150
    density: float = 0.5
    seed: int = 0
    batch: int = 512  # rules simulated together
    chunk: int = 4096  # rules per resumable chunk file


DEFAULT = AtlasProtocol()


luts_from_codes = tables_from_codes


def _fast_step(
    g: npt.NDArray[np.uint8], flat: npt.NDArray[np.uint8], base: npt.NDArray[np.int32]
) -> npt.NDArray[np.uint8]:
    """One torus step for a batch; equivalent to ``lifelike.step_batch`` but ~25% faster."""
    row_sum = g + np.roll(g, 1, axis=2) + np.roll(g, -1, axis=2)
    counts = row_sum + np.roll(row_sum, 1, axis=1) + np.roll(row_sum, -1, axis=1) - g
    return np.take(flat, base + g.astype(np.int32) * 9 + counts)


def simulate_batch(codes: npt.NDArray[np.int64], p: AtlasProtocol = DEFAULT) -> dict[str, Any]:
    """Raw simulated features for a batch of (B0-free) rule codes."""
    r = len(codes)
    luts = luts_from_codes(codes)
    flat = luts.reshape(-1)
    base = (np.arange(r, dtype=np.int32) * 18)[:, None, None]
    start = (np.random.default_rng(p.seed).random((p.grid, p.grid)) < p.density).astype(np.uint8)
    g = np.broadcast_to(start, (r, p.grid, p.grid)).copy()
    h = g.copy()
    h[:, p.grid // 2, p.grid // 2] ^= 1
    dens: list[npt.NDArray[np.float64]] = []
    act = np.zeros(r)
    n_window = p.steps - p.window_start
    for t in range(1, p.steps + 1):
        prev = g
        g = _fast_step(g, flat, base)
        h = _fast_step(h, flat, base)
        if p.window_start <= t < p.steps:
            dens.append(g.mean(axis=(1, 2)))
            act += (g != prev).mean(axis=(1, 2))
    d = np.stack(dens)
    compression = np.empty(r)
    for i in range(r):
        raw = np.packbits(g[i]).tobytes()
        compression[i] = len(zlib.compress(raw, 9)) / len(raw)
    return {
        "density": d.mean(axis=0),
        "density_std": d.std(axis=0),
        "activity": act / n_window,
        "damage": (g != h).mean(axis=(1, 2)),
        "compression": compression,
    }


def compute_rows(rows: npt.NDArray[np.int64], p: AtlasProtocol = DEFAULT) -> dict[str, Any]:
    """Features for atlas rows (codes ``2 * rows``), in batches of ``p.batch``."""
    parts = [simulate_batch(2 * rows[i : i + p.batch], p) for i in range(0, len(rows), p.batch)]
    out = {k: np.concatenate([part[k] for part in parts]) for k in RAW_FEATURES}
    out["rows"] = rows
    return out


def _chunk_job(args: tuple[str, npt.NDArray[np.int64], AtlasProtocol]) -> str:
    path, rows, p = args
    feats = compute_rows(rows, p)
    tmp = path + ".tmp.npz"
    np.savez(tmp, rows=feats["rows"].astype(np.int32),
             **{k: feats[k].astype(np.float32) for k in RAW_FEATURES})  # fmt: skip
    Path(tmp).replace(path)  # atomic: a crash never leaves a half-written chunk
    return path


def run_chunks(
    out_dir: str | Path,
    rows: npt.NDArray[np.int64] | None = None,
    p: AtlasProtocol = DEFAULT,
    workers: int | None = None,
) -> list[Path]:
    """Simulate ``rows`` (default: all) into ``out_dir/chunk_<i>.npz``, skipping existing chunks."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = np.arange(N_ROWS, dtype=np.int64) if rows is None else np.asarray(rows)
    jobs, paths = [], []
    for i, start in enumerate(range(0, len(all_rows), p.chunk)):
        path = out / f"chunk_{i:03d}.npz"
        paths.append(path)
        if not path.is_file():
            jobs.append((str(path), all_rows[start : start + p.chunk], p))
    if jobs:
        if workers == 1:
            list(map(_chunk_job, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_chunk_job, jobs))
    return paths


def load_chunks(paths: Sequence[str | Path]) -> dict[str, npt.NDArray[Any]]:
    parts = [dict(np.load(p)) for p in paths]
    merged = {k: np.concatenate([part[k] for part in parts]) for k in ("rows", *RAW_FEATURES)}
    order = np.argsort(merged["rows"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def _q(v: float) -> float:
    return round(float(v), 6)


def analyse(
    raw: dict[str, npt.NDArray[Any]],
    catalog: Sequence[tuple[str, int]],
    p: AtlasProtocol = DEFAULT,
    seed: int = 0,
) -> tuple[dict[str, npt.NDArray[Any]], dict[str, Any]]:
    """Add lambda, mean-field density, PCA coordinates and clusters; build the summary.

    Returns ``(columns, summary)`` where ``columns`` are the atlas table's columns (full float
    precision) and ``summary`` is a ``LifelikeAtlasMeta``-shaped dict (without ``header_path``).
    """
    rows = raw["rows"].astype(np.int64)
    codes = 2 * rows
    n = len(rows)
    feats = {k: raw[k].astype(np.float64) for k in RAW_FEATURES}
    lam = lambda_codes(codes)
    mf = mean_field_codes(codes)

    z = standardize(np.column_stack([feats[k] for k in RAW_FEATURES]))
    scores, comps, ratio = pca(z, 2)

    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, size=min(10_000, n), replace=False))
    k_scores = []
    fits = {}
    for k in range(3, 9):
        _, cents, _ = kmeans(z[sub], k, n_init=5, seed=seed)
        d = ((z[sub][:, None, :] - cents[None]) ** 2).sum(axis=2)
        sub_labels = d.argmin(axis=1)
        k_scores.append({"k": k, "silhouette": _q(silhouette(z[sub], sub_labels))})
        fits[k] = cents
    chosen = int(max(k_scores, key=lambda s: (s["silhouette"], -s["k"]))["k"])
    cents = fits[chosen]
    labels = np.empty(n, dtype=np.intp)
    for s in range(0, n, 16384):
        d = ((z[s : s + 16384, None, :] - cents[None]) ** 2).sum(axis=2)
        labels[s : s + 16384] = d.argmin(axis=1)
    # number clusters by increasing mean activity (0 = quietest)
    order = np.argsort([feats["activity"][labels == j].mean() if (labels == j).any() else 9
                        for j in range(chosen)])  # fmt: skip
    remap = np.empty(chosen, dtype=np.intp)
    remap[order] = np.arange(chosen)
    labels[:] = remap[labels]

    row_of = {int(c) // 2: i for i, c in enumerate(codes)}
    in_catalog = np.zeros(n, dtype=np.uint8)
    catalog_rows = []
    for rule_id, code in catalog:
        row = row_of.get(code // 2) if code % 2 == 0 else None
        if row is not None:
            in_catalog[row] = 1
        catalog_rows.append({"id": rule_id, "code": int(code),
                             "row": None if row is None else int(rows[row])})  # fmt: skip

    edges = np.unique(np.quantile(lam, np.linspace(0, 1, 21)))
    which = np.clip(np.searchsorted(edges, lam, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        m = which == b
        entry: dict[str, Any] = {"lambda_lo": _q(edges[b]), "lambda_hi": _q(edges[b + 1]),
                                 "n": int(m.sum())}  # fmt: skip
        for name in ("activity", "damage", "compression"):
            q1, med, q3 = np.quantile(feats[name][m], [0.25, 0.5, 0.75])
            entry |= {f"{name}_median": _q(med), f"{name}_q1": _q(q1), f"{name}_q3": _q(q3)}
        bins.append(entry)

    columns: dict[str, npt.NDArray[Any]] = {
        **feats, "lambda": lam, "mf_rho": mf,
        "pca_x": scores[:, 0], "pca_y": scores[:, 1],
        "cluster": labels.astype(np.uint8), "is_catalog": in_catalog,
    }  # fmt: skip
    summary = {
        "format_version": 1,
        "protocol": {"grid": p.grid, "steps": p.steps, "window_start": p.window_start,
                     "density": p.density, "seed": p.seed, "n_rows": int(n),
                     "row_key": "row = code >> 1 (only even, B0-free codes are in the atlas)"},
        "embedding_features": list(RAW_FEATURES),
        "embedding_components": [[_q(v) for v in row] for row in comps],
        "explained_variance": [_q(v) for v in ratio],
        "k_selection": k_scores,
        "chosen_k": chosen,
        "clusters": [
            {"cluster": j, "size": int((labels == j).sum()),
             "centroid": {k: _q(feats[k][labels == j].mean()) for k in RAW_FEATURES}}
            for j in range(chosen)
        ],
        "edge_of_chaos": bins,
        "correlations": [
            {"feature": k, "spearman_with_lambda": _q(spearman(lam, feats[k]))}
            for k in ("density", "activity", "damage", "compression")
        ],
        "catalog": catalog_rows,
    }  # fmt: skip
    return columns, summary


CABIN_COLUMNS: dict[str, tuple[str, str, str | None]] = {
    "density": ("uint16", "Mean density over t = 150..199.", "fraction"),
    "density_std": ("uint16", "Standard deviation of the density over the window.", "fraction"),
    "activity": ("uint16", "Mean fraction of cells changed per step over the window.", "fraction"),
    "damage": ("uint16", "Hamming fraction, original vs perturbed run, at t = 200.", "fraction"),
    "compression": ("uint16", "zlib size / raw size of the packed final grid.", "ratio"),
    "lambda": ("uint16", "Analytic Langton lambda.", "fraction"),
    "mf_rho": ("uint16", "Mean-field density iterated from 0.5.", "fraction"),
    "pca_x": ("uint16", "First principal component of the standardised simulated features.", None),
    "pca_y": ("uint16", "Second principal component.", None),
    "cluster": ("uint8", "k-means cluster (0 = quietest).", None),
    "is_catalog": ("uint8", "1 if the rule is in the named-rule catalog.", None),
}  # fmt: skip


def sample_rows(n: int = 4096, seed: int = 0) -> npt.NDArray[np.int64]:
    """The fixed random sample of atlas rows used by ``--quick`` runs."""
    return np.sort(np.random.default_rng(seed).choice(N_ROWS, size=n, replace=False)).astype(
        np.int64
    )


def build_results(
    results_dir: str | Path, quick: bool = False, workers: int | None = None
) -> dict[str, Any]:
    """Run (or resume) the atlas; write ``atlas.header.json``, ``atlas.bin`` and ``summary.json``.

    Files go to ``<results>/lifelike_atlas/`` (``<results>/quick/lifelike_atlas/`` for quick
    runs). Returns the contents of ``summary.json``.
    """
    import json
    import time

    from cellauto.analysis.eca_atlas import run_info
    from cellauto.analysis.lifelike import mean_field_report
    from cellauto.catalog import lifelike_rules
    from cellauto.contract.schema import dumps
    from cellauto.engines.lifelike import LifeRule
    from cellauto.export.binary import CabinColumn, write_cabin

    t0 = time.perf_counter()
    root = Path(results_dir) / ("quick" if quick else "") / "lifelike_atlas"
    rows = sample_rows() if quick else None
    raw = load_chunks(run_chunks(root / "chunks", rows, DEFAULT, workers))
    rules = [(r["id"], LifeRule.parse(r["rulestring"])) for r in lifelike_rules()]
    columns, summary = analyse(raw, [(i, r.code) for i, r in rules], DEFAULT)
    if quick:
        summary["protocol"]["row_key"] = "row i = the rule with code 2 * sample_rows()[i]"
    specs = {k: CabinColumn(dt, desc, unit) for k, (dt, desc, unit) in CABIN_COLUMNS.items()}  # type: ignore[arg-type]
    header = write_cabin(root / "atlas", {k: columns[k] for k in CABIN_COLUMNS}, specs,
                         summary["protocol"]["row_key"])  # fmt: skip
    summary["header_path"] = "lifelike/atlas.header.json"
    mean_field = {rid: mean_field_report(r) for rid, r in rules}
    result = {
        "summary": summary,
        "mean_field": mean_field,
        "meta": {**run_info(), "runtime_s": round(time.perf_counter() - t0, 2), "quick": quick,
                 "bin_sha256": header.bin_sha256, "workers": workers},
    }  # fmt: skip
    (root / "summary.json").write_text(dumps(result), encoding="utf-8")
    json.loads((root / "summary.json").read_text())  # sanity: valid JSON
    return result
