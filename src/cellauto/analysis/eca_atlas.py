"""Behaviour features and the unsupervised atlas of the 256 elementary rules.

Protocol (fixed; prime width avoids the GF(2) nilpotency trap of rules such as 90): ``width`` 251,
256 steps, statistics over the second half, 6 random initial rows (successive draws from
``default_rng(seed)``) shared by all rules, and a paired copy with the centre cell flipped for
the damage measure.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cellauto import __version__
from cellauto.analysis.embed import kmeans, pca, silhouette, standardize
from cellauto.analysis.stats import shannon_entropy
from cellauto.catalog import elementary_rules
from cellauto.engines.elementary import rule_table

FEATURES = ("damage", "compression", "density", "activity", "block_entropy", "period")
# Features unchanged by reflection and complement; these drive the embedding and clustering.
EMBED_FEATURES = ("damage", "compression", "activity", "block_entropy")

FEATURE_INFO: dict[str, tuple[str, str | None, bool]] = {
    "damage": (
        "Damage spreading: Hamming fraction between a run and a copy with the centre cell "
        "flipped, averaged over the window (a Lyapunov-like spreading rate).",
        "fraction", True,
    ),
    "compression": (
        "zlib (level 9) size over raw size of the bit-packed window: ~1 is incompressible, "
        "~0 is trivial.",
        "ratio", True,
    ),
    "density": ("Mean fraction of live cells over the window.", "fraction", False),
    "activity": ("Mean fraction of cells that change per step over the window.", "fraction", True),
    "block_entropy": (
        "Shannon entropy of length-3 spatial blocks over the window, divided by 3.",
        "bits per cell", True,
    ),
    "period": (
        "Smallest p <= 64 such that the last row equals the row p steps earlier shifted by some "
        "s in [-p, p]; 0 if none (aperiodic). Mode over trials.",
        "steps", True,
    ),
}  # fmt: skip


@dataclass(frozen=True)
class Protocol:
    width: int = 251
    steps: int = 256
    trials: int = 6
    seed: int = 0
    max_period: int = 64

    @property
    def window_start(self) -> int:
        return self.steps // 2


DEFAULT = Protocol()
QUICK = Protocol(width=101, steps=96, trials=2)


def _chunk_features(rules: Sequence[int], p: Protocol) -> dict[str, npt.NDArray[Any]]:
    n_rules, width = len(rules), p.width
    batch = n_rules * p.trials
    tables = np.repeat(np.stack([rule_table(r) for r in rules]), p.trials, axis=0)
    rows = np.arange(batch)[:, None]
    # Successive draws from ONE stream, shared by every rule. (Seeding each trial separately
    # gives a different sample, and damage for rules such as 54 or 110 depends strongly on the
    # initial row; the verified reference values were produced with this scheme.)
    stream = np.random.default_rng(p.seed)
    init = np.stack([stream.integers(0, 2, width) for _ in range(p.trials)]).astype(np.uint8)
    x = np.tile(init, (n_rules, 1))
    y = x.copy()
    y[:, width // 2] ^= 1

    def evolve(s: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        idx = (np.roll(s, 1, axis=1) << 2) | (s << 1) | np.roll(s, -1, axis=1)
        return tables[rows, idx]

    ws = p.window_start
    n_states = p.steps - ws + 1  # states x_ws .. x_T
    win = np.empty((n_states, batch, width), dtype=np.uint8)
    damage = np.zeros(batch)
    for t in range(1, p.steps + 1):
        x, y = evolve(x), evolve(y)
        if t >= ws:
            win[t - ws] = x
        if t > ws:
            damage += (x != y).mean(axis=1)
    damage /= p.steps - ws
    body, prev = win[1:], win[:-1]

    density = body.mean(axis=(0, 2))
    activity = (body != prev).mean(axis=(0, 2))
    compression = np.empty(batch)
    block_entropy = np.empty(batch)
    left, right = np.roll(body, 1, axis=2), np.roll(body, -1, axis=2)
    blocks = (left << 2) | (body << 1) | right
    for b in range(batch):
        raw = np.packbits(body[:, b, :]).tobytes()
        compression[b] = len(zlib.compress(raw, 9)) / len(raw)
        block_entropy[b] = shannon_entropy(np.bincount(blocks[:, b, :].ravel(), minlength=8)) / 3

    period = np.zeros(batch, dtype=np.int64)
    last = win[-1]
    open_rows = np.ones(batch, dtype=bool)
    for lag in range(1, min(p.max_period, n_states - 1) + 1):
        if not open_rows.any():
            break
        earlier = win[-1 - lag][open_rows]
        target = last[open_rows]
        hit = np.zeros(earlier.shape[0], dtype=bool)
        for s in range(-lag, lag + 1):
            hit |= (np.roll(earlier, s, axis=1) == target).all(axis=1)
        idx = np.flatnonzero(open_rows)[hit]
        period[idx] = lag
        open_rows[idx] = False

    def mean(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return a.reshape(n_rules, p.trials).mean(axis=1)

    modal = np.array(
        [
            int(np.argmax(np.bincount(r, minlength=p.max_period + 1)))
            for r in period.reshape(n_rules, -1)
        ]
    )
    return {
        "damage": mean(damage), "compression": mean(compression), "density": mean(density),
        "activity": mean(activity), "block_entropy": mean(block_entropy), "period": modal,
    }  # fmt: skip


def behaviour_features(
    rules: Sequence[int] | None = None, protocol: Protocol = DEFAULT, chunk: int = 64
) -> dict[str, npt.NDArray[Any]]:
    """Behaviour features for ``rules`` (default: all 256), vectorised over rules and trials."""
    rs = list(range(256)) if rules is None else [int(r) for r in rules]
    parts = [_chunk_features(rs[i : i + chunk], protocol) for i in range(0, len(rs), chunk)]
    return {name: np.concatenate([part[name] for part in parts]) for name in FEATURES}


def _r(v: float) -> float:
    return round(float(v), 6)


def build_atlas(
    features: dict[str, npt.NDArray[Any]], protocol: Protocol, seed: int = 0
) -> dict[str, Any]:
    """PCA + k-means (k = 3..6 by silhouette, and always k = 4) -> an ``EcaAtlas``-shaped dict."""
    n = len(features["damage"])
    if n != 256:
        raise ValueError("the atlas needs all 256 rules")
    z = standardize(np.column_stack([features[f] for f in EMBED_FEATURES]))
    scores, comps, ratio = pca(z, 2)

    labelings: dict[int, npt.NDArray[np.intp]] = {}
    k_scores: list[dict[str, Any]] = []
    for k in (3, 4, 5, 6):
        labels, _, _ = kmeans(z, k, n_init=20, seed=seed)
        # number clusters by increasing mean compression (0 = most compressible / simplest)
        order = np.argsort([features["compression"][labels == j].mean() for j in range(k)])
        remap = np.empty(k, dtype=np.intp)
        remap[order] = np.arange(k)
        labelings[k] = remap[labels]
        k_scores.append({"k": k, "silhouette": _r(silhouette(z, labelings[k]))})
    chosen = int(max(k_scores, key=lambda s: (s["silhouette"], -s["k"]))["k"])

    reference = {r["rule"]: r["wolfram_class"] for r in elementary_rules()}

    def clustering(k: int) -> dict[str, Any]:
        labels = labelings[k]
        summary = []
        for j in range(k):
            members = np.flatnonzero(labels == j)
            summary.append(
                {
                    "cluster": j,
                    "size": int(members.size),
                    "centroid": {f: _r(features[f][members].mean()) for f in FEATURES},
                    "example_rules": [int(m) for m in members[:8]],
                }
            )
        classes = sorted(set(reference.values()))
        counts = [[0] * k for _ in classes]
        assignments = []
        for rule, cls in sorted(reference.items()):
            counts[classes.index(cls)][int(labels[rule])] += 1
            assignments.append({"rule": rule, "wolfram_class": cls, "cluster": int(labels[rule])})
        return {
            "k": k,
            "silhouette": next(s["silhouette"] for s in k_scores if s["k"] == k),
            "summary": summary,
            "contingency": {
                "classes": classes, "clusters": list(range(k)),
                "counts": counts, "assignments": assignments,
            },
        }  # fmt: skip

    rows = [
        {
            "rule": r,
            **{f: (int(features[f][r]) if f == "period" else _r(features[f][r])) for f in FEATURES},
            "pca_x": _r(scores[r, 0]),
            "pca_y": _r(scores[r, 1]),
            "cluster": int(labelings[chosen][r]),
            "cluster_k4": int(labelings[4][r]),
        }
        for r in range(256)
    ]
    return {
        "format_version": 1,
        "protocol": {**asdict(protocol), "window_start": protocol.window_start},
        "features": [
            {"name": f, "description": d, "unit": u, "symmetry_invariant": inv}
            for f, (d, u, inv) in FEATURE_INFO.items()
        ],
        "embedding": {
            "features": list(EMBED_FEATURES),
            "components": [[_r(v) for v in row] for row in comps],
            "explained_variance": [_r(v) for v in ratio],
        },
        "k_selection": k_scores,
        "chosen_k": chosen,
        "clusterings": [clustering(chosen)] + ([clustering(4)] if chosen != 4 else []),
        "rows": rows,
    }


def run_info() -> dict[str, Any]:
    """Machine and version info for a results file's ``meta`` (never exported to the web)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cellauto": __version__,
        "cpu": platform.processor() or platform.machine(),
        "git_sha": sha,
    }


def run_atlas(quick: bool = False) -> dict[str, Any]:
    """Compute everything; returns ``{"atlas": EcaAtlas-shaped, "meta": {...}}``."""
    protocol = QUICK if quick else DEFAULT
    t0 = time.perf_counter()
    feats = behaviour_features(None, protocol)
    atlas = build_atlas(feats, protocol)
    return {
        "atlas": atlas,
        "meta": {**run_info(), "runtime_s": round(time.perf_counter() - t0, 2), "quick": quick},
    }
