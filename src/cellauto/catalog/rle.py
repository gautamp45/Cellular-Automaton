"""Run-length-encoded (RLE) Life patterns: parse, encode and place on a grid."""

from __future__ import annotations

import re

import numpy as np

from cellauto._types import BinaryArray
from cellauto._validate import as_binary

_HEADER = re.compile(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)", re.IGNORECASE)


def parse_rle(text: str) -> BinaryArray:
    """Parse Life RLE.

    Lines starting with ``#`` are ignored. An optional header ``x = W, y = H[, rule = ...]``
    pads the result to at least ``H x W``. The body is ``[count]('b' | 'o' | '$')...'!'``;
    whitespace is ignored and rows are padded on the right with 0.
    """
    header_hw: tuple[int, int] | None = None
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if header_hw is None and not body and (m := _HEADER.match(stripped)):
            header_hw = (int(m.group(2)), int(m.group(1)))
            continue
        body.append(stripped)
    src = "".join("".join(body).split())

    cells: list[tuple[int, int]] = []
    row = col = max_col = 0
    count = ""
    terminated = False
    for ch in src:
        if ch.isdigit():
            count += ch
            continue
        if ch == "!" and count:
            raise ValueError("RLE has a run count with no token before '!'")
        n = int(count) if count else 1
        count = ""
        if ch == "b":
            col += n
        elif ch == "o":
            cells.extend((row, col + k) for k in range(n))
            col += n
        elif ch == "$":
            max_col = max(max_col, col)
            row += n
            col = 0
        elif ch == "!":
            terminated = True
            break
        else:
            raise ValueError(f"unknown RLE token {ch!r}")
    if count:
        raise ValueError("RLE ends with a dangling run count")
    if not terminated and src:
        raise ValueError("RLE must end with '!'")
    max_col = max(max_col, col)
    height = max((r for r, _ in cells), default=-1) + 1
    width = max_col
    if header_hw is not None:
        height, width = max(height, header_hw[0]), max(width, header_hw[1])
    grid = np.zeros((height, width), dtype=np.uint8)
    for r, c in cells:
        grid[r, c] = 1
    return grid


def to_rle(grid: BinaryArray) -> str:
    """Encode without a header; the empty border is trimmed and the result ends with ``!``."""
    g = as_binary(grid, 2)
    occupied_rows = np.flatnonzero(g.any(axis=1))
    if occupied_rows.size == 0:
        return "!"
    occupied_cols = np.flatnonzero(g.any(axis=0))
    r0, r1 = int(occupied_rows[0]), int(occupied_rows[-1])
    c0, c1 = int(occupied_cols[0]), int(occupied_cols[-1])
    g = g[r0 : r1 + 1, c0 : c1 + 1]

    def run(sym: str, n: int) -> str:
        return sym if n == 1 else f"{n}{sym}"

    parts: list[str] = []
    blank = 0
    for r, line in enumerate(g):
        if not line.any():
            blank += 1
            continue
        if r > 0:
            parts.append(run("$", blank + 1))
        blank = 0
        cur, n = int(line[0]), 0
        last = int(np.nonzero(line)[0].max())
        for v in line[: last + 1]:
            if int(v) == cur:
                n += 1
            else:
                parts.append(run("o" if cur else "b", n))
                cur, n = int(v), 1
        parts.append(run("o" if cur else "b", n))
    return "".join(parts) + "!"


def place(
    pattern: BinaryArray, shape: tuple[int, int], at: tuple[int, int] | None = None
) -> BinaryArray:
    """Copy ``pattern`` into a zero grid of ``shape``; centred unless ``at=(row, col)`` is given."""
    p = as_binary(pattern, 2)
    (h, w), (height, width) = p.shape, shape
    r, c = ((height - h) // 2, (width - w) // 2) if at is None else at
    if h > height or w > width or r < 0 or c < 0 or r + h > height or c + w > width:
        raise ValueError(f"pattern {p.shape} at {(r, c)} does not fit in {shape}")
    out = np.zeros(shape, dtype=np.uint8)
    out[r : r + h, c : c + w] = p
    return out
