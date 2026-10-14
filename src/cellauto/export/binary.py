"""Tensors (base64 float32) and the "cabin" binary columnar format.

Cabin = a ``BinaryTableHeader`` JSON plus one ``.bin``: the concatenation of columns, each
little-endian and contiguous, each starting at an 8-byte-aligned byte offset. A ``uint16`` column
holding floats is quantised: ``q = round((v - zero) / scale)`` with ``zero = min`` and
``scale = (max - min) / 65535`` (1 if max == min), so ``|v - decoded| <= scale / 2``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from cellauto.contract.models import BinaryTableHeader, ColumnSpec, Tensor
from cellauto.contract.schema import dumps

_ALIGN = 8
_DTYPES = {"uint8": "<u1", "uint16": "<u2", "float32": "<f4"}


def to_tensor(arr: npt.ArrayLike) -> Tensor:
    """Encode an array as little-endian float32, row-major, base64."""
    a = np.ascontiguousarray(np.asarray(arr), dtype="<f4")
    return Tensor(
        shape=list(a.shape), dtype="float32-le", b64=base64.b64encode(a.tobytes()).decode()
    )


def from_tensor(t: Tensor | dict[str, object]) -> npt.NDArray[np.float32]:
    tensor = t if isinstance(t, Tensor) else Tensor.model_validate(t)
    flat = np.frombuffer(base64.b64decode(tensor.b64), dtype="<f4")
    return flat.reshape(tensor.shape).astype(np.float32)


@dataclass(frozen=True)
class CabinColumn:
    dtype: Literal["uint8", "uint16", "float32"]
    description: str
    unit: str | None = None


def encode_cabin(
    columns: dict[str, npt.ArrayLike],
    specs: dict[str, CabinColumn],
    bin_path: str,
    row_key: str = "row",
) -> tuple[BinaryTableHeader, bytes]:
    """Encode columns to ``(header, bin_bytes)``. Column order is the dict order."""
    if not columns:
        raise ValueError("no columns")
    if set(columns) != set(specs):
        raise ValueError("columns and specs must have the same keys")
    arrays = {k: np.asarray(v) for k, v in columns.items()}
    rows = {a.shape[0] for a in arrays.values() if a.ndim == 1}
    if len(rows) != 1 or any(a.ndim != 1 for a in arrays.values()):
        raise ValueError("all columns must be 1-D with the same length")
    (n_rows,) = rows
    blob = bytearray()
    out_cols: list[ColumnSpec] = []
    for name, arr in arrays.items():
        spec = specs[name]
        if not np.isfinite(arr.astype(np.float64)).all():
            raise ValueError(f"column {name!r} has non-finite values")
        scale: float | None = None
        zero: float | None = None
        if spec.dtype == "uint16" and np.issubdtype(arr.dtype, np.floating):
            lo, hi = float(arr.min()), float(arr.max())
            zero = lo
            scale = (hi - lo) / 65535 if hi > lo else 1.0
            q = np.rint((arr.astype(np.float64) - zero) / scale)
            stored = q.astype("<u2")
        elif spec.dtype == "float32":
            stored = arr.astype("<f4")
        else:
            info = np.iinfo(np.dtype(_DTYPES[spec.dtype]))
            if arr.size and (arr.min() < info.min or arr.max() > info.max):
                raise ValueError(f"column {name!r} does not fit {spec.dtype}")
            stored = arr.astype(_DTYPES[spec.dtype])
        blob.extend(b"\0" * (-len(blob) % _ALIGN))
        offset = len(blob)
        raw = stored.tobytes()
        blob.extend(raw)
        out_cols.append(
            ColumnSpec(
                name=name,
                dtype=spec.dtype,
                offset=offset,
                length=len(raw),
                scale=scale,
                zero=zero,
                unit=spec.unit,
                description=spec.description,
            )
        )
    data = bytes(blob)
    header = BinaryTableHeader(
        format="cabin",
        version=1,
        rows=int(n_rows),
        row_key=row_key,
        bin_path=bin_path,
        bin_sha256=hashlib.sha256(data).hexdigest(),
        columns=out_cols,
    )
    return header, data


def write_cabin(
    path_stem: str | Path,
    columns: dict[str, npt.ArrayLike],
    specs: dict[str, CabinColumn],
    row_key: str = "row",
) -> BinaryTableHeader:
    """Write ``<stem>.header.json`` and ``<stem>.bin`` and return the header."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    header, data = encode_cabin(columns, specs, f"{stem.name}.bin", row_key)
    Path(f"{stem}.bin").write_bytes(data)
    Path(f"{stem}.header.json").write_text(
        dumps(header.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )
    return header


def decode_cabin(header: BinaryTableHeader, data: bytes) -> dict[str, npt.NDArray[np.generic]]:
    if hashlib.sha256(data).hexdigest() != header.bin_sha256:
        raise ValueError("bin file does not match the header's sha256")
    out: dict[str, npt.NDArray[np.generic]] = {}
    for col in header.columns:
        raw = np.frombuffer(data, dtype=_DTYPES[col.dtype], count=header.rows, offset=col.offset)
        if col.scale is not None and col.zero is not None:
            out[col.name] = col.zero + col.scale * raw.astype(np.float64)
        else:
            out[col.name] = raw.copy()
    return out


def read_cabin(header_path: str | Path) -> dict[str, npt.NDArray[np.generic]]:
    """Load a cabin table; quantised uint16 columns are returned dequantised (float64)."""
    path = Path(header_path)
    header = BinaryTableHeader.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return decode_cabin(header, (path.parent / header.bin_path).read_bytes())
