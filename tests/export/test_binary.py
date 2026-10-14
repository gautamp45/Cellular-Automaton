import numpy as np
import pytest

from cellauto.export.binary import (
    CabinColumn,
    encode_cabin,
    from_tensor,
    read_cabin,
    to_tensor,
    write_cabin,
)


def test_tensor_roundtrip() -> None:
    a = np.random.default_rng(0).standard_normal((3, 4, 5)).astype(np.float32)
    t = to_tensor(a)
    assert t.shape == [3, 4, 5] and t.dtype == "float32-le"
    assert np.array_equal(from_tensor(t), a)
    assert np.array_equal(from_tensor(t.model_dump()), a)


def test_cabin_roundtrip_within_quantisation_bound(tmp_path) -> None:
    rng = np.random.default_rng(1)
    n = 1001
    cols = {
        "density": rng.random(n) * 0.3 - 0.1,
        "cluster": rng.integers(0, 6, n).astype(np.uint8),
        "code": np.arange(n, dtype=np.uint16) * 3,
        "mf": rng.standard_normal(n).astype(np.float32),
        "flat": np.full(n, 2.5),
    }
    specs = {
        "density": CabinColumn("uint16", "quantised float", "fraction"),
        "cluster": CabinColumn("uint8", "label"),
        "code": CabinColumn("uint16", "exact integer"),
        "mf": CabinColumn("float32", "raw float"),
        "flat": CabinColumn("uint16", "constant column"),
    }
    header = write_cabin(tmp_path / "t", cols, specs, row_key="row = code >> 1")
    assert header.rows == n and header.format == "cabin"
    assert all(c.offset % 8 == 0 for c in header.columns)
    dens = next(c for c in header.columns if c.name == "density")
    assert dens.scale is not None and dens.zero == pytest.approx(cols["density"].min())
    out = read_cabin(tmp_path / "t.header.json")
    assert np.abs(out["density"] - cols["density"]).max() <= dens.scale / 2 + 1e-12
    assert np.array_equal(out["cluster"], cols["cluster"])
    assert np.array_equal(out["code"], cols["code"])
    assert np.array_equal(out["mf"], cols["mf"])
    assert np.array_equal(out["flat"], cols["flat"])  # max == min -> scale 1, exact


def test_cabin_validation(tmp_path) -> None:
    s = {"a": CabinColumn("uint8", "x")}
    with pytest.raises(ValueError):
        encode_cabin({"a": np.array([300])}, s, "a.bin")
    with pytest.raises(ValueError):
        encode_cabin({"a": np.array([1, 2]), "b": np.array([1])}, {**s, "b": s["a"]}, "a.bin")
    with pytest.raises(ValueError):
        encode_cabin({"a": np.array([np.nan])}, {"a": CabinColumn("float32", "x")}, "a.bin")
    write_cabin(tmp_path / "ok", {"a": np.array([1, 2], dtype=np.uint8)}, s)
    (tmp_path / "ok.bin").write_bytes(b"\x09" * 16)
    with pytest.raises(ValueError):
        read_cabin(tmp_path / "ok.header.json")
