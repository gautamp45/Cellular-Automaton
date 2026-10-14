import json
import os

import numpy as np
import pytest

from cellauto.analysis import atlas as A
from cellauto.contract.models import LifelikeAtlasMeta
from cellauto.engines.lifelike import LifeRule, step_batch
from cellauto.export.binary import read_cabin

LIFE_ROW = LifeRule.parse("B3/S23").code // 2


def test_luts_match_rule_tables() -> None:
    codes = np.array([6152, 0, 2 * 12345, 2**18 - 2])
    for lut, code in zip(A.luts_from_codes(codes), codes, strict=True):
        assert np.array_equal(lut, LifeRule.from_code(int(code)).table())


def test_fast_step_equals_engine() -> None:
    rng = np.random.default_rng(0)
    codes = 2 * rng.integers(0, A.N_ROWS, 24)
    luts = A.luts_from_codes(codes)
    g = (rng.random((24, 61, 61)) < 0.5).astype(np.uint8)
    base = (np.arange(24, dtype=np.int32) * 18)[:, None, None]
    assert np.array_equal(A._fast_step(g, luts.reshape(-1), base), step_batch(g, luts))


@pytest.fixture(scope="module")
def subset() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    rows = np.array(sorted({LIFE_ROW, *rng.integers(0, A.N_ROWS, 63).tolist()}), dtype=np.int64)
    return A.compute_rows(rows)


def test_subset_is_deterministic_and_life_is_sane(subset: dict[str, np.ndarray]) -> None:
    again = A.compute_rows(subset["rows"])
    assert all(np.array_equal(subset[k], again[k]) for k in A.RAW_FEATURES)
    i = int(np.flatnonzero(subset["rows"] == LIFE_ROW)[0])
    assert 0.03 <= subset["density"][i] <= 0.12
    assert 0 < subset["activity"][i] < 0.3 and 0 < subset["damage"][i] < 1
    # features are in range
    assert (subset["density"] >= 0).all() and (subset["density"] <= 1).all()
    assert (subset["compression"] > 0).all()


def test_batch_size_does_not_change_results(subset: dict[str, np.ndarray]) -> None:
    """Up to floating-point summation order in the mean reductions (~1e-16)."""
    small = A.compute_rows(subset["rows"][:10], A.AtlasProtocol(batch=3))
    for k in A.RAW_FEATURES:
        assert np.allclose(small[k], subset[k][:10], rtol=0, atol=1e-12), k


def test_chunks_are_resumable(tmp_path) -> None:
    rows = np.arange(100, 164, dtype=np.int64)
    p = A.AtlasProtocol(chunk=32)
    paths = A.run_chunks(tmp_path, rows, p, workers=1)
    assert [x.name for x in paths] == ["chunk_000.npz", "chunk_001.npz"]
    first = paths[0].stat().st_mtime_ns
    paths[1].unlink()
    A.run_chunks(tmp_path, rows, p, workers=1)  # recomputes only the missing chunk
    assert paths[0].stat().st_mtime_ns == first and paths[1].is_file()
    loaded = A.load_chunks(paths)
    assert np.array_equal(loaded["rows"], rows)
    ref = A.compute_rows(rows)
    assert np.allclose(loaded["density"], ref["density"], atol=1e-6)  # float32 storage
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_analyse_summary_is_valid_and_consistent(subset: dict[str, np.ndarray]) -> None:
    raw = {**subset, **{k: subset[k] for k in A.RAW_FEATURES}}
    life = LifeRule.parse("B3/S23")
    columns, summary = A.analyse(raw, [("life", life.code), ("antilife", 0b1 | (1 << 9))])
    summary["header_path"] = "lifelike/atlas.header.json"
    meta = LifelikeAtlasMeta.model_validate(summary)
    n = len(subset["rows"])
    assert set(A.CABIN_COLUMNS) <= set(columns)
    assert sum(c.size for c in meta.clusters) == n
    assert 3 <= meta.chosen_k <= 8 and len(meta.k_selection) == 6
    assert sum(b.n for b in meta.edge_of_chaos) == n
    assert columns["is_catalog"].sum() == 1  # Life is in the sample, the B0 rule is not placeable
    row = {c.id: c.row for c in meta.catalog}
    assert row["life"] == LIFE_ROW and row["antilife"] is None
    assert columns["lambda"].min() >= 0 and columns["lambda"].max() <= 1
    assert abs(float(columns["pca_x"].mean())) < 1e-9  # PCA scores are centred


@pytest.mark.slow
def test_quick_results_roundtrip(tmp_path) -> None:
    result = A.build_results(tmp_path, quick=True, workers=os.cpu_count())
    root = tmp_path / "quick" / "lifelike_atlas"
    LifelikeAtlasMeta.model_validate({**result["summary"]})
    table = read_cabin(root / "atlas.header.json")
    assert set(table) == set(A.CABIN_COLUMNS)
    assert len(table["density"]) == 4096
    assert set(np.unique(table["is_catalog"])) <= {0, 1}
    assert result["meta"]["quick"] is True
    assert set(result["mean_field"]) >= {"life", "seeds", "coral"}
    saved = json.loads((root / "summary.json").read_text())
    assert (
        saved["meta"]["bin_sha256"]
        == json.loads((root / "atlas.header.json").read_text())["bin_sha256"]
    )
    # quantisation error bound
    raw = A.load_chunks(sorted((root / "chunks").glob("chunk_*.npz")))
    header = json.loads((root / "atlas.header.json").read_text())
    scale = next(c["scale"] for c in header["columns"] if c["name"] == "density")
    assert np.abs(table["density"] - raw["density"]).max() <= scale / 2 + 1e-6
