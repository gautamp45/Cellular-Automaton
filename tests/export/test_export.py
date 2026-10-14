import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cellauto.catalog import parse_rle
from cellauto.contract.models import Fixture, Index, NcaWeights
from cellauto.contract.schema import TOP_LEVEL, adapter
from cellauto.engines import elementary, lenia, lifelike
from cellauto.export import build_export, check_export, export_web
from cellauto.export.binary import from_tensor
from cellauto.nca.checkpoints import find_repo_root, load_model
from cellauto.nca.image import make_seed

ROOT = find_repo_root(Path(__file__))


@pytest.fixture(scope="module")
def files() -> dict[str, bytes]:
    return build_export(ROOT)


def test_export_is_deterministic(tmp_path: Path, files: dict[str, bytes]) -> None:
    export_web(tmp_path / "a", ROOT)
    export_web(tmp_path / "b", ROOT)
    tree_a = {p.relative_to(tmp_path / "a"): p.read_bytes() for p in (tmp_path / "a").rglob("*")
              if p.is_file()}  # fmt: skip
    tree_b = {p.relative_to(tmp_path / "b"): p.read_bytes() for p in (tmp_path / "b").rglob("*")
              if p.is_file()}  # fmt: skip
    assert tree_a == tree_b
    assert {p.as_posix() for p in tree_a} == set(files)


def test_index_matches_files(files: dict[str, bytes]) -> None:
    index = Index.model_validate_json(files["index.json"])
    assert [d.path for d in index.datasets] == sorted(p for p in files if p != "index.json")
    for d in index.datasets:
        assert hashlib.sha256(files[d.path]).hexdigest() == d.sha256
        assert len(files[d.path]) == d.bytes
        if d.schema_path:
            assert d.schema_path in files
    assert {d.id for d in index.datasets} >= {"engines", "elementary_rules", "nca_models"}


def test_no_timestamps_and_sorted_keys(files: dict[str, bytes]) -> None:
    for path, data in files.items():
        if path.endswith(".json"):
            text = data.decode("utf-8")
            assert text.endswith("\n") and "\n" not in text[:-1]
            doc = json.loads(text)
            assert text == json.dumps(doc, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False) + "\n", path  # fmt: skip


SCHEMA_OF = {
    "engines.json": "engines",
    "elementary/rules.json": "elementary_rules",
    "elementary/atlas.json": "eca_atlas",
    "lifelike/atlas_summary.json": "lifelike_atlas_meta",
    "lifelike/rules.json": "lifelike_rules",
    "lifelike/patterns.json": "lifelike_patterns",
    "lenia/presets.json": "lenia_presets",
    "nca/models.json": "nca_models",
    "content/index.json": "content_index",
    "index.json": "index",
}


def test_every_json_validates_against_its_model(files: dict[str, bytes]) -> None:
    for path, name in SCHEMA_OF.items():
        adapter(name).validate_json(files[path])
    for path, data in files.items():
        if path.startswith("nca/weights/"):
            NcaWeights.model_validate_json(data)
        elif path.startswith("fixtures/"):
            Fixture.model_validate_json(data)
    assert set(SCHEMA_OF.values()) | {"nca_weights", "fixture", "binary_table_header"} == set(
        TOP_LEVEL
    )


def test_catalog_contents(files: dict[str, bytes]) -> None:
    rules = json.loads(files["elementary/rules.json"])
    assert [r["rule"] for r in rules] == list(range(256))
    assert rules[110]["wolfram_class"] == 4 and rules[110]["name"] == "Rule 110"
    assert rules[7]["wolfram_class"] is None and rules[7]["surjective"] is False
    assert rules[30]["table"] == [0, 1, 1, 1, 1, 0, 0, 0]
    assert rules[30]["equivalence_class"] == [30, 86, 135, 149] and rules[30]["lambda"] == 0.5
    assert rules[30]["surjective"] is True and rules[30]["reversible"] is False
    assert [r["rule"] for r in rules if r["reversible"]] == [15, 51, 85, 170, 204, 240]
    assert sum(r["surjective"] for r in rules) == 30
    life = next(r for r in json.loads(files["lifelike/rules.json"]) if r["id"] == "life")
    assert life["rulestring"] == "B3/S23" and life["code"] == 8 + 2048 + 4096
    assert life["dual"] == "B0123478/S01234678" and life["self_dual"] is False
    assert life["lambda"] == 140 / 512
    mf = life["mean_field"]
    assert [round(p["rho"], 4) for p in mf["fixed_points"]] == [0.0, 0.1925, 0.3702]
    assert 0.03 < mf["simulated_density"] < 0.12 and mf["correlation_gap"] > 0.2
    pats = {p["id"]: p for p in json.loads(files["lifelike/patterns.json"])}
    assert pats["pulsar"]["width"] == pats["pulsar"]["height"] == 13
    assert len(pats["glider"]["cells"]) == 5
    models = json.loads(files["nca/models.json"])
    assert len(models) == 25 and all(m["metrics"] is None for m in models)
    assert {m["id"]: m["isotropic"] for m in models}["chest-l2-gaussian"] is True


def test_engine_param_specs(files: dict[str, bytes]) -> None:
    engines = {e["id"]: e for e in json.loads(files["engines.json"])}
    assert set(engines) == {"elementary", "lifelike", "lenia", "nca"}
    el = {p["name"]: p for p in engines["elementary"]["params"]}
    assert (el["rule"]["min"], el["rule"]["max"], el["rule"]["default"]) == (0, 255, 30)
    assert el["seed"]["max"] == 2**31 - 1
    # every `ref` parameter points at a dataset id that exists in the index
    ids = {d["id"] for d in json.loads(files["index.json"])["datasets"]}
    for e in engines.values():
        for p in e["params"]:
            if p["kind"] in ("ref", "rulestring") and p["ref"]:
                assert p["ref"] in ids, p
            if p["kind"] == "ref":
                dataset = json.loads(
                    files[
                        next(
                            d["path"]
                            for d in json.loads(files["index.json"])["datasets"]
                            if d["id"] == p["ref"]
                        )
                    ]
                )
                assert p["default"] in {row["id"] for row in dataset}, p
        assert e["fixtures"] in files and f"content/{e['content_slug']}.md" in files


def test_nca_weights_match_npz(files: dict[str, bytes]) -> None:
    model = load_model("chest-l2-sobel-damage", ROOT)
    w = NcaWeights.model_validate_json(files["nca/weights/chest-l2-sobel-damage.json"])
    assert np.array_equal(from_tensor(w.w1), model.w1)
    assert np.array_equal(from_tensor(w.b1), model.b1)
    assert np.array_equal(from_tensor(w.w2), model.w2)
    assert np.array_equal(np.asarray(w.kernels, dtype=np.float32), model.kernels)
    assert w.w1.shape == [128, 48]


def _grid(rows: list[str]) -> np.ndarray:
    return np.stack([elementary.from_bits(r) for r in rows])


def test_elementary_fixture_reproduces(files: dict[str, bytes]) -> None:
    fx = json.loads(files["fixtures/elementary.json"])
    assert len(fx["cases"]) == 20
    for c in fx["cases"]:
        got = elementary.run(elementary.from_bits(c["initial"]), c["rule"], 15, c["boundary"])
        assert [elementary.to_bits(r) for r in got] == c["rows"]


def test_lifelike_fixture_reproduces(files: dict[str, bytes]) -> None:
    fx = json.loads(files["fixtures/lifelike.json"])
    assert len(fx["cases"]) == 7
    for c in fx["cases"]:
        got = lifelike.run(_grid(c["initial"]), c["rule"], c["steps"], c["boundary"])[-1]
        assert np.array_equal(got, _grid(c["expected"])), c["name"]
    glider = fx["cases"][0]
    assert _grid(glider["expected"]).sum() == 5


def test_elementary_math_fixture_reproduces(files: dict[str, bytes]) -> None:
    from cellauto.analysis import elementary as eca

    cases = json.loads(files["fixtures/elementary_math.json"])["cases"]
    kinds = [c["kind"] for c in cases]
    assert kinds.count("preimage_count") == 10 and kinds.count("garden_of_eden") == 3
    for c in cases:
        if c["kind"] == "preimage_count":
            assert eca.preimage_count(c["rule"], c["config"]) == c["count"]
        elif c["kind"] == "garden_of_eden":
            assert eca.garden_of_eden_count(c["rule"], c["n"]) == c["count"]
        else:
            sim = elementary.run(elementary.single_cell(c["width"]), 90, 16)
            assert [elementary.to_bits(r) for r in sim] == c["rows"]


def test_atlas_dataset_matches_results(files: dict[str, bytes]) -> None:
    atlas = json.loads(files["elementary/atlas.json"])
    assert atlas == json.loads((ROOT / "results" / "eca_atlas.json").read_text())["atlas"]
    assert len(atlas["rows"]) == 256 and atlas["protocol"]["width"] == 251
    ref = next(d for d in json.loads(files["index.json"])["datasets"] if d["id"] == "eca_atlas")
    assert ref["produced_by"] == "cellauto analyze eca-atlas"


def test_rle_fixture_reproduces(files: dict[str, bytes]) -> None:
    fx = json.loads(files["fixtures/rle.json"])
    assert len(fx["cases"]) == 12 + 3
    for c in fx["cases"]:
        grid = parse_rle(c["rle"])
        assert [grid.shape[1], grid.shape[0]] == [c["width"], c["height"]]
        assert np.argwhere(grid).tolist() == c["cells"]
    padded = next(c for c in fx["cases"] if c["name"] == "header_pads_to_declared_size")
    assert (padded["width"], padded["height"]) == (5, 4)


def test_lenia_fixture_reproduces(files: dict[str, bytes]) -> None:
    fx = json.loads(files["fixtures/lenia.json"])
    assert fx["tolerance"] == 1e-5
    orbium, life = fx["cases"]
    p = orbium["params"]
    params = lenia.LeniaParams(R=p["R"], T=p["T"], mu=p["mu"], sigma=p["sigma"],
                               peaks=tuple(p["peaks"]))  # fmt: skip
    state = from_tensor(orbium["initial"])
    for t, expected in zip(orbium["steps"], orbium["states"], strict=True):
        state = lenia.step(state, params)
        assert np.allclose(state, from_tensor(expected), atol=1e-5), t
    got = lifelike.run(_grid(life["initial"]), "B3/S23", 3, "wrap")[-1]
    assert np.array_equal(got, _grid(life["expected"]))
    assert life["exact"] is True


def test_nca_fixture_reproduces(files: dict[str, bytes]) -> None:
    fx = json.loads(files["fixtures/nca.json"])
    case = fx["cases"][0]
    model = load_model(case["model_id"], ROOT)
    x = make_seed(28)
    got = {}
    for t in range(1, 11):
        x = model.step(x, fire_rate=case["fire_rate"])  # type: ignore[assignment]
        got[t] = x
    for t, expected in zip(case["steps"], case["states"], strict=True):
        assert np.allclose(got[t], from_tensor(expected), atol=fx["tolerance"]), t
    hidden = model.hidden_activations(got[case["hidden_state_step"]])
    assert np.allclose(hidden, from_tensor(case["hidden_activations"]), atol=fx["tolerance"])
    assert from_tensor(case["hidden_activations"]).shape == (128, 784)


def test_content_pages(files: dict[str, bytes]) -> None:
    pages = json.loads(files["content/index.json"])
    assert [p["slug"] for p in pages] == ["intro", "elementary", "lifelike", "lenia", "nca",
                                          "inference", "learnability", "about"]  # fmt: skip
    for p in pages:
        assert files[p["path"]].decode().startswith("---\ntitle: ")


def test_check_detects_drift_and_tolerates_float_noise(tmp_path: Path) -> None:
    out = tmp_path / "data"
    export_web(out, ROOT)
    assert check_export(out, ROOT) == []
    # byte-exact files: any change is reported
    (out / "engines.json").write_text("{}\n")
    (out / "stray.txt").write_text("x")
    (out / "elementary" / "rules.json").unlink()
    assert check_export(out, ROOT) == [
        "missing: elementary/rules.json",
        "changed: engines.json",
        "stale: stray.txt",
    ]
    export_web(out, ROOT)
    assert check_export(out, ROOT) == []
    # float fixtures: noise below tolerance is accepted, a real change is not
    path = out / "fixtures" / "nca.json"
    doc = json.loads(path.read_text())
    t = doc["cases"][0]["states"][0]
    arr = from_tensor(t) + np.float32(1e-7)
    from cellauto.export.binary import to_tensor

    doc["cases"][0]["states"][0] = to_tensor(arr).model_dump()
    path.write_text(json.dumps(doc))
    assert check_export(out, ROOT) == []
    doc["cases"][0]["states"][0] = to_tensor(arr + np.float32(0.01)).model_dump()
    path.write_text(json.dumps(doc))
    assert check_export(out, ROOT) == ["changed: fixtures/nca.json"]


def test_export_refuses_to_clobber_a_foreign_directory(tmp_path: Path) -> None:
    (tmp_path / "precious.txt").write_text("keep me")
    with pytest.raises(ValueError):
        export_web(tmp_path, ROOT)
    assert (tmp_path / "precious.txt").read_text() == "keep me"


def test_committed_export_is_up_to_date() -> None:
    assert check_export(ROOT / "web" / "public" / "data", ROOT) == []
