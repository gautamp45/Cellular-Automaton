import json

import pytest
from pydantic import ValidationError

from cellauto.contract import models as m
from cellauto.contract.schema import TOP_LEVEL, dumps, json_schemas, write_schemas


def test_models_are_strict_and_frozen() -> None:
    iv = m.Interval(mean=1.0, ci_low=0.5, ci_high=1.5)
    with pytest.raises(ValidationError):
        m.Interval(mean=1.0, ci_low=0.5, ci_high=1.5, extra=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        iv.mean = 2.0  # type: ignore[misc]


def test_lambda_alias_roundtrip() -> None:
    rule = m.ElementaryRule.model_validate(
        {"rule": 30, "name": None, "description": "d", "table": [0] * 8, "lambda": 0.5}
    )
    assert rule.lambda_ == 0.5
    dumped = rule.model_dump(mode="json", by_alias=True)
    assert dumped["lambda"] == 0.5 and "lambda_" not in dumped
    assert m.ElementaryRule.model_validate(dumped) == rule
    assert m.ElementaryRule(rule=1, name=None, description="d", table=[0] * 8).lambda_ is None


def test_later_phase_fields_default_to_none() -> None:
    life = m.LifeLikeRule(
        id="x", name="x", rulestring="B3/S23", code=1, table=[[0] * 9] * 2, description="d", tags=[]
    )
    assert life.dual is None and life.mean_field is None and life.self_dual is None
    assert m.ElementaryRule(rule=0, name=None, description="d", table=[0] * 8).surjective is None


def test_schemas_cover_every_top_level_model(tmp_path) -> None:
    schemas = json_schemas()
    assert set(schemas) == set(TOP_LEVEL)
    paths = write_schemas(tmp_path)
    assert len(paths) == len(TOP_LEVEL)
    for p in paths:
        assert json.loads(p.read_text())  # valid JSON
    assert "lambda" in json.dumps(schemas["elementary_rules"])
    assert "lambda_" not in json.dumps(schemas["elementary_rules"])


def test_schema_output_is_deterministic(tmp_path) -> None:
    write_schemas(tmp_path / "a")
    write_schemas(tmp_path / "b")
    for p in (tmp_path / "a" / "schema").iterdir():
        assert p.read_bytes() == (tmp_path / "b" / "schema" / p.name).read_bytes()
    assert dumps({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}\n'
