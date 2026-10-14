import json

from cellauto.nca.checkpoints import upsert_manifest_entry


def test_upsert_adds_replaces_and_sorts(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([{"id": "b", "v": 1}, {"id": "d", "v": 1}]))
    upsert_manifest_entry(path, {"id": "c", "v": 2})
    upsert_manifest_entry(path, {"id": "b", "v": 3})
    assert json.loads(path.read_text()) == [
        {"id": "b", "v": 3}, {"id": "c", "v": 2}, {"id": "d", "v": 1},
    ]  # fmt: skip
