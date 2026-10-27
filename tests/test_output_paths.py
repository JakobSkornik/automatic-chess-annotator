from pathlib import Path

import pytest

from app.core.output_paths import dated_game_path, find_game_path, iter_game_paths
from app.core import worker


def test_dated_lookup_precedes_legacy_and_inventory_prefers_newest(tmp_path, monkeypatch):
    monkeypatch.setenv("ACA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.chdir(tmp_path)
    job_id = "output-path-test-unique"
    old = dated_game_path(job_id, "2026-10-23")
    newest = dated_game_path(job_id, "2026-10-24")
    for path, value in ((old, "old"), (newest, "new")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    legacy = tmp_path / "data" / "games" / f"{job_id}.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    assert find_game_path(job_id) == newest
    assert [p for p in iter_game_paths() if p.stem == job_id] == [newest]


def test_job_ids_reject_traversal(monkeypatch, tmp_path):
    monkeypatch.setenv("ACA_OUTPUT_DIR", str(tmp_path / "output"))
    with pytest.raises(ValueError):
        dated_game_path("../escape", "2026-10-24")
    with pytest.raises(ValueError):
        find_game_path("a/b")
    with pytest.raises(ValueError):
        dated_game_path("safe", "2026/10/24")


def test_worker_writes_dated_path_and_honors_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ACA_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(worker, "GAMES_OUTPUT_DIR", None)
    dated = worker._job_output_path("worker-path-test")
    assert Path(dated).parent.parent == tmp_path / "output" / "games"
    override = tmp_path / "legacy"
    monkeypatch.setattr(worker, "GAMES_OUTPUT_DIR", str(override))
    assert Path(worker._job_output_path("worker-path-test")) == override / "worker-path-test.json"
