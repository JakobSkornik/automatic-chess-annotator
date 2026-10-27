import csv
import json
import sqlite3
import hashlib
from pathlib import Path

import pytest

from app.research.bank import ingest, import_ratings, packet, sha256_file


def _experiment(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    (root / "runs" / "run-1").mkdir(parents=True)
    (root / "snapshots").mkdir()
    manifest = {
        "experiment_id": "exp-1", "dataset_hash": "dataset-1", "source_hash": "source-1", "track": "component",
        "variants": [{"id": "B00", "config_hash": "cfg-1"}],
        "cases": [{"id": "case-1", "category": "quiet_maneuver", "source_group": "g1", "family_group": "f1", "split": "pilot", "initial_fen": "startpos", "target_ply": 3, "input_hash": "input-1"}],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    snapshot_body = {"engine_config_hash": "engine-1"}
    snapshot_hash = hashlib.sha256(json.dumps(snapshot_body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    (root / "snapshots" / "case-1.json").write_text(json.dumps({"snapshot_hash": snapshot_hash, **snapshot_body}), encoding="utf-8")
    result = {"run_id": "run-1", "experiment_id": "exp-1", "case_id": "case-1", "variant_id": "B00", "replicate": 0, "attempt": 0, "status": "completed", "requested_renderer": "template", "actual_renderer": "template", "seconds": 0.25, "snapshot_hash": snapshot_hash, "audit": {"checks": {"provenance": {"passed": 1, "total": 1}}}, "samples": [{"sample_id": "sample-1", "target_ply": 3, "fen_before": "fen", "move_san": "Nf3", "comment": "A sound developing move."}]}
    (root / "runs" / "run-1" / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return root


def test_ingest_is_repeatable_and_indexes_failures(tmp_path):
    root = _experiment(tmp_path)
    (root / "schedule.json").write_text(json.dumps([{"run_id": "run-1"}]))
    database = tmp_path / "results.sqlite"
    first = ingest(root, database)
    second = ingest(root, database)
    assert first["runs"] == 1 and first["samples"] == 1
    assert second["runs"] == 1  # the report counts observed artifacts, not only new rows
    with sqlite3.connect(database) as connection:
        assert connection.execute("select count(*) from samples").fetchone()[0] == 1
        assert connection.execute("select passed,total from checks").fetchone() == (1, 1)


def test_ingest_rejects_changed_result_artifact(tmp_path):
    root = _experiment(tmp_path)
    database = tmp_path / "results.sqlite"
    ingest(root, database)
    result = root / "runs" / "run-1" / "result.json"
    result.write_text(result.read_text(encoding="utf-8").replace("sound", "unsafe"), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable result artifact"):
        ingest(root, database)


def test_packet_contains_no_ratings_and_import_preserves_unknowns(tmp_path):
    root = _experiment(tmp_path)
    database = tmp_path / "results.sqlite"
    ingest(root, database)
    output = tmp_path / "packet"
    packet_info = packet(root, output, seed=7)
    with (output / "ratings.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert packet_info["samples"] == 1
    assert all(row["correctness"] == "" for row in rows)
    rows[0].update({"rater_id": "rater-a", "correctness": "5", "relevance": "", "usefulness": "4", "concision": "3", "endpoint_adequacy": "sufficient", "major_error": "no", "notes": "clear"})
    with (output / "submitted.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0]])
        writer.writeheader(); writer.writerows(rows)
    imported = import_ratings(database, output / "submitted.csv", output / "private_key.csv")
    assert imported["imported"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("select correctness,relevance,usefulness from ratings").fetchone() == (5, None, 4)
    assert list((tmp_path / "received-ratings").glob("*.csv"))


def test_import_rejects_unknown_and_duplicate_ids(tmp_path):
    root = _experiment(tmp_path)
    database = tmp_path / "results.sqlite"
    ingest(root, database)
    output = tmp_path / "packet"
    packet(root, output)
    with (output / "bad.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anonymous_id", "rater_id", *["correctness", "relevance", "usefulness", "concision"]])
        writer.writeheader(); writer.writerow({"anonymous_id": "unknown", "rater_id": "r", "correctness": "1", "relevance": "1", "usefulness": "1", "concision": "1"})
    with pytest.raises(ValueError, match="unknown anonymous_id"):
        import_ratings(database, output / "bad.csv", output / "private_key.csv")
