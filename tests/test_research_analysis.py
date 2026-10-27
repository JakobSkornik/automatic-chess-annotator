import json
import sqlite3
from pathlib import Path

from app.research.analysis import analyze


def _fixture(tmp_path: Path, rated: bool = False):
    experiment = tmp_path / "experiment"
    (experiment / "runs" / "b00").mkdir(parents=True)
    (experiment / "runs" / "b11").mkdir(parents=True)
    manifest = {
        "experiment_id": "fixture-01",
        "dataset_hash": "dataset",
        "source_hash": "source",
        "track": "component",
        "variants": [{"id": "B00"}, {"id": "B11"}],
        "cases": [{"id": "c1", "source_group": "game-a", "target_ply": 5}],
    }
    (experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for directory, variant, text in [("b00", "B00", "baseline"), ("b11", "B11", "improved")]:
        result = {"run_id": directory, "case_id": "c1", "variant_id": variant, "replicate": 0,
                  "status": "completed", "seconds": 1.0, "samples": [{
                      "sample_id": f"{variant}-c1-5", "target_ply": 5, "comment": text,
                      "facts": {"claims": {"played": [{"id": "x"}]}, "display_line": {"line_uci": ["e2e4", "e7e5"]}},
                      "audit": {"checks": {"evidence": {"passed": 2, "total": 2}}},
                  }]}
        (experiment / "runs" / directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if not rated:
        return experiment, None
    database = tmp_path / "results.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript("""
      CREATE TABLE cases(case_id TEXT PRIMARY KEY, source_group TEXT);
      CREATE TABLE runs(run_id TEXT PRIMARY KEY, experiment_id TEXT, case_id TEXT, variant_id TEXT, replicate INTEGER);
      CREATE TABLE samples(sample_id TEXT PRIMARY KEY, run_id TEXT, target_ply INTEGER);
      CREATE TABLE ratings(sample_id TEXT, rater_id TEXT, rubric_version TEXT, usefulness REAL, correctness REAL, relevance REAL, concision REAL);
    """)
    connection.execute("INSERT INTO cases VALUES ('c1','game-a')")
    for run, variant in [("b00", "B00"), ("b11", "B11")]:
        connection.execute("INSERT INTO runs VALUES (?,?,?, ?,0)", (run, "fixture-01", "c1", variant))
        connection.execute("INSERT INTO samples VALUES (?,?,5)", (f"{variant}-c1-5", run))
    connection.executemany("INSERT INTO ratings VALUES (?,?,?,?,?,?,?)", [("B00-c1-5", "r1", "v1", 2, 2, 2, 2), ("B00-c1-5", "r2", "v1", 4, 4, 4, 4), ("B11-c1-5", "r1", "v1", 4, 4, 4, 4), ("B11-c1-5", "r2", "v1", 5, 5, 5, 5)])
    connection.commit()
    connection.close()
    return experiment, database


def test_analysis_exports_automatic_metrics_without_human_scores(tmp_path):
    experiment, database = _fixture(tmp_path)
    result = analyze(experiment, database, tmp_path / "analysis", resamples=20)
    assert result["human_ratings"]["available"] is False
    assert result["human_ratings"]["primary"]["mean"] is None
    assert result["variant_metrics"][0]["mean_horizon"] == 2
    assert result["variant_metrics"][1]["changed_comments_vs_B00"] == 1
    assert (tmp_path / "analysis" / "metrics.csv").exists()
    assert (tmp_path / "analysis" / "thesis-tables.tex").exists()


def test_analysis_computes_paired_usefulness_by_source_group(tmp_path):
    experiment, database = _fixture(tmp_path, rated=True)
    result = analyze(experiment, database, tmp_path / "analysis", seed=4, resamples=50)
    primary = result["human_ratings"]["primary"]
    assert primary["mean"] == 1.5
    assert primary["n_groups"] == 1
    assert result["human_ratings"]["complete_pairs"] == 1


def test_analysis_rejects_mixed_rubric_versions(tmp_path):
    experiment, database = _fixture(tmp_path, rated=True)
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO ratings VALUES (?,?,?,?,?,?,?)", ("B00-c1-5", "r3", "v2", 3, 3, 3, 3))
    connection.commit()
    connection.close()
    try:
        analyze(experiment, database, tmp_path / "analysis")
    except ValueError as exc:
        assert "multiple rubric versions" in str(exc)
    else:
        raise AssertionError("mixed rubric versions must be rejected")


def test_analysis_locks_inputs_and_excludes_controls_from_primary(tmp_path):
    experiment, database = _fixture(tmp_path, rated=True)
    manifest = json.loads((experiment / "manifest.json").read_text(encoding="utf-8"))
    manifest["cases"].append({"id": "control", "source_group": "control-game", "target_ply": 6, "teaching_target": False})
    (experiment / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = analyze(experiment, database, tmp_path / "analysis")
    lock = json.loads((tmp_path / "analysis" / "analysis-inputs.json").read_text(encoding="utf-8"))
    assert lock["input_lock_hash"] == result["input_lock_hash"]
    assert result["human_ratings"]["complete_pairs"] == 1


def test_analysis_rejects_modified_result_sidecar(tmp_path):
    experiment, _ = _fixture(tmp_path)
    result_path = experiment / "runs" / "b00" / "result.json"
    result_path.with_suffix(".sha256").write_text("wrong", encoding="utf-8")
    try:
        analyze(experiment, output_dir=tmp_path / "analysis")
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("modified result sidecars must be rejected")


def test_analysis_uses_latest_retry_and_locks_all_attempts(tmp_path):
    experiment, _ = _fixture(tmp_path)
    retry = experiment / "runs" / "b00-retry"
    retry.mkdir()
    (retry / "result.json").write_text(json.dumps({
        "run_id": "b00-retry", "case_id": "c1", "variant_id": "B00", "replicate": 0, "attempt": 1,
        "status": "completed", "seconds": 2.0, "samples": [{
            "sample_id": "B00-c1-5-retry", "target_ply": 5, "comment": "latest",
            "facts": {"claims": {"played": [{"id": "x"}]}, "display_line": {"line_uci": ["e2e4"]}},
            "audit": {"checks": {"evidence": {"passed": 1, "total": 1}}},
        }]
    }), encoding="utf-8")
    output = tmp_path / "analysis"
    result = analyze(experiment, output_dir=output)
    examples = json.loads((output / "examples.json").read_text(encoding="utf-8"))
    b00 = next(item for item in examples if item["variant_id"] == "B00")
    assert b00["comment"] == "latest"
    lock = json.loads((output / "analysis-inputs.json").read_text(encoding="utf-8"))
    assert {item["run_id"] for item in lock["runs_all_attempts"]} >= {"b00", "b00-retry"}
    assert next(item for item in lock["runs_all_attempts"] if item["run_id"] == "b00")["selected"] is False
    assert result["variant_metrics"][0]["runs"] == 1
