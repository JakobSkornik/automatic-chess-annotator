import json
import sqlite3
from pathlib import Path

from app.research.secondary import analyze_secondary


def _fixture(tmp_path: Path, ratings=True, rubric="v1"):
    exp = tmp_path / "exp"
    (exp / "runs").mkdir(parents=True)
    manifest = {
        "experiment_id": "secondary-test", "cases": [
            {"id": "c1", "source_group": "g1", "category": "quiet", "target_ply": 3},
            {"id": "c2", "source_group": "g2", "category": "endgame", "target_ply": 5}],
        "variants": [{"id": a} for a in ("B00", "B10", "B01", "B11")],
    }
    (exp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    dbpath = tmp_path / "bank.sqlite"
    schema = Path(__file__).parents[1] / "experiments/design/results-bank.sql"
    with sqlite3.connect(dbpath) as db:
        db.executescript(schema.read_text(encoding="utf-8"))
        db.execute("INSERT INTO datasets VALUES ('d','v','m','r',NULL)")
        for case in manifest["cases"]:
            db.execute("INSERT INTO cases VALUES ('d',?,?,?,?,?,?,?,?,?,?)", (case["id"], case["source_group"], "f", "diagnostic", case["category"], "startpos", case["target_ply"], "input", "hash", "{}"))
        db.execute("INSERT INTO experiments VALUES ('secondary-test','d','mh','m','sh','','diagnostic','now')")
        for arm in ("B00", "B10", "B01", "B11"):
            db.execute("INSERT INTO variants VALUES ('secondary-test',?,?,?)", (arm, arm, "{}"))
        for ci, case in enumerate(manifest["cases"], 1):
            for ai, arm in enumerate(("B00", "B10", "B01", "B11")):
                run = f"r-{ci}-{arm}"
                db.execute("INSERT INTO runs (run_id,experiment_id,dataset_hash,case_id,variant_id,replicate,attempt,status,requested_renderer) VALUES (?,?,?,?,?,?,?,?,?)", (run, "secondary-test", "d", case["id"], arm, 0, 0, "completed", "template"))
                sample = f"s-{ci}-{arm}"
                db.execute("INSERT INTO samples VALUES (?,?,?,?,?)", (sample, run, case["target_ply"], sample, "result"))
                if ratings:
                    # B11 is one point better than B00; the other factorial arms
                    # make the interaction and simple effects deterministic.
                    base = 2 + ci
                    score = min(5, base + (1 if arm in ("B10", "B11") else 0) + (1 if arm in ("B01", "B11") else 0))
                    for rater in ("guid", "expert"):
                        db.execute("INSERT INTO ratings VALUES (?,?,?,?,?,?,?,?,?,?,?)", (sample, rater, rubric, score, score, score, score, "sufficient", "no", "", "hash"))
        db.commit()
    return exp, dbpath


def test_secondary_effects_agreement_and_outputs(tmp_path):
    exp, db = _fixture(tmp_path)
    out = tmp_path / "out"
    report = analyze_secondary(exp, db, out, resamples=100)
    assert report["status"] == "available"
    assert (out / "secondary_effects.csv").exists()
    rows = list(__import__("csv").DictReader((out / "secondary_effects.csv").open(encoding="utf-8")))
    b10 = [r for r in rows if r["comparison"] == "B10-B00" and r["metric"] == "usefulness"][0]
    assert float(b10["estimate"]) == 1.0
    agreement = list(__import__("csv").DictReader((out / "rater_agreement.csv").open(encoding="utf-8")))
    assert float(next(r for r in agreement if r["metric"] == "usefulness")["exact_rate"]) == 1.0


def test_secondary_filters_rubric_and_reports_unavailable(tmp_path):
    exp, db = _fixture(tmp_path, ratings=True, rubric="other")
    out = tmp_path / "out"
    report = analyze_secondary(exp, db, out, rubric_version="v1", resamples=10)
    assert report["status"] == "unavailable"
    effects = list(__import__("csv").DictReader((out / "secondary_effects.csv").open(encoding="utf-8")))
    assert all(r["estimate"] == "" for r in effects)


def test_secondary_no_ratings_keeps_headers(tmp_path):
    exp, db = _fixture(tmp_path, ratings=False)
    out = tmp_path / "out"
    report = analyze_secondary(exp, db, out, resamples=10)
    assert report["status"] == "unavailable"
    assert (out / "endpoint_metrics.csv").read_text(encoding="utf-8").startswith("variant_id,category,label")


def test_secondary_pairs_targets_before_equal_source_group_weighting(tmp_path):
    exp, db = _fixture(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE cases SET source_group='g1' WHERE case_id='c2'")
        connection.execute("DELETE FROM ratings WHERE sample_id IN ('s-2-B10','s-2-B11')")
        connection.commit()
    out = tmp_path / "out"
    analyze_secondary(exp, db, out, resamples=20)
    rows = list(__import__("csv").DictReader((out / "secondary_effects.csv").open(encoding="utf-8")))
    row = next(r for r in rows if r["comparison"] == "B10-B00" and r["metric"] == "usefulness")
    assert float(row["estimate"]) == 1.0
    assert row["n_complete_targets"] == "1"
    assert row["n_groups"] == "1"
