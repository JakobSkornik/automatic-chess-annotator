"""Secondary, rater-facing analyses for the thesis experiment.

This module intentionally sits beside ``app.research.analysis``.  The primary
analysis has a deliberately narrow pre-registered contrast; this report is for
the remaining factorial contrasts, agreement, error labels and endpoint
descriptions.  Missing ratings are kept missing throughout.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

METRICS = ("usefulness", "relevance", "correctness", "concision")
ARMS = ("B00", "B10", "B01", "B11")
COMPARISONS = (("B10", "B00", "B10-B00"), ("B01", "B00", "B01-B00"),
               ("B11", "B10", "B11-B10"), ("interaction", "", "(B11-B10)-(B01-B00)"))
HEADERS = {
    "effects": ["comparison", "metric", "estimate", "ci_low", "ci_high", "n_groups", "n_complete_targets", "missing_pairs", "resamples", "seed"],
    "agreement": ["metric", "n_common_samples", "n_pairs", "exact_count", "exact_rate", "within1_count", "within1_rate", "status"],
    "category": ["category", "variant_id", "metric", "mean", "n_targets", "n_ratings", "complete_targets", "status"],
    "endpoint": ["variant_id", "category", "label", "count", "rate", "n_reviewed", "status"],
}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap(values: list[float], seed: int, n: int) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    estimate = _mean(values)
    if len(values) == 1 or n <= 0:
        return estimate, estimate, estimate
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return estimate, draws[int(0.025 * (n - 1))], draws[int(0.975 * (n - 1))]


def _row_mean(rows: list[sqlite3.Row], metric: str) -> float | None:
    vals = [float(r[metric]) for r in rows if r[metric] is not None]
    return _mean(vals)


def _latest_run_rows(connection: sqlite3.Connection, experiment_id: str, rubric: str) -> tuple[dict[tuple[str, str, int], list[sqlite3.Row]], dict[str, dict[str, Any]]]:
    """Return rating rows for the highest attempt of each cell and case metadata."""
    connection.row_factory = sqlite3.Row
    cases = {str(r["case_id"]): dict(r) for r in connection.execute(
        "SELECT case_id,source_group,category FROM cases JOIN experiments USING(dataset_hash) WHERE experiment_id=?", (experiment_id,))}
    # A sample is identified by its run.  The latest attempt wins, even when a
    # previous attempt completed; failed attempts have no samples and therefore
    # naturally contribute no ratings.
    latest: dict[tuple[str, str, int], tuple[int, str]] = {}
    for r in connection.execute("SELECT run_id,case_id,variant_id,replicate,attempt FROM runs WHERE experiment_id=?", (experiment_id,)):
        key = (str(r["case_id"]), str(r["variant_id"]), int(r["replicate"]))
        candidate = (int(r["attempt"]), str(r["run_id"]))
        if key not in latest or candidate > latest[key]:
            latest[key] = candidate
    selected: dict[tuple[str, str, int], list[sqlite3.Row]] = defaultdict(list)
    sql = ("SELECT s.sample_id,s.target_ply,r.case_id,r.variant_id,r.replicate,"
           "rt.rater_id,rt.rubric_version,rt.correctness,rt.relevance,rt.usefulness,rt.concision,"
           "rt.endpoint_adequacy,rt.major_error "
           "FROM samples s JOIN runs r ON r.run_id=s.run_id "
           "JOIN ratings rt ON rt.sample_id=s.sample_id AND rt.rubric_version=? "
           "WHERE r.experiment_id=?")
    for row in connection.execute(sql, (rubric, experiment_id)):
        key = (str(row["case_id"]), str(row["variant_id"]), int(row["replicate"]))
        chosen = latest.get(key)
        # Find run id for the sample rather than relying on a run_id selected
        # in the projection; the latest-attempt filter is explicit below.
        run_id = connection.execute("SELECT run_id FROM samples WHERE sample_id=?", (row["sample_id"],)).fetchone()[0]
        if chosen and run_id == chosen[1]:
            selected[key].append(row)
    return selected, cases


def _target_values(selected: dict[tuple[str, str, int], list[sqlite3.Row]], metric: str, case_id: str, variant: str) -> tuple[dict[int, float], int]:
    reps: dict[int, float] = {}
    for (cid, arm, rep), rows in selected.items():
        if cid != case_id or arm != variant:
            continue
        # A complete target requires two independent raters for that metric.
        values = [float(r[metric]) for r in rows if r[metric] is not None]
        raters = {str(r["rater_id"]) for r in rows if r[metric] is not None}
        if len(raters) >= 2:
            reps[rep] = _mean(values)  # type: ignore[assignment]
    return reps, len(reps)


def _effect_rows(selected: dict[tuple[str, str, int], list[sqlite3.Row]], cases: dict[str, dict[str, Any]], seed: int, resamples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        specs = [*COMPARISONS]
        for left, right, label in specs:
            # Form the contrast for each target first.  Only then average
            # targets within a source group; this keeps the comparison paired
            # when one target is missing an arm and prevents dict overwrites
            # when a source group contains several positions.
            target_contrasts: dict[str, list[float]] = defaultdict(list)
            complete_targets = 0
            missing = 0
            for cid, case in cases.items():
                arms: dict[str, float] = {}
                for arm in ARMS:
                    reps, _ = _target_values(selected, metric, cid, arm)
                    if reps:
                        arms[arm] = _mean(list(reps.values()))  # type: ignore[assignment]
                if label == "(B11-B10)-(B01-B00)":
                    needed = all(a in arms for a in ("B11", "B10", "B01", "B00"))
                    value = (arms["B11"] - arms["B10"] - arms["B01"] + arms["B00"]) if needed else None
                else:
                    value = arms.get(left) - arms.get(right) if left in arms and right in arms else None
                if value is None:
                    missing += 1
                else:
                    complete_targets += 1
                    target_contrasts[str(case.get("source_group") or cid)].append(value)
            vals = [_mean(values) for values in target_contrasts.values()]
            vals = [value for value in vals if value is not None]
            estimate, low, high = _bootstrap(vals, seed + len(rows), resamples)
            rows.append({"comparison": label, "metric": metric, "estimate": estimate, "ci_low": low, "ci_high": high,
                         "n_groups": len(vals), "n_complete_targets": complete_targets,
                         "missing_pairs": missing, "resamples": resamples, "seed": seed})
    return rows


def _agreement_rows(selected: dict[tuple[str, str, int], list[sqlite3.Row]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        exact = within = pairs = 0
        for sample_rows in selected.values():
            by_rater = {str(r["rater_id"]): r[metric] for r in sample_rows if r[metric] is not None}
            if len(by_rater) < 2:
                continue
            vals = list(by_rater.values())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    pairs += 1; exact += int(vals[i] == vals[j]); within += int(abs(vals[i] - vals[j]) <= 1)
        rows.append({"metric": metric, "n_common_samples": pairs, "n_pairs": pairs, "exact_count": exact,
                     "exact_rate": exact / pairs if pairs else None, "within1_count": within,
                     "within1_rate": within / pairs if pairs else None, "status": "available" if pairs else "unavailable"})
    return rows


def _category_rows(selected: dict[tuple[str, str, int], list[sqlite3.Row]], cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in sorted({str(c.get("category") or "uncategorized") for c in cases.values()}):
        ids = [cid for cid, c in cases.items() if str(c.get("category") or "uncategorized") == category]
        for arm in ARMS:
            for metric in METRICS:
                vals: list[float] = []; complete = 0
                for cid in ids:
                    reps, count = _target_values(selected, metric, cid, arm)
                    if count:
                        complete += 1; vals.append(_mean(list(reps.values())))  # type: ignore[arg-type]
                n_ratings = sum(len(records) for (cid, selected_arm, _rep), records in selected.items() if cid in ids and selected_arm == arm)
                rows.append({"category": category, "variant_id": arm, "metric": metric, "mean": _mean(vals), "n_targets": len(ids),
                             "n_ratings": n_ratings,
                             "complete_targets": complete, "status": "available" if vals else "unavailable"})
    return rows


def _endpoint_rows(selected: dict[tuple[str, str, int], list[sqlite3.Row]], cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    labels = ("insufficient", "sufficient", "extended", "unassessable")
    for arm in ARMS:
        for category in sorted({str(c.get("category") or "uncategorized") for c in cases.values()}):
            vals = [r["endpoint_adequacy"] for (cid, a, _), rs in selected.items() if a == arm and str(cases.get(cid, {}).get("category") or "uncategorized") == category for r in rs if r["endpoint_adequacy"]]
            for label in labels:
                rows.append({"variant_id": arm, "category": category, "label": label, "count": vals.count(label), "rate": vals.count(label) / len(vals) if vals else None, "n_reviewed": len(vals), "status": "available" if vals else "unavailable"})
    return rows


def analyze_secondary(experiment_dir: Path, database: Path, output_dir: Path, rubric_version: str = "v1", seed: int = 20260905, resamples: int = 10000) -> dict[str, Any]:
    experiment_dir, database, output_dir = Path(experiment_dir), Path(database), Path(output_dir)
    manifest_path = experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment_id = str(manifest.get("experiment_id") or "")
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as db:
        selected, cases = _latest_run_rows(db, experiment_id, rubric_version)
    effects = _effect_rows(selected, cases, seed, resamples)
    agreement = _agreement_rows(selected)
    categories = _category_rows(selected, cases)
    endpoints = _endpoint_rows(selected, cases)
    inputs = {"experiment_dir": str(experiment_dir.resolve()), "database": str(database.resolve()), "manifest": str(manifest_path.resolve()),
              "manifest_sha256": _hash_file(manifest_path), "database_sha256": _hash_file(database), "rubric_version": rubric_version,
              "seed": seed, "resamples": resamples, "attempt_policy": "highest attempt per case/variant/replicate", "generated_at": datetime.now(timezone.utc).isoformat()}
    (output_dir / "secondary-inputs.json").write_text(json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "secondary_effects.csv", effects, HEADERS["effects"])
    _write_csv(output_dir / "rater_agreement.csv", agreement, HEADERS["agreement"])
    _write_csv(output_dir / "category_metrics.csv", categories, HEADERS["category"])
    _write_csv(output_dir / "endpoint_metrics.csv", endpoints, HEADERS["endpoint"])
    available = any(r["status"] == "available" for r in agreement) or any(r["status"] == "available" for r in categories)
    summary = {"status": "available" if available else "unavailable", "experiment_id": experiment_id, "rubric_version": rubric_version,
               "case_count": len(cases), "rated_cell_count": len(selected), "comparisons": effects, "agreement": agreement,
               "outputs": {name: str(output_dir / name) for name in ("secondary_effects.csv", "rater_agreement.csv", "category_metrics.csv", "endpoint_metrics.csv")}, "inputs": inputs}
    (output_dir / "secondary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
