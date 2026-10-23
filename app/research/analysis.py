"""Frozen experiment analysis and thesis export.

The experiment directory is the immutable source of truth.  This module only
reads it (and, optionally, the rebuildable results-bank SQLite index) and
writes a new analysis directory.  In particular, absent ratings are never
turned into a guessed quality score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(value: Any) -> str:
    if isinstance(value, bytes):
        raw = value
    else:
        # Match the frozen runner's digest() representation so manifest and
        # result sidecars can be verified without rewriting their JSON.
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _num(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    xs = list(values)
    return statistics.mean(xs) if xs else None


def _median(values: Iterable[float]) -> float | None:
    xs = list(values)
    return statistics.median(xs) if xs else None


def _checks(samples: list[dict[str, Any]]) -> tuple[int, int]:
    passed = total = 0
    for sample in samples:
        checks = ((sample.get("audit") or {}).get("checks") or {})
        for value in checks.values():
            if isinstance(value, dict):
                passed += int(value.get("passed", 0) or 0)
                total += int(value.get("total", 0) or 0)
    return passed, total


def _claim_count(sample: dict[str, Any]) -> int:
    facts = sample.get("facts") or {}
    claims = facts.get("claims")
    if isinstance(claims, dict):
        return sum(len(v) for v in claims.values() if isinstance(v, list))
    if isinstance(claims, list):
        return len(claims)
    return int(sample.get("claim_count", 0) or 0)


def _sample_key(case_id: str, sample: dict[str, Any], run: dict[str, Any]) -> tuple[str, int, int]:
    return case_id, int(sample.get("target_ply", 0) or 0), int(run.get("replicate", 0) or 0)


def _bootstrap(values: list[float], groups: list[str], seed: int, resamples: int) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values):
        grouped[group].append(value)
    means = [statistics.mean(v) for v in grouped.values() if v]
    if not means:
        return {"n_groups": 0, "mean": None, "ci95": None, "resamples": resamples, "seed": seed}
    point = statistics.mean(means)
    if len(means) < 2 or resamples <= 0:
        return {"n_groups": len(means), "mean": point, "ci95": None, "resamples": resamples, "seed": seed}
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        draws.append(statistics.mean(rng.choice(means) for _ in means))
    draws.sort()
    lo = draws[max(0, int(0.025 * len(draws)))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"n_groups": len(means), "mean": point, "ci95": [lo, hi], "resamples": resamples, "seed": seed}


def _load_ratings(
    database: Path | None,
    experiment_id: str | None = None,
    rubric_version: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load ratings grouped by sample, with no rater rows overwritten.

    The production bank has experiment and rubric metadata.  Small legacy
    fixtures may only contain a ratings table; those remain readable, while
    real banks are filtered to the requested experiment and reject ambiguous
    rubric versions unless the caller selects one explicitly.
    """
    if database is None or not database.exists():
        return {}
    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        rating_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(ratings)")}
        run_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")}
        has_bank = {"sample_id", "rater_id", "usefulness"}.issubset(rating_columns) and {"run_id", "experiment_id"}.issubset(run_columns)
        if has_bank:
            versions_query = (
                "SELECT DISTINCT rt.rubric_version FROM ratings rt "
                "JOIN samples s ON s.sample_id=rt.sample_id JOIN runs r ON r.run_id=s.run_id "
                "WHERE r.experiment_id=? AND rt.usefulness IS NOT NULL"
            )
            versions = {str(row[0]) for row in connection.execute(versions_query, (experiment_id,)).fetchall() if row[0] is not None}
            if rubric_version is None and len(versions) > 1:
                raise ValueError(f"multiple rubric versions for {experiment_id}: {sorted(versions)}; select one explicitly")
            selected = rubric_version or (next(iter(versions)) if versions else None)
            where = "r.experiment_id=? AND rt.usefulness IS NOT NULL"
            params: list[Any] = [experiment_id]
            if selected is not None:
                where += " AND rt.rubric_version=?"
                params.append(selected)
            rows = connection.execute(
                "SELECT rt.sample_id, rt.rater_id, rt.usefulness, rt.correctness, rt.relevance, rt.concision, rt.rubric_version "
                "FROM ratings rt JOIN samples s ON s.sample_id=rt.sample_id JOIN runs r ON r.run_id=s.run_id WHERE " + where,
                params,
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT sample_id, rater_id, usefulness, correctness, relevance, concision "
                "FROM ratings WHERE usefulness IS NOT NULL"
            ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        connection.close()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(dict(row))
    return dict(grouped)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_rubric(database: Path | None, experiment_id: str, requested: str | None) -> str | None:
    if database is None or not database.exists():
        return requested
    connection = sqlite3.connect(str(database))
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(ratings)")}
        run_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")}
        if "rubric_version" not in columns or not {"run_id", "experiment_id"}.issubset(run_columns):
            return requested
        versions = {str(row[0]) for row in connection.execute(
            "SELECT DISTINCT rt.rubric_version FROM ratings rt JOIN samples s ON s.sample_id=rt.sample_id "
            "JOIN runs r ON r.run_id=s.run_id WHERE r.experiment_id=? AND rt.usefulness IS NOT NULL",
            (experiment_id,),
        ) if row[0] is not None}
        if requested is None and len(versions) > 1:
            raise ValueError(f"multiple rubric versions for {experiment_id}: {sorted(versions)}; select one explicitly")
        return requested or (next(iter(versions)) if versions else None)
    finally:
        connection.close()


def _persist_analysis(database: Path, summary: dict[str, Any], input_lock_hash: str, config_hash: str, code_hash: str, output: Path) -> None:
    connection = sqlite3.connect(str(database))
    try:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='analyses'").fetchone()
        if exists:
            connection.execute(
                "INSERT INTO analyses (analysis_id, experiment_id, input_lock_hash, config_hash, code_hash, output_path, created_at) VALUES (?,?,?,?,?,?,?)",
                (summary["analysis_id"], summary["experiment_id"], input_lock_hash, config_hash, code_hash, str(output), summary["generated_at"]),
            )
            connection.commit()
    finally:
        connection.close()


def analyze(
    experiment_dir: Path,
    database: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 20260905,
    resamples: int = 10000,
    rubric_version: str | None = None,
) -> dict[str, Any]:
    """Analyze a frozen experiment and export reproducible thesis artifacts."""
    experiment_dir = Path(experiment_dir)
    manifest_path = experiment_dir / "manifest.json"
    manifest = _read(manifest_path)
    manifest_sidecar = experiment_dir / "manifest.sha256"
    if manifest_sidecar.exists() and manifest_sidecar.read_text(encoding="utf-8").strip() != _hash(manifest):
        raise ValueError("manifest.sha256 does not match manifest.json")
    experiment_id = str(manifest.get("experiment_id") or experiment_dir.name)
    analysis_id = f"{experiment_id}-analysis-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{_hash(manifest)[:8]}"
    output = Path(output_dir) if output_dir else experiment_dir.parent / "analyses" / analysis_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    variants = {str(v.get("id")): v for v in manifest.get("variants", []) if v.get("id")}
    cases = {str(c.get("id")): c for c in manifest.get("cases", []) if c.get("id")}
    run_files = sorted((experiment_dir / "runs").glob("*/result.json")) if (experiment_dir / "runs").exists() else []
    all_run_records: list[dict[str, Any]] = []
    samples_by_key: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for path in run_files:
        try:
            result = _read(path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append({"run_id": path.parent.name, "error": f"invalid result: {exc}"})
            continue
        result_sidecar = path.with_suffix(".sha256")
        if result_sidecar.exists() and result_sidecar.read_text(encoding="utf-8").strip() != _hash(result):
            raise ValueError(f"{result_sidecar} does not match {path}")
        run = dict(result)
        run["run_id"] = str(run.get("run_id") or path.parent.name)
        run["case_id"] = str(run.get("case_id") or "")
        run["variant_id"] = str(run.get("variant_id") or "")
        all_run_records.append(run)
        status = str(run.get("status") or "failed")
        if status != "completed":
            failures.append({"run_id": run["run_id"], "case_id": run["case_id"], "variant_id": run["variant_id"], "status": status, "error": run.get("error", "")})
        for sample in run.get("samples") or []:
            key = (run["case_id"], int(sample.get("target_ply", 0) or 0), run["variant_id"], int(run.get("replicate", 0) or 0))
            samples_by_key[key] = {**sample, "_run": run}

    # Attempts are append-only.  Only the highest attempt for a cell is used
    # for metrics and samples; every attempt remains in the input lock and in
    # the failure report so a retry cannot silently rewrite history.
    latest_ids: dict[tuple[str, str, int], tuple[int, str]] = {}
    for run in all_run_records:
        key = (run["case_id"], run["variant_id"], int(run.get("replicate", 0) or 0))
        candidate = (int(run.get("attempt", 0) or 0), str(run["run_id"]))
        if key not in latest_ids or candidate > latest_ids[key]:
            latest_ids[key] = candidate
    selected_ids = {run_id for _, run_id in latest_ids.values()}
    run_records = [run for run in all_run_records if str(run.get("run_id")) in selected_ids]
    # Rebuild the sample index from selected runs.  Building it while reading
    # files would let filesystem ordering decide whether an older retry wins.
    samples_by_key.clear()
    for run in run_records:
        for sample in run.get("samples") or []:
            key = (run["case_id"], int(sample.get("target_ply", 0) or 0), run["variant_id"], int(run.get("replicate", 0) or 0))
            samples_by_key[key] = {**sample, "_run": run}

    # The manifest defines the planned matrix.  Register absent result files
    # as failures so coverage cannot look complete merely because a runner
    # omitted a case.  This creates no result artifact and keeps the raw bank
    # untouched; it is an analysis-level missing-result record.
    planned_replicates = int(((manifest.get("execution") or {}).get("core_text_repetitions") or 1))
    if cases and variants:
        existing = {(str(r.get("case_id")), str(r.get("variant_id")), int(r.get("replicate", 0) or 0), int(r.get("attempt", 0) or 0)) for r in all_run_records}
        for case_id in cases:
            for variant_id in variants:
                for replicate in range(planned_replicates):
                    identity = (case_id, variant_id, replicate, 0)
                    if identity not in existing:
                        missing = {"run_id": f"missing-{case_id}-{variant_id}-r{replicate}", "case_id": case_id,
                                   "variant_id": variant_id, "replicate": replicate, "attempt": 0,
                                   "status": "failed", "error": "missing result artifact"}
                        run_records.append(missing)
                        failures.append({"run_id": missing["run_id"], "case_id": case_id, "variant_id": variant_id,
                                         "status": "failed", "error": missing["error"]})

    variant_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for variant_id in variants or sorted({r["variant_id"] for r in run_records}):
        runs = [r for r in run_records if r.get("variant_id") == variant_id]
        seconds = [x for x in (_num(r.get("seconds")) for r in runs) if x is not None]
        samples = [s for s in samples_by_key.values() if s["_run"].get("variant_id") == variant_id]
        passed, total = _checks(samples)
        claims = [_claim_count(s) for s in samples]
        horizons = [len(s["facts"]["display_line"]["line_uci"]) for s in samples
                    if isinstance((s.get("facts") or {}).get("display_line"), dict)
                    and "line_uci" in s["facts"]["display_line"]]
        paired = [(s, samples_by_key.get((s["_run"]["case_id"], int(s["target_ply"]), "B00", int(s["_run"].get("replicate", 0)))))
                  for s in samples]
        paired = [(s, baseline) for s, baseline in paired if baseline is not None]
        completed = sum(str(r.get("status")) == "completed" for r in runs)
        row = {"variant_id": variant_id, "runs": len(runs), "completed": completed, "failed": len(runs) - completed,
               "mean_seconds": _mean(seconds), "median_seconds": _median(seconds), "claims": sum(claims),
               "mean_claims": _mean(claims), "mean_horizon": _mean(horizons), "checks_passed": passed,
               "checks_total": total, "structural_check_rate": (passed / total if total else None),
               "paired_samples_vs_B00": len(paired),
               "changed_comments_vs_B00": sum(s.get("comment") != b.get("comment") for s, b in paired),
               "changed_horizons_vs_B00": sum((s.get("facts", {}).get("display_line") or {}).get("line_uci") !=
                   (b.get("facts", {}).get("display_line") or {}).get("line_uci") for s, b in paired)}
        variant_rows.append(row)
        for name, value in row.items():
            if name != "variant_id":
                metric_rows.append({"variant_id": variant_id, "metric": name, "value": value})

    selected_rubric = _selected_rubric(database, experiment_id, rubric_version)
    ratings = _load_ratings(database, experiment_id, selected_rubric)
    # Map the bank's sample IDs to experiment samples.  Result samples are
    # authoritative; common IDs are preferred, then the bank's sample table is
    # used below when available.
    rated: dict[tuple[str, int, str, int], list[float]] = defaultdict(list)
    selected_sample_ids = {str(sample.get("sample_id")) for sample in samples_by_key.values()}
    if database and database.exists():
        connection = sqlite3.connect(str(database))
        connection.row_factory = sqlite3.Row
        try:
            run_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")}
            if {"run_id", "experiment_id"}.issubset(run_columns):
                rows = connection.execute(
                    "SELECT s.sample_id, s.target_ply, r.variant_id, r.case_id, r.replicate "
                    "FROM samples s JOIN runs r ON r.run_id=s.run_id WHERE r.experiment_id=?",
                    (experiment_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT s.sample_id, s.target_ply, r.variant_id, r.case_id, r.replicate "
                    "FROM samples s JOIN runs r ON r.run_id=s.run_id"
                ).fetchall()
            for row in rows:
                if str(row["sample_id"]) not in selected_sample_ids:
                    continue
                for rating in ratings.get(str(row["sample_id"]), []):
                    if rating.get("usefulness") is not None:
                        rated[(str(row["case_id"]), int(row["target_ply"]), str(row["variant_id"]), int(row["replicate"] or 0))].append(float(rating["usefulness"]))
        except sqlite3.Error:
            pass
        finally:
            connection.close()

    effect_rows: list[dict[str, Any]] = []
    primary_values: list[float] = []
    primary_groups: list[str] = []
    teaching_cases = [c for c in cases.values() if int(c.get("target_ply", 0) or 0) > 0 and c.get("teaching_target", True) is not False]
    for case in teaching_cases:
        case_id, target = str(case["id"]), int(case["target_ply"])
        target_differences: list[float] = []
        for replicate in sorted({int(r.get("replicate", 0) or 0) for r in run_records} or {0}):
            b00 = rated.get((case_id, target, "B00", replicate), [])
            b11 = rated.get((case_id, target, "B11", replicate), [])
            # A missing delivered explanation is a prespecified failure floor
            # only when ratings exist for this analysis.  An absent second
            # rater remains missing; it must never be silently scored as one.
            b00_missing = (case_id, target, "B00", replicate) not in samples_by_key
            b11_missing = (case_id, target, "B11", replicate) not in samples_by_key
            b00_score = 1.0 if b00_missing and ratings else (statistics.mean(b00) if len(b00) >= 2 else None)
            b11_score = 1.0 if b11_missing and ratings else (statistics.mean(b11) if len(b11) >= 2 else None)
            complete = b00_score is not None and b11_score is not None
            if complete:
                diff = b11_score - b00_score
                target_differences.append(diff)
                effect_rows.append({"case_id": case_id, "target_ply": target, "replicate": replicate,
                                    "source_group": str(case.get("source_group") or case_id), "b00_usefulness": b00_score,
                                    "b11_usefulness": b11_score, "difference": diff, "eligible": 1, "raters_b00": len(b00), "raters_b11": len(b11),
                                    "b00_floor": int(b00_missing), "b11_floor": int(b11_missing)})
            else:
                effect_rows.append({"case_id": case_id, "target_ply": target, "replicate": replicate,
                                    "source_group": str(case.get("source_group") or case_id), "b00_usefulness": None,
                                    "b11_usefulness": None, "difference": None, "eligible": 0, "raters_b00": len(b00), "raters_b11": len(b11),
                                    "b00_floor": int(b00_missing), "b11_floor": int(b11_missing)})
        # Replicates are an operational repeat of one target.  Average them
        # before source-group bootstrap weighting so a target with more
        # repetitions cannot dominate its source game.
        if target_differences:
            primary_values.append(statistics.mean(target_differences))
            primary_groups.append(str(case.get("source_group") or case_id))
    bootstrap = _bootstrap(primary_values, primary_groups, seed, resamples) if primary_values else {"n_groups": 0, "mean": None, "ci95": None, "resamples": resamples, "seed": seed}
    human = {"available": bool(rated), "primary": bootstrap, "complete_pairs": sum(x.get("eligible") == 1 for x in effect_rows),
             "incomplete_pairs": sum(x.get("eligible") == 0 for x in effect_rows),
             "note": "Human ratings unavailable; no quality score inferred." if not rated else "Primary pairs require two raters per B00 and B11 sample."}

    generated_at = datetime.now(timezone.utc).isoformat()
    config = {"seed": seed, "resamples": resamples, "rubric_version": selected_rubric, "primary_variants": ["B00", "B11"],
              "teaching_target_default": True}
    rating_records = sorted((record for records in ratings.values() for record in records), key=lambda record: (str(record.get("sample_id")), str(record.get("rater_id"))))
    input_lock = {"manifest": {"path": str(manifest_path), "sha256": _file_hash(manifest_path)},
                  "runs": [{"path": str(path), "sha256": _file_hash(path)} for path in run_files],
                  "runs_all_attempts": [{"run_id": str(run.get("run_id")), "case_id": str(run.get("case_id")),
                                          "variant_id": str(run.get("variant_id")), "replicate": int(run.get("replicate", 0) or 0),
                                          "attempt": int(run.get("attempt", 0) or 0), "selected": str(run.get("run_id")) in selected_ids}
                                         for run in all_run_records],
                  "ratings": rating_records, "rubric_version": selected_rubric,
                  "config": config, "analysis_code_sha256": _file_hash(Path(__file__))}
    input_lock_hash = _hash(input_lock)
    (output / "analysis-inputs.json").write_text(json.dumps({**input_lock, "input_lock_hash": input_lock_hash}, indent=2, sort_keys=True, default=str), encoding="utf-8")
    config_hash = _hash(config)
    code_hash = _file_hash(Path(__file__))
    summary = {"analysis_id": analysis_id, "experiment_id": experiment_id, "manifest_hash": _hash(manifest),
               "generated_at": generated_at, "run_count": len(run_records),
               "sample_count": len(samples_by_key), "variant_metrics": variant_rows, "human_ratings": human,
               "failures": {"count": len(failures)}, "sources": {"database": str(database) if database else None},
               "config": config, "input_lock_hash": input_lock_hash, "config_hash": config_hash, "code_hash": code_hash}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output / "metrics.csv", metric_rows, ["variant_id", "metric", "value"])
    _write_csv(output / "paired_effects.csv", effect_rows, ["case_id", "target_ply", "replicate", "source_group", "b00_usefulness", "b11_usefulness", "difference", "eligible", "raters_b00", "raters_b11", "b00_floor", "b11_floor"])
    _write_csv(output / "failures.csv", failures, ["run_id", "case_id", "variant_id", "status", "error"])
    examples = []
    for key, sample in sorted(samples_by_key.items()):
        examples.append({"case_id": key[0], "target_ply": key[1], "variant_id": key[2], "replicate": key[3],
                         "comment": sample.get("comment") or sample.get("text"), "facts": sample.get("facts"), "audit": sample.get("audit")})
    (output / "examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_markdown(output / "results.md", summary, effect_rows, variant_rows)
    _write_latex(output / "thesis-tables.tex", variant_rows, bootstrap)
    if database and database.exists():
        _persist_analysis(database, summary, input_lock_hash, config_hash, code_hash, output)
    return {**summary, "output_dir": str(output), "metrics_path": str(output / "metrics.csv")}


def _write_markdown(path: Path, summary: dict[str, Any], effects: list[dict[str, Any]], variants: list[dict[str, Any]]) -> None:
    lines = [f"# Analysis {summary['analysis_id']}", "", f"Runs: {summary['run_count']}  ", f"Samples: {summary['sample_count']}", "", "## Automatic metrics", "", "| Variant | Runs | Completed | Failed | Mean seconds | Claims | Checks |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in variants:
        rate = f"{row['checks_passed']}/{row['checks_total']}" if row["checks_total"] else "n/a"
        lines.append(f"| {row['variant_id']} | {row['runs']} | {row['completed']} | {row['failed']} | {row['mean_seconds'] if row['mean_seconds'] is not None else 'n/a'} | {row['claims']} | {rate} |")
    human = summary["human_ratings"]
    lines += ["", "## Human ratings", "", human["note"]]
    primary = human["primary"]
    if primary["mean"] is not None:
        lines.append(f"Primary B11 minus B00 usefulness: {primary['mean']:.3f}; 95% grouped bootstrap CI: {primary['ci95']} ({primary['n_groups']} source groups).")
    else:
        lines.append("No eligible primary paired ratings were available.")
    lines += [f"Complete paired samples: {human['complete_pairs']}; incomplete: {human['incomplete_pairs']}.", "", "Failed runs remain in failures.csv and are included in coverage counts."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, variants: list[dict[str, Any]], primary: dict[str, Any]) -> None:
    row_end = "\\\\"
    lines = ["% Generated by app.research.analysis; do not edit.", "\\begin{tabular}{lrrrrr}", "Variant & Runs & Complete & Failed & Claims & Checks " + row_end, "\\hline"]
    for row in variants:
        lines.append(f"{row['variant_id']} & {row['runs']} & {row['completed']} & {row['failed']} & {row['claims']} & {row['checks_passed']}/{row['checks_total']} " + row_end)
    lines += ["\\end{tabular}", "", "% Primary paired usefulness (B11-B00): " + ("unavailable" if primary["mean"] is None else f"{primary['mean']:.4f}")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
