"""Append-only index and blinded rating exchange for thesis experiments.

The JSON/CSV files remain the source of truth.  SQLite is only an index and
can be rebuilt from the experiment directory.  This module deliberately does
not calculate research metrics or invent ratings for missing observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import secrets
import shutil
import sqlite3
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "experiments" / "design" / "results-bank.sql"
SCORES = ("correctness", "relevance", "usefulness", "concision")
EXTRAS = ("endpoint_adequacy", "major_error", "notes")
NULLS = {"", "na", "n/a", "null", "unknown"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _runner_digest(value: Any) -> str:
    """Digest convention used by app.research.runner.digest."""
    return sha256_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return _runner_digest(manifest)


def _safe_id(value: Any, label: str) -> str:
    value = str(value)
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone() is None:
        connection.executescript(SCHEMA.read_text(encoding="utf-8"))
    return connection


def _required(manifest: dict[str, Any], name: str) -> Any:
    value = manifest.get(name)
    if value in (None, ""):
        raise ValueError(f"manifest is missing {name}")
    return value


def _artifact_copy(source: Path, database: Path, experiment_id: str, kind: str) -> tuple[str, str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    relative = Path("bank-artifacts") / experiment_id / kind / f"{digest[:16]}-{source.name}"
    target = database.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256_file(target) != digest:
        raise ValueError(f"artifact hash collision: {target}")
    if not target.exists():
        shutil.copyfile(source, target)
    return str(relative).replace("\\", "/"), digest


def _ensure_same(connection: sqlite3.Connection, table: str, key_column: str, key: str, values: dict[str, Any]) -> None:
    row = connection.execute(f"SELECT * FROM {table} WHERE {key_column} = ?", (key,)).fetchone()
    if row is None:
        return
    columns = [item[1] for item in connection.execute(f"PRAGMA table_info({table})")]
    existing = dict(zip(columns, row))
    for column, value in values.items():
        if column in existing and existing[column] != value:
            raise ValueError(f"immutable {table} conflict for {key}: {column}")


def ingest(experiment_dir: Path, database: Path) -> dict[str, int]:
    """Index one experiment directory, safely repeatable and append-only."""
    experiment_dir = Path(experiment_dir).resolve()
    database = Path(database).resolve()
    manifest_path = experiment_dir / "manifest.json"
    manifest = _load(manifest_path)
    experiment_id = _safe_id(_required(manifest, "experiment_id"), "experiment_id")
    declared_manifest_hash = manifest_path.with_suffix(".sha256")
    if declared_manifest_hash.exists() and declared_manifest_hash.read_text(encoding="utf-8").strip() != _runner_digest(manifest):
        raise ValueError("Experiment manifest changed after preparation")
    source_path = experiment_dir / "source.json"
    if source_path.exists() and _runner_digest(_load(source_path)) != str(_required(manifest, "source_hash")):
        raise ValueError("Archived source snapshot changed")
    dataset_hash = str(_required(manifest, "dataset_hash"))
    source_hash = str(_required(manifest, "source_hash"))
    track = str(_required(manifest, "track"))
    manifest_hash = _manifest_hash(manifest)
    now = datetime.now(timezone.utc).isoformat()
    schedule_path = experiment_dir / "schedule.json"
    if schedule_path.exists():
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        cells = schedule.get("cells", []) if isinstance(schedule, dict) else schedule
        if not isinstance(cells, list):
            raise ValueError("schedule.json must contain a list or a cells list")
        expected = {_safe_id(cell.get("run_id"), "run_id") for cell in cells}
        actual = {_safe_id(path.parent.name, "run_id") for path in (experiment_dir / "runs").glob("*/result.json")}
        missing = expected - actual
        if missing:
            raise ValueError(f"scheduled runs missing results: {sorted(missing)}")
    connection = _connect(database)
    counts = {"experiments": 0, "cases": 0, "variants": 0, "snapshots": 0, "runs": 0, "samples": 0, "checks": 0, "artifacts": 0}
    try:
        connection.execute("BEGIN")
        # Keep the preparation lock and source snapshot beside the copied raw runs.
        _artifact_copy(manifest_path, database, experiment_id, "metadata")
        if source_path.exists():
            _artifact_copy(source_path, database, experiment_id, "metadata")
        connection.execute("INSERT OR IGNORE INTO datasets VALUES (?, ?, ?, ?, ?)", (dataset_hash, str(manifest.get("dataset_version", "unspecified")), str(manifest_path), str(manifest.get("reference_version", "unspecified")), manifest.get("frozen_at")))
        _ensure_same(connection, "experiments", "experiment_id", experiment_id, {"dataset_hash": dataset_hash, "manifest_hash": manifest_hash, "source_hash": source_hash})
        before = connection.total_changes
        connection.execute("INSERT OR IGNORE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (experiment_id, dataset_hash, manifest_hash, str(manifest_path), source_hash, str(manifest.get("environment_path", "")), track, str(manifest.get("created_at", now))))
        counts["experiments"] += connection.total_changes - before
        for case in manifest.get("cases", []):
            case_id = _safe_id(_required(case, "id"), "case_id")
            input_path = str(case.get("input_path", f"cases/{case_id}.pgn"))
            connection.execute("INSERT OR IGNORE INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (dataset_hash, case_id, str(case.get("source_group", "")), str(case.get("family_group", "")), str(case.get("split", "diagnostic")), str(case.get("category", "")), str(_required(case, "initial_fen")), int(_required(case, "target_ply")), input_path, str(_required(case, "input_hash")), _json({k: v for k, v in case.items() if k not in {"id", "source_group", "family_group", "split", "category", "initial_fen", "target_ply", "input_path", "input_hash"}})))
            counts["cases"] += 1
        for variant in manifest.get("variants", []):
            variant_id = _safe_id(_required(variant, "id"), "variant_id")
            config_hash = str(variant.get("config_hash", sha256_bytes(_json(variant).encode())))
            connection.execute("INSERT OR IGNORE INTO variants VALUES (?, ?, ?, ?)", (experiment_id, variant_id, config_hash, _json(variant)))
            counts["variants"] += 1
        snapshots_dir = experiment_dir / "snapshots"
        for case in manifest.get("cases", []):
            case_id = str(case["id"])
            snapshot_path = snapshots_dir / f"{case_id}.json"
            if not snapshot_path.exists():
                continue
            snapshot = _load(snapshot_path)
            snapshot_hash = str(snapshot.get("snapshot_hash", sha256_file(snapshot_path)))
            if "snapshot_hash" in snapshot and _runner_digest({k: v for k, v in snapshot.items() if k != "snapshot_hash"}) != snapshot_hash:
                raise ValueError(f"snapshot hash mismatch: {case_id}")
            relative, _ = _artifact_copy(snapshot_path, database, experiment_id, "snapshots")
            connection.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?, ?, ?)", (snapshot_hash, dataset_hash, case_id, relative, str(snapshot.get("engine_config_hash", "unspecified"))))
            counts["snapshots"] += 1
        runs_dir = experiment_dir / "runs"
        for result_path in sorted(runs_dir.glob("*/result.json")):
            result = _load(result_path)
            lock = result_path.with_suffix(".sha256")
            if lock.exists() and lock.read_text(encoding="utf-8").strip() != _runner_digest(result):
                raise ValueError("Run result hash mismatch")
            run_id = _safe_id(_required(result, "run_id"), "run_id")
            case_id = _safe_id(_required(result, "case_id"), "case_id")
            variant_id = _safe_id(_required(result, "variant_id"), "variant_id")
            status = str(_required(result, "status"))
            relative, artifact_hash = _artifact_copy(result_path, database, experiment_id, "runs")
            old_artifact = connection.execute("SELECT sha256 FROM artifacts WHERE run_id = ? AND kind = 'result'", (run_id,)).fetchone()
            if old_artifact is not None and old_artifact[0] != artifact_hash:
                raise ValueError(f"immutable result artifact conflict for {run_id}")
            snapshot_hash = result.get("snapshot_hash")
            connection.execute("INSERT OR IGNORE INTO runs (run_id,experiment_id,dataset_hash,case_id,variant_id,replicate,attempt,snapshot_hash,status,requested_renderer,actual_renderer,explanation_outcome,fallback_reason,error_category,seconds,input_tokens,output_tokens,cost_amount,cost_currency,cost_basis_json,result_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (run_id, experiment_id, dataset_hash, case_id, variant_id, int(result.get("replicate", 0)), int(result.get("attempt", 0)), snapshot_hash, status, str(result.get("requested_renderer", "template")), result.get("actual_renderer"), result.get("explanation_outcome"), result.get("fallback_reason"), result.get("error_category"), result.get("seconds"), result.get("input_tokens"), result.get("output_tokens"), result.get("cost_amount"), result.get("cost_currency"), _json(result.get("cost_basis")) if result.get("cost_basis") is not None else None, relative))
            _ensure_same(connection, "runs", "run_id", run_id, {"result_path": relative})
            connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?)", (run_id, "result", relative, artifact_hash))
            counts["artifacts"] += 1
            counts["runs"] += 1
            aggregate_checks: dict[str, list[int]] = {}
            for metric, check in (result.get("audit", {}).get("checks", {}) or {}).items():
                if isinstance(check, dict):
                    aggregate_checks[metric] = [int(check.get("passed", 0)), int(check.get("total", 0))]
            for sample in result.get("samples", []) or []:
                sample_id = str(_required(sample, "sample_id"))
                sample_path = relative
                text_hash = sha256_bytes(str(sample.get("comment", "")).encode("utf-8"))
                _ensure_same(connection, "samples", "sample_id", sample_id, {"run_id": run_id, "text_hash": text_hash})
                connection.execute("INSERT OR IGNORE INTO samples VALUES (?, ?, ?, ?, ?)", (sample_id, run_id, int(_required(sample, "target_ply")), text_hash, sample_path))
                counts["samples"] += 1
                for metric, check in (sample.get("audit", {}).get("checks", {}) or {}).items():
                    if isinstance(check, dict):
                        current = aggregate_checks.setdefault(metric, [0, 0])
                        current[0] += int(check.get("passed", 0)); current[1] += int(check.get("total", 0))
            for metric, (passed, total) in aggregate_checks.items():
                connection.execute("INSERT OR IGNORE INTO checks VALUES (?, ?, ?, ?, ?)", (run_id, metric, passed, total, None))
                counts["checks"] += 1
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return counts


def packet(experiment_dir: Path, output_dir: Path, seed: int = 20260905) -> dict[str, Any]:
    """Create a blinded, unrated packet and a private mode key."""
    experiment_dir, output_dir = Path(experiment_dir), Path(output_dir)
    manifest = _load(experiment_dir / "manifest.json")
    rows: list[dict[str, str]] = []
    mapping: list[dict[str, str]] = []
    rng = random.Random(seed)
    samples: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for result_path in sorted((experiment_dir / "runs").glob("*/result.json")):
        result = _load(result_path)
        for sample in result.get("samples", []) or []:
            samples.append((result, sample))
    rng.shuffle(samples)
    used: set[str] = set()
    for result, sample in samples:
        anon = secrets.token_urlsafe(12)
        while anon in used:
            anon = secrets.token_urlsafe(12)
        used.add(anon)
        fields = {"anonymous_id": anon, "rater_id": "", "fen": str(sample.get("fen_before", "")), "move": str(sample.get("move_san", "")), "comment": str(sample.get("comment", ""))}
        fields.update({key: "" for key in (*SCORES, *EXTRAS)})
        rows.append(fields)
        mapping.append({"anonymous_id": anon, "sample_id": str(sample["sample_id"]), "run_id": str(result["run_id"]), "case_id": str(result["case_id"]), "variant_id": str(result["variant_id"]), "replicate": str(result.get("replicate", 0))})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anonymous_id", "rater_id", "fen", "move", "comment", *SCORES, *EXTRAS])
        writer.writeheader(); writer.writerows(rows)
    with (output_dir / "private_key.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]) if mapping else ["anonymous_id", "sample_id", "run_id", "case_id", "variant_id", "replicate"])
        writer.writeheader(); writer.writerows(mapping)
    return {"experiment_id": manifest.get("experiment_id"), "samples": len(rows), "ratings_path": str(output_dir / "ratings.csv"), "key_path": str(output_dir / "private_key.csv"), "seed": seed}


def _value(row: dict[str, str], key: str) -> str | None:
    value = (row.get(key) or "").strip()
    return None if value.lower() in NULLS else value


def import_ratings(database: Path, path: Path, key_path: Path, rubric_version: str = "v1") -> dict[str, int]:
    """Validate and import one rater sheet; original bytes are archived."""
    database, path, key_path = Path(database), Path(path), Path(key_path)
    original_hash = sha256_file(path)
    with key_path.open("r", newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    key = {row["anonymous_id"]: row for row in key_rows}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("ratings sheet is empty")
    rater_id = _value(rows[0], "rater_id")
    if not rater_id:
        raise ValueError("ratings sheet requires one rater_id")
    if any(_value(row, "rater_id") != rater_id for row in rows):
        raise ValueError("all rows must use the same rater_id")
    seen: set[str] = set()
    connection = _connect(database)
    try:
        connection.execute("BEGIN")
        for row in rows:
            anon = _value(row, "anonymous_id")
            if not anon or anon not in key:
                raise ValueError(f"unknown anonymous_id: {anon}")
            if anon in seen:
                raise ValueError(f"duplicate anonymous_id: {anon}")
            seen.add(anon)
            for score in SCORES:
                value = _value(row, score)
                if value is not None and (not value.isdigit() or not 1 <= int(value) <= 5):
                    raise ValueError(f"{score} must be an integer from 1 to 5")
            mapped = key[anon]
            if mapped.get("rater_id") and mapped["rater_id"] != rater_id:
                raise ValueError("rater_id does not match private assignment key")
            sample_id = mapped["sample_id"]
            for field, allowed in (("endpoint_adequacy", {"insufficient", "sufficient", "extended", "unassessable"}), ("major_error", {"yes", "no", "unassessable"})):
                value = _value(row, field)
                if value is not None and value not in allowed:
                    raise ValueError(f"invalid {field}: {value}")
            connection.execute("INSERT INTO ratings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (sample_id, rater_id, rubric_version, *(int(_value(row, s)) if _value(row, s) is not None else None for s in SCORES), _value(row, "endpoint_adequacy"), _value(row, "major_error"), _value(row, "notes"), original_hash))
        archive = database.parent / "received-ratings" / f"{path.stem}-{original_hash}.csv"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists() and sha256_file(archive) != original_hash:
            raise ValueError("received-ratings archive collision")
        if not archive.exists():
            shutil.copyfile(path, archive)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {"imported": len(rows), "rater_id": rater_id, "original_sheet_hash": original_hash}
