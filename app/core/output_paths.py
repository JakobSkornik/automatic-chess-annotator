"""Paths for durable API output, with compatibility for the legacy layout."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_job_id(job_id: str) -> str:
    """Validate an identifier before using it in a filesystem path."""
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid job id")
    return job_id


def output_root() -> Path:
    configured = os.environ.get("ACA_OUTPUT_DIR", "").strip()
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2] / "output"


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def games_root() -> Path:
    return output_root() / "games"


def logs_root() -> Path:
    return output_root() / "logs"


def dated_game_path(job_id: str, date: str | None = None) -> Path:
    selected = date or utc_date()
    if not _DATE.fullmatch(selected):
        raise ValueError("invalid output date")
    return games_root() / selected / f"{validate_job_id(job_id)}.json"


def legacy_game_path(job_id: str) -> Path:
    return Path("data/games") / f"{validate_job_id(job_id)}.json"


def find_game_path(job_id: str) -> Path | None:
    """Return dated output first, then legacy output, without traversing paths."""
    validate_job_id(job_id)
    dated = sorted(games_root().glob(f"*/{job_id}.json"), reverse=True)
    for path in dated:
        if path.is_file():
            return path
    legacy = legacy_game_path(job_id)
    return legacy if legacy.is_file() else None


def iter_game_paths() -> list[Path]:
    """List games with dated output taking precedence over legacy duplicates."""
    found: dict[str, Path] = {}
    if games_root().is_dir():
        for path in sorted(games_root().glob("*/*.json"), reverse=True):
            if path.is_file() and _JOB_ID.fullmatch(path.stem):
                found.setdefault(path.stem, path)
    legacy_dir = Path("data/games")
    if legacy_dir.is_dir():
        for path in sorted(legacy_dir.glob("*.json")):
            if path.is_file() and _JOB_ID.fullmatch(path.stem):
                found.setdefault(path.stem, path)
    return sorted(found.values(), key=lambda p: p.stem)
