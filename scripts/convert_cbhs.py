#!/usr/bin/env python3
"""Recursively convert CBH files under data/cbh to PGN files under data/pgn."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from cbh2pgn import read_cbh


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CBH_DIR = _REPO_ROOT / "data" / "cbh"
_DEFAULT_PGN_DIR = _REPO_ROOT / "data" / "pgn"


def _iter_cbh_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.cbh") if path.is_file())


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _eta_text(started_at: float, processed: int, total: int) -> str:
    if processed <= 0:
        return "ETA unknown (need at least one completed file)"

    elapsed = time.perf_counter() - started_at
    avg_per_file = elapsed / processed
    remaining = max(0, total - processed)
    eta_seconds = avg_per_file * remaining
    finish_at = datetime.now() + timedelta(seconds=eta_seconds)
    return (
        f"elapsed={_format_duration(elapsed)}, "
        f"avg/file={_format_duration(avg_per_file)}, "
        f"eta={_format_duration(eta_seconds)} "
        f"(finishes around {finish_at.strftime('%H:%M:%S')})"
    )


def convert_all(cbh_dir: Path, pgn_dir: Path, overwrite: bool, max_games: int) -> int:
    cbh_files = _iter_cbh_files(cbh_dir)
    if not cbh_files:
        print(f"No .cbh files found under: {cbh_dir}")
        return 0

    total = len(cbh_files)
    converted = 0
    skipped = 0
    errors = 0
    started_at = time.perf_counter()

    print(f"Found {total} .cbh files under: {cbh_dir}", flush=True)

    for index, cbh_path in enumerate(cbh_files, start=1):
        rel_cbh = cbh_path.relative_to(cbh_dir)
        out_path = (pgn_dir / rel_cbh).with_suffix(".pgn")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_started_at = time.perf_counter()

        processed_so_far = index - 1
        print(f"[{index}/{total}] Processing: {rel_cbh}", flush=True)
        print(f"         {_eta_text(started_at, processed_so_far, total)}", flush=True)

        if out_path.exists() and not overwrite:
            skipped += 1
            file_elapsed = time.perf_counter() - file_started_at
            print(
                f"[SKIP] {rel_cbh} -> {out_path.relative_to(pgn_dir)} "
                f"(already exists, took {_format_duration(file_elapsed)})",
                flush=True,
            )
            continue

        try:
            pgn_text = read_cbh(
                cbh_path.as_posix(),
                max_games=max_games,
                output="text",
            )
            if not isinstance(pgn_text, str):
                raise TypeError("cbh2pgn.read_cbh did not return PGN text.")
            out_path.write_text(pgn_text, encoding="utf-8")
            converted += 1
            file_elapsed = time.perf_counter() - file_started_at
            print(
                f"[OK]   {rel_cbh} -> {out_path.relative_to(pgn_dir)} "
                f"(took {_format_duration(file_elapsed)})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            file_elapsed = time.perf_counter() - file_started_at
            print(f"[ERR]  {rel_cbh}: {exc}", file=sys.stderr)
            print(f"       failed after {_format_duration(file_elapsed)}", file=sys.stderr)

        print(
            f"       Progress: converted={converted}, skipped={skipped}, errors={errors} | "
            f"{_eta_text(started_at, index, total)}",
            flush=True,
        )

    total_elapsed = time.perf_counter() - started_at
    files_per_min = (total / total_elapsed * 60) if total_elapsed > 0 else 0.0
    print(
        f"Done. Converted: {converted}, Skipped: {skipped}, Errors: {errors}, "
        f"Total: {total}, Runtime: {_format_duration(total_elapsed)}, "
        f"Speed: {files_per_min:.2f} files/min"
    )
    return 1 if errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cbh-dir",
        type=Path,
        default=_DEFAULT_CBH_DIR,
        help=f"Input directory containing .cbh files (default: {_DEFAULT_CBH_DIR})",
    )
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        default=_DEFAULT_PGN_DIR,
        help=f"Output directory for .pgn files (default: {_DEFAULT_PGN_DIR})",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=10**9,
        help="Maximum number of games to convert per .cbh file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pgn files in the output directory.",
    )
    args = parser.parse_args()

    cbh_dir = args.cbh_dir.resolve()
    pgn_dir = args.pgn_dir.resolve()

    if not cbh_dir.is_dir():
        print(f"Input directory does not exist: {cbh_dir}", file=sys.stderr)
        sys.exit(1)

    exit_code = convert_all(cbh_dir, pgn_dir, overwrite=args.overwrite, max_games=args.max_games)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
