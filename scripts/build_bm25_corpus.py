#!/usr/bin/env python3
"""Build a Tantivy BM25 index from annotated PGN files (Bolcic tokens + Stockfish PV + master comments)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Set

# Repo root for `app` imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import chess.engine
import chess.pgn
import tantivy

from app.core.commentary.annotation_corpus import (
    nearest_annotation_distance,
    ply_to_comment_map_from_game,
)
from app.core.commentary.features.positional_tokens import ENCODER_VERSION, encode_position
from app.core.commentary.features.rag_phase_features import (
    CORPUS_VERSION,
    build_opening_tags,
    classify_rag_phase,
    endgame_signature,
    material_bucket,
    material_signature,
    middlegame_strategic_tags,
    pawn_structure_fingerprint,
)
from app.core.commentary.openings.eco_book import ECOBook

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("build_bm25_corpus")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _eta_text(started_at: float, completed: int, total: int, unit: str) -> str:
    if completed <= 0:
        return f"ETA unknown (need at least one completed {unit})"
    elapsed = time.perf_counter() - started_at
    avg_per_unit = elapsed / completed
    remaining = max(0, total - completed)
    eta_seconds = avg_per_unit * remaining
    finish_at = datetime.now() + timedelta(seconds=eta_seconds)
    return (
        f"elapsed={_format_duration(elapsed)}, "
        f"avg/{unit}={_format_duration(avg_per_unit)}, "
        f"eta={_format_duration(eta_seconds)} "
        f"(finishes around {finish_at.strftime('%H:%M:%S')})"
    )


def _resolve_stockfish_path() -> Path:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    exe = "stockfish.exe" if platform.system() == "Windows" else "stockfish"
    candidates = [
        _REPO_ROOT / exe,
        _REPO_ROOT / "app" / "stockfish",
        _REPO_ROOT / "app" / "stockfish.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No Stockfish binary found. Tried STOCKFISH_PATH, {candidates}. "
        "Place stockfish.exe at repo root or set STOCKFISH_PATH."
    )


def _build_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    for name in (
        "static_attributes",
        "pawn_structure",
        "center",
        "dynamic_general",
        "dynamic_solution",
        "king_placement",
        "imbalance_signature",
        "strategic_tags",
        "endgame_signature",
    ):
        sb.add_text_field(name, tokenizer_name="whitespace", stored=False)
    for name in (
        "player_color",
        "game_id",
        "eco",
        "source",
        "fen",
        "pv_san",
        "annotation_text",
        "annotation_ply",
        "plies_to_next_annotation",
        "rag_phase",
        "opening_eco",
        "opening_prefix",
        "opening_name",
        "opening_matched_ply",
        "opening_ply_bucket",
        "material_signature",
        "material_bucket",
        "pawn_fingerprint",
        "corpus_version",
        "endgame_sig",
    ):
        sb.add_text_field(name, tokenizer_name="raw", stored=True)
    return sb.build()


def _iter_pgn_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.pgn") if p.is_file())


def _source_relative(pgn_path: Path, pgn_dir: Path) -> str:
    try:
        return str(pgn_path.resolve().relative_to(pgn_dir.resolve()))
    except ValueError:
        return pgn_path.name


def _collect_rows_from_pgns(
    files: List[Path],
    pgn_dir: Path,
    engine_path: str,
    depth: int,
    min_ply: int,
    max_ply: int,
    limit: int,
    seen_fens: Set[str],
    max_annotation_delta: int,
    progress_every_games: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    total_files = len(files)
    total_games_processed = 0
    collect_started_at = time.perf_counter()
    start_seen_fens = len(seen_fens)
    LOG.info(
        "Starting row collection from %d PGN files (starting seen_fens=%d, limit=%s)",
        total_files,
        start_seen_fens,
        "none" if limit >= 10**9 else limit,
    )

    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Hash": 128, "Threads": 2, "Use NNUE": False})
    eco_book = ECOBook()
    try:
        for file_index, pgn_path in enumerate(files, start=1):
            if len(seen_fens) >= limit:
                LOG.info("Reached unique FEN limit before %s; stopping collection.", pgn_path.name)
                break

            file_started_at = time.perf_counter()
            rows_before = len(rows)
            seen_before = len(seen_fens)
            games_before = total_games_processed
            LOG.info("[%d/%d] Processing %s", file_index, total_files, pgn_path)
            LOG.info("         %s", _eta_text(collect_started_at, file_index - 1, total_files, "file"))

            text = pgn_path.read_text(encoding="utf-8", errors="replace")
            buf = StringIO(text)
            rel_source = _source_relative(pgn_path, pgn_dir)
            gi = 0
            while len(seen_fens) < limit:
                game = chess.pgn.read_game(buf)
                if game is None:
                    break
                total_games_processed += 1
                ply_to_comment = ply_to_comment_map_from_game(game)
                board = game.board()
                node = game
                eco = (game.headers.get("ECO") or "")[:16] if game.headers else ""
                header_eco = (game.headers.get("ECO") or "").strip() if game.headers else ""
                ply = 0
                uci_made: List[str] = []
                while not node.is_end():
                    if len(seen_fens) >= limit:
                        break
                    next_node = node.variation(0)
                    move = next_node.move
                    if move is None:
                        break
                    board.push(move)
                    uci_made.append(move.uci())
                    ply += 1
                    node = next_node

                    if ply < min_ply or ply > max_ply:
                        continue

                    delta = nearest_annotation_distance(
                        ply_to_comment, ply, max_delta=max_annotation_delta
                    )
                    if delta is None:
                        continue

                    ann_ply = ply + delta
                    ann_text = ply_to_comment.get(ann_ply, "")
                    if not ann_text:
                        continue

                    fen = board.fen()
                    if fen in seen_fens:
                        continue

                    try:
                        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
                        if isinstance(info, list) and info:
                            info = info[0]
                        if not isinstance(info, dict):
                            continue
                        pv = info.get("pv") or []
                        b2 = board.copy()
                        pv_san: List[str] = []
                        for m in pv[:8]:
                            if m not in b2.legal_moves:
                                break
                            pv_san.append(b2.san(m))
                            b2.push(m)
                    except Exception as e:
                        LOG.debug("analyse failed: %s", e)
                        continue

                    if len(pv_san) < 2:
                        continue

                    enc = encode_position(board, pv_san)
                    seen_fens.add(fen)

                    rag_phase = classify_rag_phase(board)
                    ot = build_opening_tags(uci_made, ply, eco_book=eco_book, header_eco=header_eco)
                    pfp = pawn_structure_fingerprint(board)
                    mat_sig = material_signature(board)
                    mat_b = material_bucket(board)
                    if rag_phase == "endgame":
                        end_sig = endgame_signature(board)
                        strat = ""
                    elif rag_phase == "middlegame":
                        end_sig = ""
                        strat = middlegame_strategic_tags(board)
                    else:
                        end_sig = ""
                        strat = ""

                    rows.append(
                        {
                            "static_attributes": enc["static_attributes"],
                            "pawn_structure": enc["pawn_structure"],
                            "center": enc["center"],
                            "dynamic_general": enc["dynamic_general"],
                            "dynamic_solution": enc["dynamic_solution"],
                            "king_placement": enc.get("king_placement", ""),
                            "imbalance_signature": enc.get("imbalance_signature", ""),
                            "strategic_tags": strat,
                            "endgame_signature": end_sig,
                            "endgame_sig": end_sig,
                            "player_color": enc["player_color"],
                            "game_id": f"{rel_source}:g{gi}",
                            "eco": eco,
                            "source": rel_source,
                            "fen": fen,
                            "pv_san": enc["pv_san"],
                            "annotation_text": ann_text,
                            "annotation_ply": str(ann_ply),
                            "plies_to_next_annotation": str(delta),
                            "rag_phase": rag_phase,
                            "opening_eco": ot.opening_eco,
                            "opening_prefix": ot.opening_prefix,
                            "opening_name": ot.opening_name[:200],
                            "opening_matched_ply": str(ot.opening_matched_ply),
                            "opening_ply_bucket": ot.opening_ply_bucket,
                            "material_signature": mat_sig,
                            "material_bucket": mat_b,
                            "pawn_fingerprint": pfp,
                            "corpus_version": CORPUS_VERSION,
                        }
                    )
                gi += 1

                if progress_every_games > 0 and total_games_processed % progress_every_games == 0:
                    elapsed = time.perf_counter() - collect_started_at
                    docs_per_min = (len(rows) / elapsed * 60) if elapsed > 0 else 0.0
                    LOG.info(
                        "         Games=%d, docs=%d, seen_fens=%d, speed=%.1f docs/min",
                        total_games_processed,
                        len(rows),
                        len(seen_fens),
                        docs_per_min,
                    )

            file_elapsed = time.perf_counter() - file_started_at
            file_docs = len(rows) - rows_before
            file_new_fens = len(seen_fens) - seen_before
            file_games = total_games_processed - games_before
            LOG.info(
                "[%d/%d] Done %s | games=%d, +docs=%d, +fens=%d in %s",
                file_index,
                total_files,
                pgn_path.name,
                file_games,
                file_docs,
                file_new_fens,
                _format_duration(file_elapsed),
            )
            LOG.info(
                "         Totals: games=%d, docs=%d, seen_fens=%d | %s",
                total_games_processed,
                len(rows),
                len(seen_fens),
                _eta_text(collect_started_at, file_index, total_files, "file"),
            )
    finally:
        engine.quit()

    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pgn-dir",
        type=Path,
        default=_REPO_ROOT.parent / "data",
        help="Directory containing .pgn files (default: ../data from repo)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "bm25_positions_v2",
        help="Output directory for Tantivy index (default: bm25_positions_v2; legacy was bm25_positions)",
    )
    p.add_argument(
        "--min-ply",
        type=int,
        default=1,
        help="Minimum half-move ply to index (default 1; early opening positions included)",
    )
    p.add_argument(
        "--max-ply",
        type=int,
        default=200,
        help="Maximum half-move ply to index (default 200; endgame coverage)",
    )
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--limit", type=int, default=30_000, help="Max unique FENs (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="Sample encode_position only; no Stockfish / no index")
    p.add_argument("--resume", action="store_true", help="Skip FENs listed in output/fen_seen.txt")
    p.add_argument(
        "--max-annotation-delta",
        type=int,
        default=4,
        help="Max forward plies to look for a non-junk comment (strict: skip position if none)",
    )
    p.add_argument(
        "--progress-every-games",
        type=int,
        default=100,
        help="Log collection progress every N games (0 disables).",
    )
    p.add_argument(
        "--progress-every-docs",
        type=int,
        default=1000,
        help="Log index-write progress every N docs (0 disables).",
    )
    args = p.parse_args()

    pgn_dir: Path = args.pgn_dir.resolve()
    out_dir: Path = args.output.resolve()

    if not pgn_dir.is_dir():
        LOG.error("pgn-dir does not exist: %s", pgn_dir)
        sys.exit(1)

    files = _iter_pgn_files(pgn_dir)
    if not files:
        LOG.error("No .pgn files under %s", pgn_dir)
        sys.exit(1)

    if args.dry_run:
        text = files[0].read_text(encoding="utf-8", errors="replace")
        g = chess.pgn.read_game(StringIO(text))
        if g:
            board = g.board()
            node = g
            for _ in range(12):
                if node.is_end():
                    break
                node = node.variation(0)
                board.push(node.move)
            enc = encode_position(board, ["d4", "d5", "Nf3"])
            LOG.info("dry-run sample file=%s", files[0].name)
            for k, v in enc.items():
                LOG.info("  %s=%.200s...", k, str(v))
        LOG.info("dry-run ok; %d pgn files found", len(files))
        return

    limit = args.limit if args.limit > 0 else 10**9
    engine_path = str(_resolve_stockfish_path())
    LOG.info("Using Stockfish at %s", engine_path)

    seen_fens: Set[str] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    fen_seen_path = out_dir / "fen_seen.txt"
    if args.resume and fen_seen_path.is_file():
        seen_fens = set(fen_seen_path.read_text(encoding="utf-8").splitlines())
        LOG.info("resume: loaded %d FENs from %s", len(seen_fens), fen_seen_path)

    t0 = time.perf_counter()
    rows = _collect_rows_from_pgns(
        files,
        pgn_dir,
        engine_path,
        args.depth,
        args.min_ply,
        args.max_ply,
        limit,
        seen_fens,
        max_annotation_delta=args.max_annotation_delta,
        progress_every_games=args.progress_every_games,
    )
    elapsed = time.perf_counter() - t0
    LOG.info("Collected %d documents in %.1fs", len(rows), elapsed)

    schema = _build_schema()
    if out_dir.exists() and any(out_dir.iterdir()):
        # Tantivy needs empty dir or we remove old index
        import shutil

        for child in out_dir.iterdir():
            if child.name != "fen_seen.txt":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    index = tantivy.Index(schema, path=str(out_dir))
    writer = index.writer(heap_size=256_000_000)
    write_started_at = time.perf_counter()
    LOG.info("Writing %d documents to index at %s", len(rows), out_dir)
    for doc_index, r in enumerate(rows, start=1):
        writer.add_document(
            tantivy.Document(
                static_attributes=r["static_attributes"],
                pawn_structure=r["pawn_structure"],
                center=r["center"],
                dynamic_general=r["dynamic_general"],
                dynamic_solution=r["dynamic_solution"],
                king_placement=r["king_placement"],
                imbalance_signature=r["imbalance_signature"],
                strategic_tags=r.get("strategic_tags", ""),
                endgame_signature=r.get("endgame_signature", ""),
                player_color=r["player_color"],
                game_id=r["game_id"],
                eco=r["eco"],
                source=r["source"],
                fen=r["fen"],
                pv_san=r["pv_san"],
                annotation_text=r.get("annotation_text", ""),
                annotation_ply=r.get("annotation_ply", ""),
                plies_to_next_annotation=r.get("plies_to_next_annotation", "0"),
                rag_phase=r.get("rag_phase", ""),
                opening_eco=r.get("opening_eco", ""),
                opening_prefix=r.get("opening_prefix", ""),
                opening_name=r.get("opening_name", ""),
                opening_matched_ply=r.get("opening_matched_ply", ""),
                opening_ply_bucket=r.get("opening_ply_bucket", ""),
                material_signature=r.get("material_signature", ""),
                material_bucket=r.get("material_bucket", ""),
                pawn_fingerprint=r.get("pawn_fingerprint", ""),
                corpus_version=r.get("corpus_version", CORPUS_VERSION),
                endgame_sig=r.get("endgame_sig", ""),
            )
        )
        if (
            args.progress_every_docs > 0
            and (doc_index % args.progress_every_docs == 0 or doc_index == len(rows))
        ):
            LOG.info(
                "Index write progress: %d/%d docs | %s",
                doc_index,
                len(rows),
                _eta_text(write_started_at, doc_index, len(rows), "doc"),
            )
    writer.commit()
    index.reload()
    LOG.info(
        "Index write+reload done in %s",
        _format_duration(time.perf_counter() - write_started_at),
    )

    fen_seen_path.write_text("\n".join(sorted(seen_fens)), encoding="utf-8")
    meta = {
        "corpus_version": CORPUS_VERSION,
        "encoder_version": ENCODER_VERSION,
        "corpus_source": "pgn",
        "n_docs": len(rows),
        "annotated_only": True,
        "annotated_rows": len(rows),
        "max_annotation_delta": args.max_annotation_delta,
        "pgn_dir": str(pgn_dir),
        "pgn_files": [_source_relative(f, pgn_dir) for f in files],
        "stockfish_depth": args.depth,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "rag_min_score_suggested": {
            "opening": float(os.environ.get("RAG_MIN_SCORE_OPENING", "0.45")),
            "middlegame": float(os.environ.get("RAG_MIN_SCORE_MIDDLEGAME", "0.42")),
            "endgame": float(os.environ.get("RAG_MIN_SCORE_ENDGAME", "0.40")),
        },
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(elapsed, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("Wrote index to %s (%d docs)", out_dir, len(rows))


if __name__ == "__main__":
    main()
