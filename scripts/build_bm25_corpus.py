#!/usr/bin/env python3
"""Build a Tantivy BM25 index from PGN files (Bolčič 2024-style tokens + Stockfish PV)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
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

from app.core.commentary.features.positional_tokens import ENCODER_VERSION, encode_position

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("build_bm25_corpus")


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
    ):
        sb.add_text_field(name, tokenizer_name="whitespace", stored=False)
    for name in ("player_color", "game_id", "eco", "source", "fen", "pv_san"):
        sb.add_text_field(name, tokenizer_name="raw", stored=True)
    return sb.build()


def _iter_pgn_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.pgn") if p.is_file())


def _collect_rows_from_pgns(
    files: List[Path],
    engine_path: str,
    depth: int,
    min_ply: int,
    max_ply: int,
    limit: int,
    seen_fens: Set[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Hash": 128, "Threads": 2, "Use NNUE": False})
    try:
        for pgn_path in files:
            if len(seen_fens) >= limit:
                break
            text = pgn_path.read_text(encoding="utf-8", errors="replace")
            buf = StringIO(text)
            gi = 0
            while len(seen_fens) < limit:
                game = chess.pgn.read_game(buf)
                if game is None:
                    break
                board = game.board()
                node = game
                eco = (game.headers.get("ECO") or "")[:16] if game.headers else ""
                ply = 0
                while not node.is_end():
                    if len(seen_fens) >= limit:
                        break
                    next_node = node.variation(0)
                    move = next_node.move
                    if move is None:
                        break
                    board.push(move)
                    ply += 1
                    node = next_node

                    if ply < min_ply or ply > max_ply:
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

                    rows.append(
                        {
                            "static_attributes": enc["static_attributes"],
                            "pawn_structure": enc["pawn_structure"],
                            "center": enc["center"],
                            "dynamic_general": enc["dynamic_general"],
                            "dynamic_solution": enc["dynamic_solution"],
                            "player_color": enc["player_color"],
                            "game_id": f"{pgn_path.name}:g{gi}",
                            "eco": eco,
                            "source": pgn_path.name,
                            "fen": fen,
                            "pv_san": enc["pv_san"],
                        }
                    )
                gi += 1
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
        default=_REPO_ROOT / "data" / "bm25_positions",
        help="Output directory for Tantivy index",
    )
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--max-ply", type=int, default=40)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--limit", type=int, default=30_000, help="Max unique FENs (0 = no limit)")
    p.add_argument("--dry-run", action="store_true", help="Sample encode_position only; no Stockfish / no index")
    p.add_argument("--resume", action="store_true", help="Skip FENs listed in output/fen_seen.txt")
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
        engine_path,
        args.depth,
        args.min_ply,
        args.max_ply,
        limit,
        seen_fens,
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
    for r in rows:
        writer.add_document(
            tantivy.Document(
                static_attributes=r["static_attributes"],
                pawn_structure=r["pawn_structure"],
                center=r["center"],
                dynamic_general=r["dynamic_general"],
                dynamic_solution=r["dynamic_solution"],
                player_color=r["player_color"],
                game_id=r["game_id"],
                eco=r["eco"],
                source=r["source"],
                fen=r["fen"],
                pv_san=r["pv_san"],
            )
        )
    writer.commit()
    index.reload()

    fen_seen_path.write_text("\n".join(sorted(seen_fens)), encoding="utf-8")
    meta = {
        "encoder_version": ENCODER_VERSION,
        "corpus_source": "pgn",
        "n_docs": len(rows),
        "pgn_files": [str(f.name) for f in files],
        "stockfish_depth": args.depth,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(elapsed, 2),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOG.info("Wrote index to %s (%d docs)", out_dir, len(rows))


if __name__ == "__main__":
    main()
