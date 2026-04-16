#!/usr/bin/env python3
"""Build a ChromaDB RAG corpus from annotated PGN games (mainline comments + positional features)."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from io import StringIO
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Repo root for `app` imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import chess
import chess.pgn
from openai import OpenAI

from app.core.commentary.features.positional_features import compute_hidden_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("build_rag_corpus")

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "chess_annotations"

# Comments that are only glyphs / eval / move numbers (no prose letters)
_NON_LETTER = re.compile(r"[A-Za-z]")


def _infer_phase(board: chess.Board) -> str:
    ply = board.ply()
    if ply < 36:
        return "opening"
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))
    if wq == 0 and bq == 0:
        return "endgame"
    pieces_non_king = 0
    for c in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP):
            pieces_non_king += len(board.pieces(pt, c))
    if pieces_non_king <= 6:
        return "endgame"
    return "middlegame"


def _flatten_metadata_for_chroma(
    source_file: str,
    game_index: int,
    san: str,
    fen: str,
    features: Dict[str, Any],
    pawn_structure_type: str,
    phase: str,
) -> Dict[str, Any]:
    """Chroma allows str, int, float, bool, None."""
    mat = features.get("material") or {}
    diff = mat.get("diff") or {}
    imb = mat.get("imbalance")
    if isinstance(imb, list):
        imbalance_s = ",".join(str(x) for x in imb[:6])
    else:
        imbalance_s = ""

    of = features.get("openFiles") or {}
    open_files = of.get("open") or []

    white = features.get("white") or {}
    black = features.get("black") or {}

    return {
        "source_file": source_file,
        "game_index": int(game_index),
        "san": san[:64],
        "fen": fen[:512],
        "phase": phase,
        "pawn_structure_type": str(pawn_structure_type),
        "material_total": int(diff.get("total", 0)),
        "material_imbalance": imbalance_s[:256],
        "white_mobility": int(white.get("mobility", 0)),
        "black_mobility": int(black.get("mobility", 0)),
        "white_king_exposure": float(white.get("kingExposure", 0.0)),
        "black_king_exposure": float(black.get("kingExposure", 0.0)),
        "white_space": int(white.get("space", 0)),
        "black_space": int(black.get("space", 0)),
        "white_has_bishop_pair": bool(white.get("hasBishopPair", False)),
        "black_has_bishop_pair": bool(black.get("hasBishopPair", False)),
        "open_files_count": int(len(open_files)),
    }


def _build_document(san: str, phase: str, center_type: str, comment: str) -> str:
    return f"{san} in {phase}, {center_type} center. {comment}"


def _quality_ok(
    text: str,
    min_len: int,
    drop_reasons: Counter,
) -> bool:
    s = text.strip()
    if len(s) < min_len:
        drop_reasons["too_short"] += 1
        return False
    if _NON_LETTER.search(s) is None:
        drop_reasons["no_letters"] += 1
        return False
    return True


def _iter_pgn_files(data_dir: Path) -> Iterable[Path]:
    for p in sorted(data_dir.rglob("*.pgn")):
        if p.is_file():
            yield p


def _parse_one_game(
    handle,
) -> Tuple[Optional[chess.pgn.Game], Optional[str]]:
    try:
        return chess.pgn.read_game(handle), None
    except Exception as e:
        return None, str(e)


@dataclass
class Row:
    doc: str
    metadata: Dict[str, Any]
    id: str


@dataclass
class RunMetrics:
    files: int = 0
    games: int = 0
    comments_raw: int = 0  # non-empty comment nodes before quality/dedupe
    rows_kept: int = 0
    parse_errors: int = 0
    drop_reasons: Counter = field(default_factory=Counter)
    embedding_tokens_est: int = 0
    api_calls: int = 0
    wall_ms: float = 0.0


def _walk_mainline_comments(
    game: chess.pgn.Game,
    source_file: str,
    game_index: int,
    min_len: int,
    dedupe_keys: Set[str],
    drop_reasons: Counter,
    raw_comments: List[int],
) -> List[Row]:
    rows: List[Row] = []
    board = game.board()
    node = game

    path_tag = Path(source_file).name

    while not node.is_end():
        next_node = node.variation(0)
        move = next_node.move
        if move is None:
            break
        san = board.san(move)
        board.push(move)
        comment = (next_node.comment or "").strip()

        if comment:
            raw_comments[0] += 1
            if not _quality_ok(comment, min_len, drop_reasons):
                node = next_node
                continue

            key = f"{board.fen()}|{comment.lower()[:200]}"
            if key in dedupe_keys:
                drop_reasons["deduped"] += 1
                node = next_node
                continue
            dedupe_keys.add(key)

            fen = board.fen()
            feats = compute_hidden_features(board)
            ps = feats.get("pawnStructure") or {}
            center_type = str(ps.get("centerType", "semi-open"))
            phase = _infer_phase(board)

            meta = _flatten_metadata_for_chroma(
                path_tag,
                game_index,
                san,
                fen,
                feats,
                center_type,
                phase,
            )
            doc = _build_document(san, phase, center_type, comment)
            h = hashlib.sha256(f"{fen}|{comment}".encode()).hexdigest()[:16]
            rid = f"{path_tag}:g{game_index}:m{board.fullmove_number}:{h}"
            rid = re.sub(r"[^\w\-.:]", "_", rid)[:256]
            rows.append(Row(doc=doc, metadata=meta, id=rid))

        node = next_node

    return rows


def _embed_batches(
    client: OpenAI,
    texts: List[str],
    batch_size: int,
    metrics: RunMetrics,
) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        metrics.api_calls += 1
        metrics.embedding_tokens_est += sum(max(1, len(t) // 4) for t in batch)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        # Preserve order
        data = sorted(resp.data, key=lambda x: x.index)
        for d in data:
            out.append(list(d.embedding))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_REPO_ROOT.parent / "data",
        help="Root folder to scan for .pgn files (default: ../data from repo)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "rag_corpus",
        help="ChromaDB persist directory",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--min-comment-length", type=int, default=12)
    p.add_argument("--dry-run", action="store_true", help="Parse + features only; no OpenAI or Chroma")
    p.add_argument("--limit-games", type=int, default=0, help="Stop after N games total (0 = no limit)")
    args = p.parse_args()

    data_dir: Path = args.data_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not data_dir.is_dir():
        LOG.error("data-dir does not exist: %s", data_dir)
        sys.exit(1)

    metrics = RunMetrics()
    t0 = time.perf_counter()
    all_rows: List[Row] = []
    dedupe_keys: Set[str] = set()
    seen_games = 0

    for pgn_path in _iter_pgn_files(data_dir):
        metrics.files += 1
        try:
            text = pgn_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            LOG.warning("Skip %s: %s", pgn_path, e)
            metrics.drop_reasons[f"read_error:{e}"] += 1
            continue

        buf = StringIO(text)
        gi = 0
        while True:
            if args.limit_games and seen_games >= args.limit_games:
                break
            game, err = _parse_one_game(buf)
            if err:
                metrics.parse_errors += 1
                LOG.debug("Parse error in %s: %s", pgn_path.name, err)
                break
            if game is None:
                break
            seen_games += 1
            metrics.games += 1
            raw_holder = [0]
            rows = _walk_mainline_comments(
                game,
                str(pgn_path),
                gi,
                args.min_comment_length,
                dedupe_keys,
                metrics.drop_reasons,
                raw_holder,
            )
            metrics.comments_raw += raw_holder[0]
            all_rows.extend(rows)
            gi += 1
        if args.limit_games and seen_games >= args.limit_games:
            break

    metrics.rows_kept = len(all_rows)

    if args.dry_run or not all_rows:
        metrics.wall_ms = (time.perf_counter() - t0) * 1000
        LOG.info("=== dry-run summary ===")
        LOG.info(
            "files=%d games=%d comments_raw=%d rows_kept=%d",
            metrics.files,
            metrics.games,
            metrics.comments_raw,
            metrics.rows_kept,
        )
        LOG.info("drop_reasons=%s", dict(metrics.drop_reasons))
        LOG.info("wall_ms=%.1f", metrics.wall_ms)
        if all_rows:
            sample = all_rows[:3]
            for r in sample:
                LOG.info("sample doc: %s", r.doc[:200])
                LOG.info("sample meta keys: %s", list(r.metadata.keys()))
        return

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        LOG.error("OPENAI_API_KEY is required for embedding (omit --dry-run)")
        sys.exit(1)

    import chromadb

    client = OpenAI()
    texts = [r.doc for r in all_rows]
    embeddings = _embed_batches(client, texts, args.batch_size, metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    chroma = chromadb.PersistentClient(path=str(output_dir))
    coll = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )
    chroma_batch = 300
    for start in range(0, len(all_rows), chroma_batch):
        end = start + chroma_batch
        coll.add(
            ids=[r.id for r in all_rows[start:end]],
            documents=texts[start:end],
            metadatas=[r.metadata for r in all_rows[start:end]],
            embeddings=embeddings[start:end],
        )

    metrics.wall_ms = (time.perf_counter() - t0) * 1000
    LOG.info("=== corpus build complete ===")
    LOG.info(
        "files=%d games=%d comments_raw=%d rows_stored=%d",
        metrics.files,
        metrics.games,
        metrics.comments_raw,
        len(all_rows),
    )
    LOG.info("chroma_path=%s collection=%s", output_dir, COLLECTION_NAME)
    LOG.info("embedding_batches=%d token_est=%d wall_ms=%.1f", metrics.api_calls, metrics.embedding_tokens_est, metrics.wall_ms)
    LOG.info("drop_reasons=%s", dict(metrics.drop_reasons))


if __name__ == "__main__":
    main()
