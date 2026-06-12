"""Annotated PGN export from a finished ``GameJson``.

Renders comments, evals (``[%eval]``), NAGs, the displayed continuation as a
variation after the move, and the better alternative as a variation at the
same point — readable by Lichess/ChessBase/any PGN viewer. Interactive
``[type:content]`` tokens are flattened to plain text; ``[pv:...]`` tokens are
lifted out of the text and become the variations.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.pgn

from app.models.GameJson import GameJson, GameMove

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\[(\w+):([^\]]+)\]")

# NAG numbers: $1 !, $2 ?, $3 !!, $4 ??, $5 !?, $6 ?!
_CLASSIFICATION_NAGS = {
    "brilliant": 3,
    "great_move": 1,
}
_QUALITY_NAGS = {
    "blunder": 4,
    "mistake": 2,
    "inaccuracy": 6,
}


def _flatten_tokens(text: str) -> Tuple[str, List[List[str]]]:
    """Strip interactive tokens; return (plain_text, pv_lines_as_san_lists)."""
    pv_lines: List[List[str]] = []

    # "... after [pv:x y z] (+0.12, ...)" -> drop the connective with the token;
    # the line itself becomes a PGN variation instead.
    text = re.sub(r"\s*\bafter\s+(?=\[pv:)", " ", text or "")

    def repl(m: re.Match) -> str:
        ttype = m.group(1).lower()
        content = m.group(2).strip()
        if ttype == "pv":
            sans = [s for s in content.split() if s]
            if sans:
                pv_lines.append(sans)
            return ""
        if ttype == "file":
            return f"the {content}-file"
        # move / square / piece / eval: keep the raw content
        return content

    plain = _TOKEN_RE.sub(repl, text)
    plain = re.sub(r"\s+", " ", plain)
    plain = re.sub(r"\s+([.,;:!?])", r"\1", plain)
    return plain.strip(), pv_lines


def _eval_tag(move: GameMove) -> str:
    sc = move.score
    if sc is None:
        return ""
    if sc.mate is not None:
        return f"[%eval #{sc.mate}]"
    if sc.cp is not None:
        return f"[%eval {sc.cp / 100:.2f}]"
    return ""


def _nag_for_move(move: GameMove) -> Optional[int]:
    if move.classification in _CLASSIFICATION_NAGS:
        return _CLASSIFICATION_NAGS[move.classification]
    if move.move_quality in _QUALITY_NAGS:
        return _QUALITY_NAGS[move.move_quality]
    return None


def _feature_note(move: GameMove, max_items: int = 4) -> str:
    if not move.feature_refs:
        return ""
    parts = []
    for fr in move.feature_refs[:max_items]:
        sign = "+" if fr.delta_cp >= 0 else ""
        parts.append(f"{fr.name} {sign}{fr.delta_cp}")
    return "Features: " + ", ".join(parts)


def _try_add_line(
    start_board: chess.Board,
    node: chess.pgn.GameNode,
    sans: List[str],
) -> None:
    """Attach a SAN line as a variation from ``node`` (position = start_board)."""
    board = start_board.copy(stack=False)
    cursor = node
    for i, san in enumerate(sans):
        try:
            mv = board.parse_san(san)
        except Exception:
            break
        if i == 0:
            if cursor.has_variation(mv):
                cursor = cursor.variation(mv)
                board.push(mv)
                continue
            cursor = cursor.add_variation(mv)
        else:
            cursor = cursor.add_variation(mv) if not cursor.has_variation(mv) else cursor.variation(mv)
        board.push(mv)


def game_json_to_pgn(gj: GameJson, *, include_features: bool = False) -> str:
    game = chess.pgn.Game()
    md = gj.metadata
    game.headers["Event"] = md.eventId or "?"
    game.headers["White"] = md.white or "?"
    game.headers["Black"] = md.black or "?"
    game.headers["Result"] = md.result or "*"
    if md.date:
        game.headers["Date"] = md.date
    if md.whiteElo:
        game.headers["WhiteElo"] = str(md.whiteElo)
    if md.blackElo:
        game.headers["BlackElo"] = str(md.blackElo)
    if md.opening_eco:
        game.headers["ECO"] = md.opening_eco
    if md.opening:
        game.headers["Opening"] = md.opening
    game.headers["Annotator"] = "automatic-chess-annotator"

    board = chess.Board()
    node: chess.pgn.GameNode = game
    for move in gj.moves:
        try:
            mv = chess.Move.from_uci(move.uci)
        except Exception:
            logger.warning("PGN export: bad uci %s — stopping", move.uci)
            break
        if mv not in board.legal_moves:
            logger.warning("PGN export: illegal %s at %s — stopping", move.uci, board.fen())
            break
        board_before = board.copy(stack=False)
        parent = node
        node = node.add_main_variation(mv)
        board.push(mv)

        nag = _nag_for_move(move)
        if nag:
            node.nags.add(nag)

        comment_bits: List[str] = []
        ev = _eval_tag(move)
        if ev:
            comment_bits.append(ev)

        if move.comment:
            plain, pv_lines = _flatten_tokens(move.comment)
            if plain:
                comment_bits.append(plain)
            for sans in pv_lines:
                if not sans:
                    continue
                if sans[0] == move.san:
                    # The displayed continuation (starts with the played move):
                    # attach the reply line after the move.
                    _try_add_line(board, node, sans[1:])
                else:
                    # An alternative to the played move (e.g. "Better was ..."):
                    # a true variation at the same point.
                    _try_add_line(board_before, parent, sans)

        if include_features and move.comment:
            fn = _feature_note(move)
            if fn:
                comment_bits.append(fn)

        if comment_bits:
            node.comment = " ".join(comment_bits).strip()

    out = io.StringIO()
    exporter = chess.pgn.FileExporter(out)
    game.accept(exporter)
    return out.getvalue()
