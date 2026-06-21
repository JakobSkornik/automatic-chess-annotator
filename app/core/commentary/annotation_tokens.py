"""
Inline annotation tokens for LLM commentary: [type:content] syntax.

Resolved token payloads are JSON-serializable dicts for the frontend.
"""

from __future__ import annotations

import re
from typing import Any

import chess

TOKEN_RE = re.compile(r"\[(\w+):([^\]]+)\]")

# Conservative SAN-like tokens for auto_tokenize (validated with Board.parse_san)
_SAN_CANDIDATE_RE = re.compile(
    r"\b(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b"
)
_EVAL_CANDIDATE_RE = re.compile(r"[+-]\d+\.\d+")
_FILE_CANDIDATE_RE = re.compile(r"\b(?:the\s+)?([a-hA-H])-file\b")

# Types we recognize; unknown types are left as plain text in resolved_tokens with data=None
KNOWN_TYPES = frozenset({"pv", "move", "square", "file", "eval", "piece"})


def parse_tokens(text: str) -> list[dict[str, Any]]:
    """Find all [type:content] tokens in text."""
    out: list[dict[str, Any]] = []
    for m in TOKEN_RE.finditer(text or ""):
        out.append(
            {
                "type": m.group(1).lower(),
                "raw": m.group(0),
                "content": m.group(2).strip(),
                "start": m.start(),
                "end": m.end(),
            }
        )
    return out


def _split_san_moves(content: str) -> list[str]:
    """Split PV content on whitespace into SAN tokens."""
    return [p for p in (content or "").split() if p]


def _resolve_pv_line(content: str, start_fen: str) -> list[dict[str, str]] | None:
    """Replay space-separated SAN from start_fen; return list of {san, fen, from, to}."""
    moves = _split_san_moves(content)
    if not moves:
        return None
    try:
        board = chess.Board(start_fen)
    except Exception:
        return None
    out: list[dict[str, str]] = []
    for san in moves:
        try:
            mv = board.parse_san(san)
            uci = mv.uci()
            from_sq, to_sq = uci[:2], uci[2:4]
            board.push(mv)
            out.append(
                {
                    "san": san,
                    "fen": board.fen(),
                    "from": from_sq,
                    "to": to_sq,
                }
            )
        except Exception:
            break
    return out if out else None


def _resolve_single_move(content: str, fen_before: str) -> dict[str, str] | None:
    """Parse one SAN on fen_before; return from, to, san."""
    san = (content or "").strip().split()
    if len(san) != 1:
        return None
    try:
        board = chess.Board(fen_before)
        mv = board.parse_san(san[0])
        uci = mv.uci()
        from_sq, to_sq = uci[:2], uci[2:4]
        return {"san": san[0], "from": from_sq, "to": to_sq}
    except Exception:
        return None


def _parse_square(content: str) -> str | None:
    s = (content or "").strip().lower()
    if len(s) != 2:
        return None
    if s[0] not in "abcdefgh" or s[1] not in "12345678":
        return None
    return s


def _parse_file(content: str) -> str | None:
    s = (content or "").strip().lower()
    if len(s) != 1 or s not in "abcdefgh":
        return None
    return s


def _parse_eval(content: str) -> float | None:
    t = (content or "").strip().replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


# Piece letter + a 2-char square name, e.g. 'Nd7'.
MIN_PIECE_SQUARE_LEN = 3


def _parse_piece_square(content: str) -> dict[str, str] | None:
    """
    Expect piece letter (PNBRQK) + square, e.g. Nd7, Bg5.
    Highlights the destination square.
    """
    c = (content or "").strip()
    if len(c) < MIN_PIECE_SQUARE_LEN:
        return None
    piece = c[0].upper()
    if piece not in "PNBRQK":
        return None
    sq = c[1:].lower()
    if len(sq) != 2:
        return None
    if sq[0] not in "abcdefgh" or sq[1] not in "12345678":
        return None
    return {"piece": piece, "square": sq}


def _resolve_pv_token(
    content: str, fen_before: str, fen_after: str
) -> dict[str, Any] | None:
    """Guid display lines start WITH the played move (replay from fen_before);
    plain continuations start after it (fen_after). Keep whichever covers more."""
    best_fen: str | None = None
    best_line: list[dict[str, str]] | None = None
    for candidate_fen in (fen_before, fen_after):
        line = _resolve_pv_line(content, candidate_fen)
        if line and (best_line is None or len(line) > len(best_line)):
            best_fen, best_line = candidate_fen, line
    return {"line": best_line, "start_fen": best_fen} if best_line else None


def _resolve_token_data(
    t: str, content: str, fen_before: str, fen_after: str
) -> dict[str, Any] | None:
    """Resolve one known token to its interactive ``data`` payload (or None)."""
    if t == "pv":
        return _resolve_pv_token(content, fen_before, fen_after)
    if t == "move":
        return _resolve_single_move(content, fen_before) or None
    if t == "square":
        sq = _parse_square(content)
        return {"square": sq} if sq else None
    if t == "file":
        f = _parse_file(content)
        return {"file": f} if f else None
    if t == "eval":
        ev = _parse_eval(content)
        return {"pawns": ev} if ev is not None else None
    if t == "piece":
        return _parse_piece_square(content) or None
    return None


def resolve_tokens(
    text: str,
    fen_before: str,
    fen_after: str,
) -> list[dict[str, Any]]:
    """
    Parse and resolve all tokens in `text`.

    fen_before: position before the move being commented (for [move:]).
    fen_after: position after the played move (for [pv:] continuations).

    Each item: {type, raw, start, end, content, data?}
    `data` is None if unknown type or resolution failed.
    """
    default_fen = chess.Board().fen()
    fen_before = fen_before or default_fen
    fen_after = fen_after or default_fen

    resolved: list[dict[str, Any]] = []
    for p in parse_tokens(text):
        t = p["type"]
        entry: dict[str, Any] = {
            "type": t,
            "raw": p["raw"],
            "content": p["content"],
            "start": p["start"],
            "end": p["end"],
            "data": None,
        }
        if t in KNOWN_TYPES:
            try:
                entry["data"] = _resolve_token_data(
                    t, p["content"], fen_before, fen_after
                )
            except Exception:
                entry["data"] = None
        resolved.append(entry)
    return resolved


def resolve_tokens_for_comment(
    text: str,
    fen_before: str | None,
    fen_after: str | None,
) -> list[dict[str, Any]]:
    """Convenience wrapper with safe defaults."""
    return resolve_tokens(
        text,
        fen_before or chess.Board().fen(),
        fen_after or chess.Board().fen(),
    )
