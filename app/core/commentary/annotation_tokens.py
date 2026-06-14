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


def _parse_piece_square(content: str) -> dict[str, str] | None:
    """
    Expect piece letter (PNBRQK) + square, e.g. Nd7, Bg5.
    Highlights the destination square.
    """
    c = (content or "").strip()
    if len(c) < 3:
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
    parsed = parse_tokens(text)
    resolved: list[dict[str, Any]] = []
    default_fen = chess.Board().fen()
    fb = fen_before or default_fen
    fa = fen_after or default_fen

    for p in parsed:
        t = p["type"]
        content = p["content"]
        raw = p["raw"]
        start = p["start"]
        end = p["end"]
        entry: dict[str, Any] = {
            "type": t,
            "raw": raw,
            "content": content,
            "start": start,
            "end": end,
            "data": None,
        }

        if t not in KNOWN_TYPES:
            resolved.append(entry)
            continue

        data: dict[str, Any] | None = None
        try:
            if t == "pv":
                # Guid display lines start WITH the played move (replay from
                # fen_before); plain continuations start after it (fen_after).
                # Try both and keep the resolution that covers more plies.
                best_fen: str | None = None
                best_line: list[dict[str, str]] | None = None
                for candidate_fen in (fb, fa):
                    line = _resolve_pv_line(content, candidate_fen)
                    if line and (best_line is None or len(line) > len(best_line)):
                        best_fen = candidate_fen
                        best_line = line
                if best_line:
                    data = {"line": best_line, "start_fen": best_fen}
            elif t == "move":
                m = _resolve_single_move(content, fb)
                if m:
                    data = m
            elif t == "square":
                sq = _parse_square(content)
                if sq:
                    data = {"square": sq}
            elif t == "file":
                f = _parse_file(content)
                if f:
                    data = {"file": f}
            elif t == "eval":
                ev = _parse_eval(content)
                if ev is not None:
                    data = {"pawns": ev}
            elif t == "piece":
                ps = _parse_piece_square(content)
                if ps:
                    data = ps
        except Exception:
            data = None

        entry["data"] = data
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


def _existing_token_spans(text: str) -> list[tuple[int, int]]:
    return [(p["start"], p["end"]) for p in parse_tokens(text)]


def _spans_overlap(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and end > s for s, e in spans)


def _san_legal_on_fen(fen: str, san: str) -> bool:
    try:
        board = chess.Board(fen)
        board.parse_san(san)
        return True
    except Exception:
        return False


def auto_tokenize(text: str, fen_before: str, fen_after: str) -> str:
    """
    Wrap bare SAN, signed eval numbers, and file phrases (e.g. "the e-file") in
    [move:...] / [eval:...] / [file:x] outside existing bracket tokens.

    Tries each SAN candidate on ``fen_before`` first, then ``fen_after``.
    """
    if not text or not text.strip():
        return text or ""
    fb = fen_before or chess.Board().fen()
    fa = fen_after or chess.Board().fen()
    spans = _existing_token_spans(text)
    edits: list[tuple[int, int, str]] = []

    for m in _EVAL_CANDIDATE_RE.finditer(text):
        s, e = m.start(), m.end()
        if _spans_overlap(s, e, spans):
            continue
        prefix = text[:s].rstrip()
        prev_tokens = prefix.split()
        last = prev_tokens[-1].lower() if prev_tokens else ""
        if last in ("move", "ply"):
            continue
        edits.append((s, e, f"[eval:{m.group(0)}]"))

    for m in _SAN_CANDIDATE_RE.finditer(text):
        s, e = m.start(), m.end()
        if _spans_overlap(s, e, spans):
            continue
        cand = m.group(0)
        if _san_legal_on_fen(fb, cand) or _san_legal_on_fen(fa, cand):
            edits.append((s, e, f"[move:{cand}]"))

    for m in _FILE_CANDIDATE_RE.finditer(text):
        s, e = m.start(), m.end()
        if _spans_overlap(s, e, spans):
            continue
        letter = m.group(1).lower()
        edits.append((s, e, f"[file:{letter}]"))

    edits.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s, e, repl in edits:
        out = out[:s] + repl + out[e:]
    return out
