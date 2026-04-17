"""
Inline annotation tokens for LLM commentary: [type:content] syntax.

Resolved token payloads are JSON-serializable dicts for the frontend.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import chess

TOKEN_RE = re.compile(r"\[(\w+):([^\]]+)\]")

# Conservative SAN-like tokens for auto_tokenize (validated with Board.parse_san)
_SAN_CANDIDATE_RE = re.compile(
    r"\b(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b"
)
_EVAL_CANDIDATE_RE = re.compile(r"[+-]\d+\.\d+")

# Types we recognize; unknown types are left as plain text in resolved_tokens with data=None
KNOWN_TYPES = frozenset({"pv", "move", "square", "file", "eval", "piece"})


def parse_tokens(text: str) -> List[Dict[str, Any]]:
    """Find all [type:content] tokens in text."""
    out: List[Dict[str, Any]] = []
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


def _split_san_moves(content: str) -> List[str]:
    """Split PV content on whitespace into SAN tokens."""
    return [p for p in (content or "").split() if p]


def _resolve_pv_line(content: str, start_fen: str) -> Optional[List[Dict[str, str]]]:
    """Replay space-separated SAN from start_fen; return list of {san, fen, from, to}."""
    moves = _split_san_moves(content)
    if not moves:
        return None
    try:
        board = chess.Board(start_fen)
    except Exception:
        return None
    out: List[Dict[str, str]] = []
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


def _resolve_single_move(content: str, fen_before: str) -> Optional[Dict[str, str]]:
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


def _parse_square(content: str) -> Optional[str]:
    s = (content or "").strip().lower()
    if len(s) != 2:
        return None
    if s[0] not in "abcdefgh" or s[1] not in "12345678":
        return None
    return s


def _parse_file(content: str) -> Optional[str]:
    s = (content or "").strip().lower()
    if len(s) != 1 or s not in "abcdefgh":
        return None
    return s


def _parse_eval(content: str) -> Optional[float]:
    t = (content or "").strip().replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def _parse_piece_square(content: str) -> Optional[Dict[str, str]]:
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
) -> List[Dict[str, Any]]:
    """
    Parse and resolve all tokens in `text`.

    fen_before: position before the move being commented (for [move:]).
    fen_after: position after the played move (for [pv:] continuations).

    Each item: {type, raw, start, end, content, data?}
    `data` is None if unknown type or resolution failed.
    """
    parsed = parse_tokens(text)
    resolved: List[Dict[str, Any]] = []
    default_fen = chess.Board().fen()
    fb = fen_before or default_fen
    fa = fen_after or default_fen

    for p in parsed:
        t = p["type"]
        content = p["content"]
        raw = p["raw"]
        start = p["start"]
        end = p["end"]
        entry: Dict[str, Any] = {
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

        data: Optional[Dict[str, Any]] = None
        try:
            if t == "pv":
                line = _resolve_pv_line(content, fa)
                if line:
                    data = {"line": line}
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
    fen_before: Optional[str],
    fen_after: Optional[str],
) -> List[Dict[str, Any]]:
    """Convenience wrapper with safe defaults."""
    return resolve_tokens(
        text,
        fen_before or chess.Board().fen(),
        fen_after or chess.Board().fen(),
    )


def _existing_token_spans(text: str) -> List[Tuple[int, int]]:
    return [(p["start"], p["end"]) for p in parse_tokens(text)]


def _spans_overlap(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def _san_legal_on_fen(fen: str, san: str) -> bool:
    try:
        board = chess.Board(fen)
        board.parse_san(san)
        return True
    except Exception:
        return False


def auto_tokenize(text: str, fen_before: str, fen_after: str) -> str:
    """
    Wrap bare SAN and signed eval numbers in [move:...] / [eval:...] outside existing tokens.

    Tries each SAN candidate on ``fen_before`` first, then ``fen_after``.
    """
    if not text or not text.strip():
        return text or ""
    fb = fen_before or chess.Board().fen()
    fa = fen_after or chess.Board().fen()
    spans = _existing_token_spans(text)
    edits: List[Tuple[int, int, str]] = []

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

    edits.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s, e, repl in edits:
        out = out[:s] + repl + out[e:]
    return out
