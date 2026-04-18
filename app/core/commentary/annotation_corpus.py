"""Helpers for annotated-PGN BM25 corpus (comment quality + distance to next annotation)."""

from __future__ import annotations

import re
from typing import Dict, Optional

import chess.pgn

_JUNK_SYMBOLS = re.compile(r"^[\?\!\+\-=\u00b1\u2213\u221e\s]+$")


def is_junk_comment(text: str) -> bool:
    """Drop NAG-only / too-short / bibliographic junk from corpus."""
    t = text.strip()
    if len(t) < 25:
        return True
    if _JUNK_SYMBOLS.match(t):
        return True
    if t.upper() in {"D", "RR", "TN", "N"}:
        return True
    return False


def nearest_annotation_distance(
    ply_to_comment: dict[int, str],
    current_ply: int,
    max_delta: int = 4,
) -> Optional[int]:
    """Smallest Δ in [0, max_delta] with a non-junk comment at ply+Δ."""
    for d in range(0, max_delta + 1):
        txt = ply_to_comment.get(current_ply + d)
        if txt and not is_junk_comment(txt):
            return d
    return None


def ply_to_comment_map_from_game(game: chess.pgn.Game) -> Dict[int, str]:
    """
    Map half-move ply -> non-junk comment text from PGN {...} comments.
    Ply counts half-moves from the start (1 after the first half-move).
    """
    out: Dict[int, str] = {}
    root_comment = (game.comment or "").strip()
    if root_comment and not is_junk_comment(root_comment):
        out[0] = root_comment
    ply = 0
    node = game
    while not node.is_end():
        nxt = node.variation(0)
        if nxt.move is None:
            break
        ply += 1
        node = nxt
        raw = (node.comment or "").strip()
        if raw and not is_junk_comment(raw):
            out[ply] = raw
    return out
