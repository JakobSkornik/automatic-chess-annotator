"""Knight outposts (Stockfish ``outpost`` restricted to knights).

A knight on an outpost square — supported by a friendly pawn and immune to enemy
pawn attack. (Stockfish also rewards bishops on outposts; we keep this to knights
to match the existing "strong knight" commentary.)
"""

from __future__ import annotations

import chess

from ._squares import is_outpost_square


def count(board: chess.Board, color: chess.Color) -> int:
    """Number of knights of ``color`` standing on an outpost square."""
    return sum(
        1
        for square in board.pieces(chess.KNIGHT, color)
        if is_outpost_square(board, square, color)
    )
