"""Doubled pawns (Stockfish ``doubled``).

A pawn is doubled when a friendly pawn stands directly behind it on the same
file and it is *not* supported by a friendly pawn on either diagonal behind it.
"""

from __future__ import annotations

import chess

from .._board import forward_step, is_pawn


def count(board: chess.Board, color: chess.Color) -> int:
    """Number of (unsupported) doubled pawns for ``color``."""
    behind = -forward_step(color)
    total = 0
    for square in board.pieces(chess.PAWN, color):
        file, rank = chess.square_file(square), chess.square_rank(square)
        if not is_pawn(board, file, rank + behind, color):
            continue
        supported = is_pawn(board, file - 1, rank + behind, color) or is_pawn(
            board, file + 1, rank + behind, color
        )
        if supported:
            continue
        total += 1
    return total
