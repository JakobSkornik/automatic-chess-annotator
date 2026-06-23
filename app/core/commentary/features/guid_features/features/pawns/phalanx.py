"""Pawn phalanx (Stockfish ``phalanx``).

A pawn is part of a phalanx when a friendly pawn stands beside it on an adjacent
file at the same rank.
"""

from __future__ import annotations

import chess

from .._board import is_pawn


def count(board: chess.Board, color: chess.Color) -> int:
    """Number of pawns standing in a phalanx for ``color``."""
    total = 0
    for square in board.pieces(chess.PAWN, color):
        file, rank = chess.square_file(square), chess.square_rank(square)
        if is_pawn(board, file - 1, rank, color) or is_pawn(
            board, file + 1, rank, color
        ):
            total += 1
    return total
