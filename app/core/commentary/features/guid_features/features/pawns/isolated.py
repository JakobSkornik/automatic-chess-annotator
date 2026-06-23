"""Isolated pawns (Stockfish ``isolated``).

A pawn is isolated when there is no friendly pawn on either adjacent file.
"""

from __future__ import annotations

import chess

from .._board import is_pawn


def _has_neighbour_on_adjacent_file(
    board: chess.Board, file: int, color: chess.Color
) -> bool:
    return any(
        is_pawn(board, file - 1, rank, color) or is_pawn(board, file + 1, rank, color)
        for rank in range(8)
    )


def count(board: chess.Board, color: chess.Color) -> int:
    """Number of isolated pawns for ``color``."""
    total = 0
    for square in board.pieces(chess.PAWN, color):
        file = chess.square_file(square)
        if not _has_neighbour_on_adjacent_file(board, file, color):
            total += 1
    return total
