"""Rook on an open or semi-open file (Stockfish ``rook_on_file``).

Per rook: 0 if a friendly pawn shares its file, 1 if only an enemy pawn does
(semi-open), 2 if the file is fully open. The helpers expose the open and
semi-open rook counts the commentary uses.
"""

from __future__ import annotations

import chess

_BLOCKED = 0
_SEMI_OPEN = 1
_OPEN = 2


def _file_status(board: chess.Board, square: int, color: chess.Color) -> int:
    """0 blocked by own pawn, 1 semi-open (enemy pawn only), 2 fully open."""
    file = chess.square_file(square)
    has_enemy_pawn = False
    for rank in range(8):
        piece = board.piece_at(chess.square(file, rank))
        if piece is None or piece.piece_type != chess.PAWN:
            continue
        if piece.color == color:
            return _BLOCKED
        has_enemy_pawn = True
    return _SEMI_OPEN if has_enemy_pawn else _OPEN


def open_file_count(board: chess.Board, color: chess.Color) -> int:
    """Rooks of ``color`` on a fully open file."""
    return sum(
        1
        for rook in board.pieces(chess.ROOK, color)
        if _file_status(board, rook, color) == _OPEN
    )


def semi_open_file_count(board: chess.Board, color: chess.Color) -> int:
    """Rooks of ``color`` on a semi-open file."""
    return sum(
        1
        for rook in board.pieces(chess.ROOK, color)
        if _file_status(board, rook, color) == _SEMI_OPEN
    )
