"""Color-relative board lookups shared by the feature predicates.

The Stockfish guide is written from White's perspective with the y-axis pointing
down the board; these helpers express the same ideas in python-chess squares,
parameterised by colour so one predicate serves both sides.
"""

from __future__ import annotations

import chess


def is_pawn(board: chess.Board, file: int, rank: int, color: chess.Color) -> bool:
    """A pawn of ``color`` stands on (file, rank), if that square is on the board."""
    if not (0 <= file < 8 and 0 <= rank < 8):
        return False
    piece = board.piece_at(chess.square(file, rank))
    return piece is not None and piece.piece_type == chess.PAWN and piece.color == color


def forward_step(color: chess.Color) -> int:
    """Rank increment in the direction the colour's pawns advance (toward the foe)."""
    return 1 if color == chess.WHITE else -1


def relative_rank(square: int, color: chess.Color) -> int:
    """0-based rank from ``color``'s side (0 = own back rank, 7 = far side)."""
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank
