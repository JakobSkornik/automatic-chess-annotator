"""Hanging pieces (Stockfish ``hanging``).

A weak enemy that is either undefended, or a non-pawn we attack more than once.
"""

from __future__ import annotations

import chess

from ._attack import attacker_count
from .weak_enemies import is_weak_enemy


def is_hanging(board: chess.Board, square: int, attacker: chess.Color) -> bool:
    """Whether the enemy piece on ``square`` hangs to ``attacker``."""
    if not is_weak_enemy(board, square, attacker):
        return False
    enemy = not attacker
    piece = board.piece_at(square)
    if piece.piece_type != chess.PAWN and attacker_count(board, attacker, square) > 1:
        return True
    return attacker_count(board, enemy, square) == 0


def is_concretely_hanging(
    board: chess.Board, square: int, attacker: chess.Color
) -> bool:
    """Conservative evidence for saying an enemy piece is hanging.

    Require an undefended non-king target and at least one legal capture. The
    feature count below remains the broader positional pressure heuristic; a
    commentary claim needs evidence that the capture is actually available.
    """
    piece = board.piece_at(square)
    if piece is None or piece.color == attacker or piece.piece_type == chess.KING:
        return False
    enemy = not attacker
    if attacker_count(board, enemy, square) != 0:
        return False
    # A target that can safely take its attacker (queen against queen) is not hanging.
    from app.core.commentary.rules.tactical_evidence import takes_attacker_safely
    if takes_attacker_safely(board, square, board.attackers(attacker, square)):
        return False

    # board.attackers() is geometric, so a pinned piece can appear to attack
    # the target even though moving it would expose its own king. Test capture
    # legality with the attacking side to move, including when this is a
    # non-moving-side static comparison.
    probe = board.copy(stack=False)
    if probe.turn != attacker:
        probe.ep_square = None
    probe.turn = attacker
    return any(move.to_square == square for move in probe.generate_legal_captures())


def squares(board: chess.Board, attacker: chess.Color) -> list[str]:
    """Squares of undefended non-king pieces with a legal capture available."""
    enemy = not attacker
    return [
        chess.square_name(square)
        for square in chess.SquareSet(board.occupied_co[enemy])
        if is_concretely_hanging(board, square, attacker)
    ]


def count(board: chess.Board, attacker: chess.Color) -> int:
    """Number of hanging enemy pieces from ``attacker``'s side."""
    enemy = not attacker
    return sum(
        1
        for square in chess.SquareSet(board.occupied_co[enemy])
        if is_hanging(board, square, attacker)
    )
