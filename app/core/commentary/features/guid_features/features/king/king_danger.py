"""King danger, ported from the dominant terms of Stockfish ``king_danger``.

Stockfish's full king-danger is a tuned polynomial over ~10 sub-terms (attacker
count/weight, attacks on the king, weak squares, safe checks, pawn shelter,
flank pressure, mobility). We port its attacking core — the terms that dominate
when a king actually comes under fire — and omit the shelter/flank/safe-check
refinements:

    danger = attackers * weight + 69 * king_attacks + 185 * weak_squares

The result is 0 when the king is not under attack (matching Stockfish, whose
danger only registers once real pressure exists). Slider x-rays are approximated
by direct attacks.
"""

from __future__ import annotations

import chess

# Stockfish king_attackers_weight by piece type, and the king_danger multipliers.
_ATTACKER_WEIGHT = {
    chess.KNIGHT: 81,
    chess.BISHOP: 52,
    chess.ROOK: 44,
    chess.QUEEN: 10,
}
_KING_ATTACKS_CP = 69
_WEAK_SQUARE_CP = 185
_MAX_DEFENDERS_FOR_WEAK = 1


def _king_ring(king: int) -> chess.SquareSet:
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king])
    ring.add(king)
    return ring


def _attacker_pressure(
    board: chess.Board, enemy: chess.Color, ring: chess.SquareSet
) -> tuple[int, int]:
    """(number of enemy pieces attacking the king ring, sum of their weights)."""
    count = 0
    weight = 0
    for piece_type, piece_weight in _ATTACKER_WEIGHT.items():
        for square in board.pieces(piece_type, enemy):
            if board.attacks(square) & ring:
                count += 1
                weight += piece_weight
    return count, weight


def _attacks_on_king(board: chess.Board, enemy: chess.Color, king: int) -> int:
    """Enemy minor/major attacks on squares adjacent to the king (with multiplicity)."""
    adjacent = chess.SquareSet(chess.BB_KING_ATTACKS[king])
    return sum(
        1
        for square in adjacent
        for attacker in board.attackers(enemy, square)
        if board.piece_type_at(attacker) in _ATTACKER_WEIGHT
    )


def _weak_squares(board: chess.Board, color: chess.Color, ring: chess.SquareSet) -> int:
    """King-ring squares the enemy attacks that we defend at most once."""
    enemy = not color
    return sum(
        1
        for square in ring
        if board.attackers(enemy, square)
        and len(board.attackers(color, square)) <= _MAX_DEFENDERS_FOR_WEAK
    )


def king_danger(board: chess.Board, color: chess.Color) -> int:
    """Danger to ``color``'s own king (0 when it is not under attack)."""
    king = board.king(color)
    if king is None:
        return 0
    enemy = not color
    ring = _king_ring(king)
    count, weight = _attacker_pressure(board, enemy, ring)
    if count == 0:
        return 0
    king_attacks = _attacks_on_king(board, enemy, king)
    weak = _weak_squares(board, color, ring)
    return count * weight + _KING_ATTACKS_CP * king_attacks + _WEAK_SQUARE_CP * weak


def king_zone_attacks(board: chess.Board, color: chess.Color) -> int:
    """Number of ``color``'s minor/major pieces attacking the ENEMY king's
    zone (Stockfish king-zone notion). Always computable — unlike ``king_danger``
    this registers pressure even before it becomes 'danger' (no attacker-weight
    gate), which is what an attack-arc narrative needs."""
    enemy = not color
    king = board.king(enemy)
    if king is None:
        return 0
    ring = _king_ring(king)
    return sum(
        1
        for piece_type in _ATTACKER_WEIGHT
        for square in board.pieces(piece_type, color)
        if board.attacks(square) & ring
    )
