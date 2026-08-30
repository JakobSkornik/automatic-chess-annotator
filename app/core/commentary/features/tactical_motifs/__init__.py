"""Heuristic tactical motif detection from board state before/after a move."""

from __future__ import annotations

import chess

from app.models.chess_events import TacticalMotif

from .detectors import (
    MATE_SCORE,
    _back_rank_threat,
    _clearance_move,
    _decoy_sacrifice,
    _deflection_attack,
    _double_attack_non_fork,
    _fork_after_move,
    _interference_block,
    _overloaded_defender,
    _quiet_move_threatens_major,
    _removal_of_guard,
    _skewer_heuristic,
    _x_ray_attack,
    _zwischenzug_check,
)
from .sacrifice_core import (
    POSITIONAL_PAWN_SAC_MAX_CP,
    POSITIONAL_PAWN_SAC_MIN_CP,
    is_sacrifice as _is_sacrifice_core,
    settled_material_deficit,
)

__all__ = ["detect_tactical_motifs"]

# Piece types a "pin" motif may be claimed about, matching the PINS feature in
# guid_features: a pinned pawn is rarely worth annotating and reads wrong when
# the comment calls it a pinned piece.
_PINNABLE = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def _pinned_squares(board: chess.Board, color: chess.Color) -> set[int]:
    out: set[int] = set()
    for piece_type in _PINNABLE:
        for sq in board.pieces(piece_type, color):
            try:
                if board.is_pinned(color, sq):
                    out.add(sq)
            except Exception:
                continue
    return out


def _new_pin(
    board_before: chess.Board, board_after: chess.Board, enemy: chess.Color
) -> bool:
    """Whether the move *created* a pin, rather than one that already stood.

    Without the before/after comparison a long-standing pin is re-attributed to
    every later move, inflating criticality and inviting the renderer to credit
    the move with a bind it had nothing to do with.
    """
    after = _pinned_squares(board_after, enemy)
    if not after:
        return False
    return bool(after - _pinned_squares(board_before, enemy))


def detect_tactical_motifs(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    eval_before_cp: int | None,
    eval_after_cp: int | None,
) -> list[TacticalMotif]:
    """Return detected tactical motifs for the played move."""
    motifs: list[TacticalMotif] = []
    piece_after = board_after.piece_at(move.to_square)
    if piece_after is None:
        return motifs

    moved_color = piece_after.color

    if board_after.is_checkmate():
        motifs.append(TacticalMotif.MATING_NET)

    if board_after.is_check():
        checkers = board_after.checkers()
        if len(checkers) >= 2:
            motifs.append(TacticalMotif.DOUBLE_CHECK)
        elif move.to_square not in checkers:
            motifs.append(TacticalMotif.DISCOVERED_CHECK)

    if _fork_after_move(board_after, move.to_square, moved_color):
        motifs.append(TacticalMotif.FORK)

    if _back_rank_threat(board_after, moved_color):
        motifs.append(TacticalMotif.BACK_RANK_THREAT)

    if eval_after_cp is not None and abs(eval_after_cp) > MATE_SCORE - 1000:
        if TacticalMotif.MATING_NET not in motifs:
            motifs.append(TacticalMotif.MATING_NET)

    # SACRIFICE / POSITIONAL_PAWN_SAC share the settled-material-deficit core
    # (sacrifice_core.py): a persistent deficit at the settled/quiescent leaf
    # reached after the opponent's reply, with real compensation (the eval
    # floor), rather than the old same-half-move mat_before/mat_after diff
    # that could never fire (a single legal move can't reduce the mover's own
    # material). POSITIONAL_PAWN_SAC is the same mechanism narrowed to a
    # single pawn of persistent deficit and no bigger tactical follow-up
    # (i.e. SACRIFICE itself didn't already fire).
    if eval_after_cp is not None:
        settle_deficit = settled_material_deficit(board_before, board_after, moved_color)
        if _is_sacrifice_core(board_before, board_after, moved_color, eval_after_cp):
            motifs.append(TacticalMotif.SACRIFICE)
        elif (
            piece_after.piece_type == chess.PAWN
            and POSITIONAL_PAWN_SAC_MIN_CP <= settle_deficit < POSITIONAL_PAWN_SAC_MAX_CP
            and _is_sacrifice_core(
                board_before,
                board_after,
                moved_color,
                eval_after_cp,
                min_deficit_cp=POSITIONAL_PAWN_SAC_MIN_CP,
            )
        ):
            motifs.append(TacticalMotif.POSITIONAL_PAWN_SAC)

    enemy = not moved_color
    if _new_pin(board_before, board_after, enemy):
        motifs.append(TacticalMotif.PIN)

    if _skewer_heuristic(board_after, move.to_square, moved_color):
        motifs.append(TacticalMotif.SKEWER)

    if _removal_of_guard(board_before, move):
        motifs.append(TacticalMotif.REMOVAL_OF_GUARD)

    if _deflection_attack(board_before, board_after, move, moved_color):
        motifs.append(TacticalMotif.DEFLECTION)

    if _overloaded_defender(board_before, move):
        motifs.append(TacticalMotif.OVERLOADED_PIECE)

    if _decoy_sacrifice(board_before, board_after, move, moved_color):
        motifs.append(TacticalMotif.DECOY)

    if _interference_block(board_before, board_after, move, moved_color):
        motifs.append(TacticalMotif.INTERFERENCE)

    if _zwischenzug_check(board_before, board_after, move):
        motifs.append(TacticalMotif.ZWISCHENZUG)

    if _quiet_move_threatens_major(board_before, board_after, move, moved_color):
        motifs.append(TacticalMotif.QUIET_MOVE_THREAT)

    if _x_ray_attack(board_after, move.to_square, moved_color):
        motifs.append(TacticalMotif.X_RAY)

    if _double_attack_non_fork(board_after, move.to_square, moved_color):
        motifs.append(TacticalMotif.DOUBLE_ATTACK)

    if _clearance_move(board_before, move, moved_color):
        motifs.append(TacticalMotif.CLEARANCE)

    if board_before.is_capture(move) and board_before.is_attacked_by(
        enemy, move.from_square
    ):
        motifs.append(TacticalMotif.DESPERADO)

    if board_before.is_capture(move):
        victim = board_before.piece_at(move.to_square)
        attacker = board_before.piece_at(move.from_square)
        if (
            victim
            and victim.piece_type == chess.QUEEN
            and attacker
            and attacker.piece_type
            in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
        ):
            motifs.append(TacticalMotif.TRADE_TO_DEFUSE_ATTACK)

    seen: set[TacticalMotif] = set()
    out: list[TacticalMotif] = []
    for m in motifs:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
