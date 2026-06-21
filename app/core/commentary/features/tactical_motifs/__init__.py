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
    _material_sum,
    _overloaded_defender,
    _quiet_move_threatens_major,
    _removal_of_guard,
    _skewer_heuristic,
    _x_ray_attack,
    _zwischenzug_check,
)

__all__ = ["detect_tactical_motifs"]


SACRIFICE_GAIN_CP = 50  # material the side nets to flag a tactic
THREAT_DROP_CP = 40  # eval drop that flags a created threat


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

    mat_before = _material_sum(board_before, moved_color)
    mat_after = _material_sum(board_after, moved_color)
    if (
        mat_after < mat_before - 1
        and eval_before_cp is not None
        and eval_after_cp is not None
    ):
        is_white = moved_color == chess.WHITE
        gain = (
            (eval_after_cp - eval_before_cp)
            if is_white
            else (eval_before_cp - eval_after_cp)
        )
        if gain >= SACRIFICE_GAIN_CP:
            motifs.append(TacticalMotif.SACRIFICE)

    enemy = not moved_color
    for sq in chess.SQUARES:
        pie = board_after.piece_at(sq)
        if pie and pie.color == enemy and pie.piece_type != chess.KING:
            try:
                if board_after.is_pinned(enemy, sq):
                    motifs.append(TacticalMotif.PIN)
                    break
            except Exception:
                pass

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

    if (
        board_before.is_capture(move)
        and piece_after.piece_type == chess.PAWN
        and mat_after < mat_before
        and eval_before_cp is not None
        and eval_after_cp is not None
    ):
        is_w = moved_color == chess.WHITE
        drop = (
            (eval_after_cp - eval_before_cp)
            if is_w
            else (eval_before_cp - eval_after_cp)
        )
        if drop <= THREAT_DROP_CP:
            motifs.append(TacticalMotif.POSITIONAL_PAWN_SAC)

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
