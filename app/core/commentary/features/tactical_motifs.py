"""Heuristic tactical motif detection from board state before/after a move."""

from __future__ import annotations

from typing import List, Optional, Set

import chess

from app.models.chess_events import TacticalMotif

MATE_SCORE = 1_000_000

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _material_sum(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color:
            total += PIECE_VALUES.get(p.piece_type, 0)
    return total


def _fork_after_move(board_after: chess.Board, moved_to: int, mover_color: chess.Color) -> bool:
    """Moved piece attacks two or more valuable enemy targets (incl. king + piece)."""
    enemy = not mover_color
    count = 0
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if not p or p.color != enemy:
            continue
        if sq not in board_after.attacks(moved_to):
            continue
        if not board_after.is_attacked_by(mover_color, sq):
            continue
        if p.piece_type == chess.KING or PIECE_VALUES.get(p.piece_type, 0) >= 3:
            count += 1
    return count >= 2


def _back_rank_threat(board_after: chess.Board, mover_color: chess.Color) -> bool:
    enemy = not mover_color
    king_sq = board_after.king(enemy)
    if king_sq is None:
        return False
    kr = chess.square_rank(king_sq)
    if enemy == chess.WHITE and kr != 0:
        return False
    if enemy == chess.BLACK and kr != 7:
        return False
    if not board_after.is_attacked_by(mover_color, king_sq):
        return False
    for sq in chess.SQUARES:
        if chess.square_rank(sq) != kr:
            continue
        p = board_after.piece_at(sq)
        if (
            p
            and p.color == mover_color
            and p.piece_type in (chess.ROOK, chess.QUEEN)
            and sq in board_after.attackers(mover_color, king_sq)
        ):
            return True
    return False


def detect_tactical_motifs(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    eval_before_cp: Optional[int],
    eval_after_cp: Optional[int],
) -> List[TacticalMotif]:
    """Return detected tactical motifs for the played move."""
    motifs: List[TacticalMotif] = []
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
    if mat_after < mat_before - 1 and eval_before_cp is not None and eval_after_cp is not None:
        is_white = moved_color == chess.WHITE
        gain = (eval_after_cp - eval_before_cp) if is_white else (eval_before_cp - eval_after_cp)
        if gain >= 50:
            motifs.append(TacticalMotif.SACRIFICE)

    seen: Set[TacticalMotif] = set()
    out: List[TacticalMotif] = []
    for m in motifs:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
