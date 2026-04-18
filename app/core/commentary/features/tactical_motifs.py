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


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _skewer_heuristic(board_after: chess.Board, moved_to: int, mover_color: chess.Color) -> bool:
    """Moved piece attacks a valuable enemy unit with a more valuable one further along the ray."""
    enemy = not mover_color
    for tgt_sq in chess.SQUARES:
        p = board_after.piece_at(tgt_sq)
        if not p or p.color != enemy or p.piece_type == chess.KING:
            continue
        if tgt_sq not in board_after.attacks(moved_to):
            continue
        v1 = PIECE_VALUES.get(p.piece_type, 0)
        if v1 < 3:
            continue
        df = _sign(chess.square_file(tgt_sq) - chess.square_file(moved_to))
        dr = _sign(chess.square_rank(tgt_sq) - chess.square_rank(moved_to))
        if df not in (-1, 0, 1) or dr not in (-1, 0, 1) or (df == 0 and dr == 0):
            continue
        f, r = chess.square_file(tgt_sq), chess.square_rank(tgt_sq)
        for _ in range(8):
            f += df
            r += dr
            if not (0 <= f <= 7 and 0 <= r <= 7):
                break
            sq = chess.square(f, r)
            p2 = board_after.piece_at(sq)
            if p2 and p2.color == enemy:
                v2 = PIECE_VALUES.get(p2.piece_type, 0)
                if v2 > v1:
                    return True
                break
    return False


def _removal_of_guard(board_before: chess.Board, move: chess.Move) -> bool:
    """Capture removes a defender that was covering a valuable enemy unit."""
    if not board_before.is_capture(move):
        return False
    victim_sq = move.to_square
    victim = board_before.piece_at(victim_sq)
    if victim is None or victim.piece_type == chess.KING:
        return False
    enemy = victim.color
    for tgt in chess.SQUARES:
        p = board_before.piece_at(tgt)
        if not p or p.color != enemy or tgt == victim_sq:
            continue
        if PIECE_VALUES.get(p.piece_type, 0) < 3 and p.piece_type != chess.KING:
            continue
        if victim_sq in board_before.attackers(enemy, tgt):
            return True
    return False


def _deflection_attack(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> bool:
    """Moved piece creates a new attack on a valuable enemy unit (fork-like but one target)."""
    enemy = not mover_color
    piece_after = board_after.piece_at(move.to_square)
    if piece_after is None:
        return False
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if not p or p.color != enemy or p.piece_type == chess.KING:
            continue
        if sq not in board_after.attacks(move.to_square):
            continue
        if PIECE_VALUES.get(p.piece_type, 0) < 3:
            continue
        if sq not in board_before.attacks(move.from_square):
            return True
    return False


def _overloaded_defender(board_before: chess.Board, move: chess.Move) -> bool:
    """Captured piece was defending multiple valuable enemy units."""
    if not board_before.is_capture(move):
        return False
    victim_sq = move.to_square
    victim = board_before.piece_at(victim_sq)
    if victim is None or victim.piece_type == chess.KING:
        return False
    enemy = victim.color
    n = 0
    for tgt in chess.SQUARES:
        p = board_before.piece_at(tgt)
        if not p or p.color != enemy or tgt == victim_sq:
            continue
        if PIECE_VALUES.get(p.piece_type, 0) < 3 and p.piece_type != chess.KING:
            continue
        if victim_sq in board_before.attackers(enemy, tgt):
            n += 1
    return n >= 2


def _decoy_sacrifice(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> bool:
    """Material lost but creates a concrete threat to king or queen next."""
    mat_before = _material_sum(board_before, mover_color)
    mat_after = _material_sum(board_after, mover_color)
    if mat_after >= mat_before - 1:
        return False
    if board_after.is_check() or board_after.is_checkmate():
        return True
    enemy = not mover_color
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if p and p.color == enemy and p.piece_type == chess.QUEEN:
            if board_after.is_attacked_by(mover_color, sq):
                return True
    return False


def _interference_block(board_before: chess.Board, move: chess.Move, mover_color: chess.Color) -> bool:
    """Piece lands on a square between a sliding attacker and a valuable target."""
    enemy = not mover_color
    piece = board_before.piece_at(move.from_square)
    if not piece or piece.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.PAWN):
        return False
    inter_sq = move.to_square
    for atk_sq in chess.SQUARES:
        ap = board_before.piece_at(atk_sq)
        if not ap or ap.color != enemy or ap.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        for tgt_sq in chess.SQUARES:
            tp = board_before.piece_at(tgt_sq)
            if not tp or tp.color != mover_color:
                continue
            if PIECE_VALUES.get(tp.piece_type, 0) < 3 and tp.piece_type != chess.KING:
                continue
            between_bb = chess.between(atk_sq, tgt_sq)
            if inter_sq in chess.SquareSet(between_bb):
                return True
    return False


def _zwischenzug_check(board_before: chess.Board, board_after: chess.Board, move: chess.Move) -> bool:
    """Check with a piece that did not attack the enemy king before the move (intermezzo)."""
    if board_before.is_capture(move):
        return False
    if not board_after.is_check():
        return False
    enemy = not board_before.turn
    ek = board_before.king(enemy)
    if ek is None:
        return False
    return ek not in board_before.attacks(move.from_square)


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

    if _interference_block(board_before, move, moved_color):
        motifs.append(TacticalMotif.INTERFERENCE)

    if _zwischenzug_check(board_before, board_after, move):
        motifs.append(TacticalMotif.ZWISCHENZUG)

    seen: Set[TacticalMotif] = set()
    out: List[TacticalMotif] = []
    for m in motifs:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
