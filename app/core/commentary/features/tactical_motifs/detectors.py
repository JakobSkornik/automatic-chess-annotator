"""Per-motif tactical heuristics (fork, skewer, deflection, x-ray, ...) plus the
material helpers they share."""

from __future__ import annotations

import chess

MATE_SCORE = 1_000_000

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


MINOR_PIECE_VALUE = 3  # value cutoff: a minor piece or better (knight/bishop+)
OPENING_FULLMOVE_MAX = 4  # zwischenzug heuristic only applies this early


def _material_sum(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color:
            total += PIECE_VALUES.get(p.piece_type, 0)
    return total


def _fork_after_move(
    board_after: chess.Board, moved_to: int, mover_color: chess.Color
) -> bool:
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
        if (
            p.piece_type == chess.KING
            or PIECE_VALUES.get(p.piece_type, 0) >= MINOR_PIECE_VALUE
        ):
            count += 1
    return count >= 2


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _skewer_heuristic(
    board_after: chess.Board, moved_to: int, mover_color: chess.Color
) -> bool:
    """Moved piece attacks a valuable enemy unit with a more valuable one further along the ray."""
    enemy = not mover_color
    for tgt_sq in chess.SQUARES:
        p = board_after.piece_at(tgt_sq)
        if not p or p.color != enemy or p.piece_type == chess.KING:
            continue
        if tgt_sq not in board_after.attacks(moved_to):
            continue
        v1 = PIECE_VALUES.get(p.piece_type, 0)
        if v1 < MINOR_PIECE_VALUE:
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
        if (
            PIECE_VALUES.get(p.piece_type, 0) < MINOR_PIECE_VALUE
            and p.piece_type != chess.KING
        ):
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
        if PIECE_VALUES.get(p.piece_type, 0) < MINOR_PIECE_VALUE:
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
        if (
            PIECE_VALUES.get(p.piece_type, 0) < MINOR_PIECE_VALUE
            and p.piece_type != chess.KING
        ):
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


def _interference_block(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> bool:
    """Piece lands on a square between a sliding attacker and a valuable target (clear ray)."""
    enemy = not mover_color
    piece = board_before.piece_at(move.from_square)
    if not piece or piece.piece_type not in (
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.PAWN,
    ):
        return False
    if (
        not board_before.is_capture(move)
        and not board_after.is_check()
        and board_before.fullmove_number <= OPENING_FULLMOVE_MAX
        and piece.piece_type in (chess.PAWN, chess.KNIGHT)
    ):
        return False
    inter_sq = move.to_square
    for atk_sq in chess.SQUARES:
        ap = board_before.piece_at(atk_sq)
        if (
            not ap
            or ap.color != enemy
            or ap.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN)
        ):
            continue
        for tgt_sq in chess.SQUARES:
            tp = board_before.piece_at(tgt_sq)
            if not tp or tp.color != mover_color:
                continue
            if (
                PIECE_VALUES.get(tp.piece_type, 0) < MINOR_PIECE_VALUE
                and tp.piece_type != chess.KING
            ):
                continue
            between_bb = chess.between(atk_sq, tgt_sq)
            if inter_sq not in chess.SquareSet(between_bb):
                continue
            blocked = False
            for sq in chess.SquareSet(between_bb):
                if sq == inter_sq:
                    continue
                if board_before.piece_at(sq) is not None:
                    blocked = True
                    break
            if blocked:
                continue
            return True
    return False


def _zwischenzug_check(
    board_before: chess.Board, board_after: chess.Board, move: chess.Move
) -> bool:
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


def _quiet_move_threatens_major(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    mover_color: chess.Color,
) -> bool:
    if board_before.is_capture(move) or board_after.is_check():
        return False
    enemy = not mover_color
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if p and p.color == enemy and p.piece_type in (chess.QUEEN, chess.ROOK):
            if sq in board_after.attacks(move.to_square):
                return True
    return False


def _x_ray_attack(
    board_after: chess.Board, moved_to: int, mover_color: chess.Color
) -> bool:
    """Piece on moved_to attacks an enemy piece through a friendly blocker."""
    enemy = not mover_color
    for df, dr in (
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ):
        f, r = chess.square_file(moved_to), chess.square_rank(moved_to)
        seen_own = False
        for _ in range(8):
            f += df
            r += dr
            if not (0 <= f <= 7 and 0 <= r <= 7):
                break
            sq = chess.square(f, r)
            p = board_after.piece_at(sq)
            if p is None:
                continue
            if p.color == mover_color:
                if seen_own:
                    break
                seen_own = True
                continue
            if (
                p.color == enemy
                and seen_own
                and p.piece_type != chess.KING
                and PIECE_VALUES.get(p.piece_type, 0) >= MINOR_PIECE_VALUE
            ):
                return True
            break
    return False


def _double_attack_non_fork(
    board_after: chess.Board, moved_to: int, mover_color: chess.Color
) -> bool:
    """Two+ enemy pieces attacked but not meeting fork heuristic (e.g. two minors)."""
    enemy = not mover_color
    n = 0
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if not p or p.color != enemy or p.piece_type == chess.KING:
            continue
        if sq not in board_after.attacks(moved_to):
            continue
        if PIECE_VALUES.get(p.piece_type, 0) >= MINOR_PIECE_VALUE:
            n += 1
    return n >= 2 and not _fork_after_move(board_after, moved_to, mover_color)


def _clearance_move(
    board_before: chess.Board, move: chess.Move, mover_color: chess.Color
) -> bool:
    """Rook/queen steps off a line so a friendly rook/queen behind gains a new attacked square."""
    p = board_before.piece_at(move.from_square)
    if not p or p.piece_type not in (chess.ROOK, chess.QUEEN):
        return False
    fr, ff = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    tr, tf = chess.square_rank(move.to_square), chess.square_file(move.to_square)
    if fr != tr and ff != tf:
        return False
    df = tf - ff
    dr = tr - fr
    if df != 0 and dr != 0:
        return False
    bf = -_sign(df)
    br = -_sign(dr)
    if bf == 0 and br == 0:
        return False
    f, r = ff, fr
    behind_sq: int | None = None
    while True:
        f += bf
        r += br
        if not (0 <= f <= 7 and 0 <= r <= 7):
            break
        sq = chess.square(f, r)
        pc = board_before.piece_at(sq)
        if pc is None:
            continue
        if pc.color == mover_color and pc.piece_type in (chess.ROOK, chess.QUEEN):
            behind_sq = sq
            break
        break
    if behind_sq is None:
        return False
    between_bb = chess.between(behind_sq, move.from_square)
    for sq in chess.SquareSet(between_bb):
        if sq in (behind_sq, move.from_square):
            continue
        if board_before.piece_at(sq) is not None:
            return False
    uf = _sign(ff - chess.square_file(behind_sq))
    ur = _sign(fr - chess.square_rank(behind_sq))
    past_sq = chess.square(ff + uf, fr + ur)
    if not (
        0 <= chess.square_file(past_sq) <= 7 and 0 <= chess.square_rank(past_sq) <= 7
    ):
        return False
    try:
        seen_before = board_before.attacks(behind_sq)
    except Exception:
        return False
    board_after = board_before.copy()
    board_after.push(move)
    try:
        seen_after = board_after.attacks(behind_sq)
    except Exception:
        return False
    return past_sq in seen_after and past_sq not in seen_before


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
