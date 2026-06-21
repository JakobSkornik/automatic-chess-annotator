"""Piece-activity and king-safety feature helpers."""

from __future__ import annotations

from typing import Any

import chess

from .util import (
    CENTER_SQUARES,
    FILES,
    _clear_between,
    _is_light_square,
    _is_weak_square,
    _side_squares,
    _sq_name,
)


def _attacked_and_attacking(board: chess.Board, color: chess.Color) -> dict[str, int]:
    enemy = not color
    own_sqs = _side_squares(board, color)
    enemy_sqs = _side_squares(board, enemy)
    attacked_pieces = sum(1 for sq in own_sqs if board.is_attacked_by(enemy, sq))
    attackers: set[int] = set()
    for sq in enemy_sqs:
        for atk_sq in board.attackers(color, sq):
            attackers.add(atk_sq)
    return {
        "attackedPieces": attacked_pieces,
        "attackingPieces": len(attackers),
    }


def _mobility(board: chess.Board, color: chess.Color) -> int:
    original_turn = board.turn
    board.turn = color
    try:
        return sum(1 for _ in board.legal_moves)
    finally:
        board.turn = original_turn


def _center_control(board: chess.Board, color: chess.Color) -> int:
    return sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(color, sq))


def _king_safety(board: chess.Board, color: chess.Color) -> dict[str, int]:
    king_sq = board.king(color)
    if king_sq is None:
        return {
            "kingShieldPawns": 0,
            "openFilesAdjacent": 0,
            "semiOpenFilesAdjacent": 0,
            "kingZoneAttacks": 0,
        }

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    forward = 1 if color == chess.WHITE else -1

    shield_squares: list[int] = []
    for df in (-1, 0, 1):
        f = king_file + df
        if 0 <= f <= 7:
            for step in (1, 2):
                r = king_rank + forward * step
                if 0 <= r <= 7:
                    shield_squares.append(chess.square(f, r))
    king_shield_pawns = sum(
        1
        for sq in shield_squares
        if board.piece_at(sq) == chess.Piece(chess.PAWN, color)
    )

    adjacent_files = {
        f for f in (king_file - 1, king_file, king_file + 1) if 0 <= f <= 7
    }
    open_adjacent = 0
    semi_open_adjacent = 0
    for f in adjacent_files:
        has_own_pawn = any(sq in board.pieces(chess.PAWN, color) for sq in FILES[f])
        has_enemy_pawn = any(
            sq in board.pieces(chess.PAWN, not color) for sq in FILES[f]
        )
        if not has_own_pawn and not has_enemy_pawn:
            open_adjacent += 1
        elif not has_own_pawn and has_enemy_pawn:
            semi_open_adjacent += 1

    zone_attacks = 0
    for sq in chess.SQUARES:
        if chess.square_distance(sq, king_sq) == 1:
            if board.is_attacked_by(not color, sq):
                zone_attacks += 1

    return {
        "kingShieldPawns": king_shield_pawns,
        "openFilesAdjacent": open_adjacent,
        "semiOpenFilesAdjacent": semi_open_adjacent,
        "kingZoneAttacks": zone_attacks,
    }


def _weak_squares_full_board(board: chess.Board, color: chess.Color) -> list[str]:
    out: list[str] = []
    for sq in chess.SQUARES:
        if _is_weak_square(board, sq, color):
            out.append(_sq_name(sq))
    return out


def _holes(board: chess.Board, color: chess.Color) -> list[str]:
    """Weak squares in own half (can't be defended by own pawns)."""
    out: list[str] = []
    for sq in chess.SQUARES:
        r = chess.square_rank(sq)
        if color == chess.WHITE and r > 3:
            continue
        if color == chess.BLACK and r < 4:
            continue
        if _is_weak_square(board, sq, color):
            out.append(_sq_name(sq))
    return out


def _outposts(board: chess.Board, color: chess.Color) -> dict[str, list[str]]:
    enemy = not color
    own_pawns = board.pieces(chess.PAWN, color)

    if color == chess.WHITE:
        rank_range = range(3, 6)
    else:
        rank_range = range(2, 5)

    occupied: list[str] = []
    available: list[str] = []

    for rank in rank_range:
        for file_idx in range(8):
            sq = chess.square(file_idx, rank)
            if not _is_weak_square(board, sq, enemy):
                continue
            pawn_defends = False
            if color == chess.WHITE:
                for df in (-1, 1):
                    f = file_idx + df
                    r = rank - 1
                    if 0 <= f <= 7 and 0 <= r <= 7:
                        if chess.square(f, r) in own_pawns:
                            pawn_defends = True
                            break
            else:
                for df in (-1, 1):
                    f = file_idx + df
                    r = rank + 1
                    if 0 <= f <= 7 and 0 <= r <= 7:
                        if chess.square(f, r) in own_pawns:
                            pawn_defends = True
                            break
            if not pawn_defends:
                continue

            name = _sq_name(sq)
            piece = board.piece_at(sq)
            if (
                piece
                and piece.color == color
                and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
            ):
                occupied.append(name)
            else:
                available.append(name)

    return {"occupied": occupied, "available": available}


def _bishop_quality(board: chess.Board, color: chess.Color) -> dict[str, int]:
    bishops = list(board.pieces(chess.BISHOP, color))
    own_pawns = list(board.pieces(chess.PAWN, color))
    good = 0
    bad = 0
    for b_sq in bishops:
        is_light = _is_light_square(b_sq)
        same_color_pawns = sum(1 for p in own_pawns if _is_light_square(p) == is_light)
        if same_color_pawns >= 3:
            bad += 1
        elif same_color_pawns <= 1:
            good += 1
    return {"goodBishops": good, "badBishops": bad}


def _space(board: chess.Board, color: chess.Color) -> int:
    if color == chess.WHITE:
        territory = [chess.square(f, r) for r in range(4) for f in range(8)]
    else:
        territory = [chess.square(f, r) for r in range(4, 8) for f in range(8)]
    return sum(1 for sq in territory if board.is_attacked_by(color, sq))


def _centralization(board: chess.Board, color: chess.Color) -> float:
    pieces = []
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        pieces.extend(board.pieces(pt, color))
    if not pieces:
        return 0.0

    total = 0.0
    for sq in pieces:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        min_dist = min(
            abs(f - cf) + abs(r - cr) for cf, cr in [(3, 3), (3, 4), (4, 3), (4, 4)]
        )
        total += max(0, 4 - min_dist)
    return round(total / len(pieces), 2)


def _king_exposure(board: chess.Board, color: chess.Color) -> float:
    ks = _king_safety(board, color)
    score = 0.0
    score += max(0, 3 - ks["kingShieldPawns"]) * 1.5
    score += ks["openFilesAdjacent"] * 1.5
    score += ks["semiOpenFilesAdjacent"] * 0.5
    score += min(ks["kingZoneAttacks"], 6) * 0.3
    king_sq = board.king(color)
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        if (
            2 <= king_file <= 5
            and not board.has_kingside_castling_rights(color)
            and not board.has_queenside_castling_rights(color)
        ):
            score += 2.0
    return round(min(10.0, score), 1)


def _batteries(board: chess.Board, color: chess.Color) -> list[str]:
    queens = list(board.pieces(chess.QUEEN, color))
    bishops = list(board.pieces(chess.BISHOP, color))
    rooks = list(board.pieces(chess.ROOK, color))

    result: list[str] = []
    for q in queens:
        qf, qr = chess.square_file(q), chess.square_rank(q)
        for b in bishops:
            bf, br = chess.square_file(b), chess.square_rank(b)
            if abs(qf - bf) == abs(qr - br) and _clear_between(board, q, b):
                result.append(f"Q{_sq_name(q)}+B{_sq_name(b)}")
        for r_sq in rooks:
            rf, rr = chess.square_file(r_sq), chess.square_rank(r_sq)
            if (qf == rf or qr == rr) and _clear_between(board, q, r_sq):
                result.append(f"Q{_sq_name(q)}+R{_sq_name(r_sq)}")
    return result[:4]


def _trapped_pieces(
    board: chess.Board, color: chess.Color, skip: bool
) -> list[dict[str, Any]]:
    if skip:
        return []
    enemy = not color
    out: list[dict[str, Any]] = []
    original_turn = board.turn
    for sq in _side_squares(board, color):
        piece = board.piece_at(sq)
        if piece is None or piece.color != color:
            continue
        if piece.piece_type in (chess.KING, chess.PAWN):
            continue
        board.turn = color
        try:
            n_moves = sum(1 for m in board.legal_moves if m.from_square == sq)
        finally:
            board.turn = original_turn
        if n_moves <= 1 and board.is_attacked_by(enemy, sq):
            sym = piece.symbol().upper()
            out.append({"sq": _sq_name(sq), "piece": sym, "escapes": n_moves})
    return out


def _king_pawn_tropism(board: chess.Board, color: chess.Color) -> float | None:
    ksq = board.king(color)
    if ksq is None:
        return None
    enemy = not color
    epawns = board.pieces(chess.PAWN, enemy)
    if not epawns:
        return None
    tot = sum(chess.square_distance(ksq, psq) for psq in epawns)
    return round(tot / len(epawns), 2)


def _connectivity(board: chess.Board, color: chess.Color) -> dict[str, int]:
    own = _side_squares(board, color)
    adj: dict[int, set[int]] = {sq: set() for sq in own}
    defenders: set[int] = set()
    for defender_sq in own:
        piece = board.piece_at(defender_sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        for target_sq in own:
            if target_sq == defender_sq:
                continue
            if defender_sq in board.attackers(color, target_sq):
                defenders.add(defender_sq)
                adj[defender_sq].add(target_sq)
                adj[target_sq].add(defender_sq)
    visited: set[int] = set()
    chains = 0
    for sq in own:
        if sq in visited:
            continue
        stack = [sq]
        comp: list[int] = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    stack.append(v)
        if len(comp) >= 2:
            chains += 1
    return {"defenders": len(defenders), "chains": chains}


def _rooks_on_files(
    rooks: list[int], open_files: set[int], semi_open_files: set[int]
) -> tuple[int, int]:
    """(rooks on open files, rooks on this side's semi-open files)."""
    on_open = sum(1 for r in rooks if chess.square_file(r) in open_files)
    on_semi = sum(1 for r in rooks if chess.square_file(r) in semi_open_files)
    return on_open, on_semi


def _has_connected_rooks(board: chess.Board, rooks: list[int]) -> bool:
    """Two rooks defend each other along a clear shared file or rank."""
    for i in range(len(rooks)):
        for j in range(i + 1, len(rooks)):
            a, b = rooks[i], rooks[j]
            same_line = chess.square_file(a) == chess.square_file(
                b
            ) or chess.square_rank(a) == chess.square_rank(b)
            if same_line and _clear_between(board, a, b):
                return True
    return False
