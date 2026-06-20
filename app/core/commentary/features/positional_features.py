from __future__ import annotations

from typing import Any

import chess

FILES = [[chess.square(file_idx, rank) for rank in range(8)] for file_idx in range(8)]

CENTER_SQUARES = [
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
]

FILE_NAMES = "abcdefgh"
# Endgame heuristic: fewer than this many minor+major pieces on the board.
ENDGAME_MINOR_MAJOR_MAX = 7
CONTESTED_SQUARES_CAP = 10

CASTLING_KEYS = frozenset({"canCastleKingSide", "canCastleQueenSide"})
PROTECTED_TOP_LEVEL_KEYS = frozenset({"material", "pawnStructure", "openFiles"})
KEEP_ZERO_METRICS = frozenset(
    {
        "mobility",
        "centerControl",
        "kingShieldPawns",
        "openFilesAdjacent",
        "semiOpenFilesAdjacent",
        "kingZoneAttacks",
        "attackedPieces",
        "attackingPieces",
        "rooksOnOpenFiles",
        "rooksOnSemiOpenFiles",
        "space",
        "lightSquareBishops",
        "darkSquareBishops",
        "goodBishops",
        "badBishops",
        "kingExposure",
        "centralization",
    }
)


def _sq_name(sq: int) -> str:
    return chess.square_name(sq)


def _file_index(square: int) -> int:
    return chess.square_file(square)


def _is_light_square(square: int) -> bool:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return (file_idx + rank_idx) % 2 == 0


def _pawn_files(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    files: dict[int, list[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        fi = _file_index(sq)
        files.setdefault(fi, []).append(sq)
    for fi in files:
        files[fi].sort(key=lambda s: chess.square_rank(s))
    return files


def _is_passed_pawn(board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
    pawn_file = chess.square_file(pawn_sq)
    enemy = not color
    ranks = (
        range(chess.square_rank(pawn_sq) + 1, 8)
        if color == chess.WHITE
        else range(chess.square_rank(pawn_sq) - 1, -1, -1)
    )
    for rank in ranks:
        for f in (pawn_file - 1, pawn_file, pawn_file + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, rank)
                if sq in board.pieces(chess.PAWN, enemy):
                    return False
    return True


def _open_and_semi_open_files(board: chess.Board) -> dict[str, list[int]]:
    open_files: list[int] = []
    semi_open_white: list[int] = []
    semi_open_black: list[int] = []
    for f in range(8):
        has_white_pawn = any(
            sq in board.pieces(chess.PAWN, chess.WHITE) for sq in FILES[f]
        )
        has_black_pawn = any(
            sq in board.pieces(chess.PAWN, chess.BLACK) for sq in FILES[f]
        )
        if not has_white_pawn and not has_black_pawn:
            open_files.append(f)
        else:
            if not has_white_pawn and has_black_pawn:
                semi_open_white.append(f)
            if not has_black_pawn and has_white_pawn:
                semi_open_black.append(f)
    return {
        "open": open_files,
        "semiOpenWhite": semi_open_white,
        "semiOpenBlack": semi_open_black,
    }


def _side_squares(board: chess.Board, color: chess.Color) -> set[int]:
    squares: set[int] = set()
    for pt in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ):
        squares.update(board.pieces(pt, color))
    return squares


def _minor_major_count(board: chess.Board) -> int:
    s = 0
    for c in (chess.WHITE, chess.BLACK):
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            s += len(board.pieces(pt, c))
    return s


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


def _material_snapshot(board: chess.Board) -> dict[str, dict]:
    values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
    counts_white = {
        "p": len(board.pieces(chess.PAWN, chess.WHITE)),
        "n": len(board.pieces(chess.KNIGHT, chess.WHITE)),
        "b": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "r": len(board.pieces(chess.ROOK, chess.WHITE)),
        "q": len(board.pieces(chess.QUEEN, chess.WHITE)),
    }
    counts_black = {
        "p": len(board.pieces(chess.PAWN, chess.BLACK)),
        "n": len(board.pieces(chess.KNIGHT, chess.BLACK)),
        "b": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "r": len(board.pieces(chess.ROOK, chess.BLACK)),
        "q": len(board.pieces(chess.QUEEN, chess.BLACK)),
    }
    total_white = sum(counts_white[k] * values[k] for k in values)
    total_black = sum(counts_black[k] * values[k] for k in values)
    diff = {k: counts_white[k] - counts_black[k] for k in values}
    diff["total"] = total_white - total_black

    imbalance: list[str] = []
    if diff["r"] != 0 or diff["b"] != 0 or diff["n"] != 0:
        if diff["r"] == -1 and (diff["b"] + diff["n"]) >= 2:
            imbalance.append("twoMinorsForRook")
        if diff["r"] == 1 and (diff["b"] + diff["n"]) <= -2:
            imbalance.append("rookForTwoMinors")
    if diff["q"] != 0:
        imbalance.append("queenTradeImbalance")

    return {
        "white": {"counts": counts_white, "total": total_white},
        "black": {"counts": counts_black, "total": total_black},
        "diff": diff,
        "imbalance": imbalance,
    }


def _is_weak_square(board: chess.Board, sq: int, for_color: chess.Color) -> bool:
    defender = for_color
    sq_file = chess.square_file(sq)
    sq_rank = chess.square_rank(sq)
    for f in (sq_file - 1, sq_file + 1):
        if not 0 <= f <= 7:
            continue
        if defender == chess.WHITE:
            for r in range(1, sq_rank):
                pawn_sq = chess.square(f, r)
                if pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
                    return False
        else:
            for r in range(sq_rank + 1, 7):
                pawn_sq = chess.square(f, r)
                if pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
                    return False
    return True


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


def _pawn_structure_classification(board: chess.Board) -> dict[str, object]:
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    center_files = {2, 3, 4, 5}
    white_center = [
        sq
        for sq in white_pawns
        if chess.square_file(sq) in center_files and 2 <= chess.square_rank(sq) <= 5
    ]
    black_center = [
        sq
        for sq in black_pawns
        if chess.square_file(sq) in center_files and 2 <= chess.square_rank(sq) <= 5
    ]

    tension: list[str] = []
    for wp in white_pawns:
        wf, wr = chess.square_file(wp), chess.square_rank(wp)
        for df in (-1, 1):
            target_f = wf + df
            target_r = wr + 1
            if 0 <= target_f <= 7 and 0 <= target_r <= 7:
                target_sq = chess.square(target_f, target_r)
                if target_sq in black_pawns:
                    tension.append(f"{_sq_name(wp)}-{_sq_name(target_sq)}")

    locked_count = 0
    for wp in white_center:
        wf, wr = chess.square_file(wp), chess.square_rank(wp)
        front_sq = chess.square(wf, wr + 1)
        if front_sq in black_pawns:
            locked_count += 1

    if locked_count >= 2:
        center_type = "closed"
    elif not white_center and not black_center:
        center_type = "open"
    else:
        center_type = "semi-open"

    breaks: list[str] = []
    for color, own_pawns_set, enemy_pawns_set, forward in [
        (chess.WHITE, white_pawns, black_pawns, 1),
        (chess.BLACK, black_pawns, white_pawns, -1),
    ]:
        side = "White" if color == chess.WHITE else "Black"
        for pawn_sq in own_pawns_set:
            pf, pr = chess.square_file(pawn_sq), chess.square_rank(pawn_sq)
            advance_sq = (
                chess.square(pf, pr + forward) if 0 <= pr + forward <= 7 else None
            )
            if advance_sq is None:
                continue
            if board.piece_at(advance_sq) is not None:
                continue
            for df in (-1, 1):
                diag_f = pf + df
                diag_r = pr + forward + forward
                if not (0 <= diag_f <= 7 and 0 <= diag_r <= 7):
                    continue
                diag_sq = chess.square(diag_f, diag_r)
                if diag_sq in enemy_pawns_set:
                    breaks.append(f"{side} {_sq_name(pawn_sq)}-{_sq_name(advance_sq)}")
                    break
    return {
        "centerType": center_type,
        "tension": tension[:6],
        "breaks": breaks[:6],
    }


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


def _clear_between(board: chess.Board, a: int, b: int) -> bool:
    if a == b:
        return True
    fa, ra = chess.square_file(a), chess.square_rank(a)
    fb, rb = chess.square_file(b), chess.square_rank(b)
    df = (fb > fa) - (fb < fa)
    dr = (rb > ra) - (rb < ra)
    if df != 0 and dr != 0 and abs(fb - fa) != abs(rb - ra):
        return False
    f, r = fa + df, ra + dr
    while f != fb or r != rb:
        sq = chess.square(f, r)
        if board.piece_at(sq) is not None:
            return False
        f += df
        r += dr
    return True


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


def _doubled_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    pf = _pawn_files(board, color)
    out: list[str] = []
    for lst in pf.values():
        if len(lst) > 1:
            out.extend(_sq_name(s) for s in lst)
    return sorted(set(out), key=lambda n: (n[1], n[0]))


def _isolated_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    pf = _pawn_files(board, color)
    out: list[str] = []
    for f, lst in pf.items():
        has_left = (f - 1) in pf
        has_right = (f + 1) in pf
        if not has_left and not has_right:
            out.extend(_sq_name(s) for s in lst)
    return out


def _passed_pawn_records(
    board: chess.Board, color: chess.Color
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sq in board.pieces(chess.PAWN, color):
        if _is_passed_pawn(board, sq, color):
            human_rank = chess.square_rank(sq) + 1
            out.append({"sq": _sq_name(sq), "rank": human_rank})
    return sorted(out, key=lambda d: d["sq"])


def _backward_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    enemy = not color
    pf = _pawn_files(board, color)
    out: list[str] = []
    for pawn_sq in board.pieces(chess.PAWN, color):
        pr = chess.square_rank(pawn_sq)
        pf_i = chess.square_file(pawn_sq)
        forward = 1 if color == chess.WHITE else -1
        adv_r = pr + forward
        if not (0 <= adv_r <= 7):
            continue
        advance_sq = chess.square(pf_i, adv_r)
        blocked_or_attacked = board.piece_at(
            advance_sq
        ) is not None or board.is_attacked_by(enemy, advance_sq)
        if not blocked_or_attacked:
            continue
        has_advanced_neighbor = False
        for df in (-1, 1):
            nf = pf_i + df
            if not 0 <= nf <= 7:
                continue
            if nf not in pf:
                continue
            for osq in pf[nf]:
                orank = chess.square_rank(osq)
                if color == chess.WHITE and orank > pr:
                    has_advanced_neighbor = True
                    break
                if color == chess.BLACK and orank < pr:
                    has_advanced_neighbor = True
                    break
            if has_advanced_neighbor:
                break
        if not has_advanced_neighbor:
            out.append(_sq_name(pawn_sq))
    return out


def _contested_squares(board: chess.Board, cap: int = 10) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for sq in chess.SQUARES:
        wa = len(board.attackers(chess.WHITE, sq))
        ba = len(board.attackers(chess.BLACK, sq))
        if wa > 0 and ba > 0:
            scored.append((wa + ba, {"sq": _sq_name(sq), "white": wa, "black": ba}))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:cap]]


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


def _pawn_islands(board: chess.Board, color: chess.Color) -> dict[str, Any]:
    pf = sorted(_pawn_files(board, color).keys())
    if not pf:
        return {"count": 0, "runs": []}
    runs: list[str] = []
    start = prev = pf[0]
    for f in pf[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append(f"{FILE_NAMES[start]}-{FILE_NAMES[prev]}")
            start = prev = f
    runs.append(f"{FILE_NAMES[start]}-{FILE_NAMES[prev]}")
    return {"count": len(runs), "runs": runs}


def _blockades(board: chess.Board) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for color in (chess.WHITE, chess.BLACK):
        enemy = not color
        fwd = 1 if color == chess.WHITE else -1
        for psq in board.pieces(chess.PAWN, color):
            if not _is_passed_pawn(board, psq, color):
                continue
            pf, pr = chess.square_file(psq), chess.square_rank(psq)
            nr = pr + fwd
            if not (0 <= nr <= 7):
                continue
            front_sq = chess.square(pf, nr)
            pc = board.piece_at(front_sq)
            if pc and pc.color == enemy and pc.piece_type != chess.PAWN:
                side = "white" if color == chess.WHITE else "black"
                out.append(
                    {
                        "passer": _sq_name(psq),
                        "blocker": _sq_name(front_sq),
                        "side": side,
                    }
                )
    return out


def _opposite_color_bishops(board: chess.Board) -> bool:
    wb = list(board.pieces(chess.BISHOP, chess.WHITE))
    bb = list(board.pieces(chess.BISHOP, chess.BLACK))
    if len(wb) != 1 or len(bb) != 1:
        return False
    wl = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[wb[0]])
    bl = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[bb[0]])
    return wl != bl


def _wrong_rook_pawn_flags(board: chess.Board) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for color in (chess.WHITE, chess.BLACK):
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) != 1:
            continue
        bsq = bishops[0]
        b_light = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[bsq])
        for psq in board.pieces(chess.PAWN, color):
            f = chess.square_file(psq)
            if f not in (0, 7):
                continue
            if not _is_passed_pawn(board, psq, color):
                continue
            promo_sq = chess.square(f, 7 if color == chess.WHITE else 0)
            promo_light = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[promo_sq])
            if promo_light != b_light:
                side = "white" if color == chess.WHITE else "black"
                out.append({"side": side, "pawn": _sq_name(psq)})
    return out


def _mop_up_distance(
    board: chess.Board, material: dict[str, Any]
) -> dict[str, Any] | None:
    diff = material.get("diff") if isinstance(material.get("diff"), dict) else {}
    total = diff.get("total")
    if not isinstance(total, (int, float)) or abs(total) < 500:
        return None
    losing = chess.BLACK if total > 0 else chess.WHITE
    ksq = board.king(losing)
    if ksq is None:
        return None
    corners = (chess.A1, chess.H1, chess.A8, chess.H8)
    dist = min(chess.square_distance(ksq, c) for c in corners)
    side = "black" if losing == chess.BLACK else "white"
    return {"losingSide": side, "kingCornerDist": dist}


def _pawn_heavy_no_major(board: chess.Board) -> bool:
    rq = (
        len(board.pieces(chess.QUEEN, chess.WHITE))
        + len(board.pieces(chess.QUEEN, chess.BLACK))
        + len(board.pieces(chess.ROOK, chess.WHITE))
        + len(board.pieces(chess.ROOK, chess.BLACK))
    )
    return rq == 0


def _rule_of_square_flags(board: chess.Board) -> list[dict[str, Any]]:
    if not _pawn_heavy_no_major(board):
        return []
    out: list[dict[str, Any]] = []
    for color in (chess.WHITE, chess.BLACK):
        enemy_k = board.king(not color)
        promo_r = 7 if color == chess.WHITE else 0
        for psq in board.pieces(chess.PAWN, color):
            if not _is_passed_pawn(board, psq, color):
                continue
            pf, pr = chess.square_file(psq), chess.square_rank(psq)
            promo_sq = chess.square(pf, promo_r)
            steps = (7 - pr) if color == chess.WHITE else pr
            dist_k = chess.square_distance(enemy_k, promo_sq)
            tempo_bonus = 1 if board.turn == color else 0
            wins_race = steps + tempo_bonus <= dist_k
            out.append(
                {
                    "sq": _sq_name(psq),
                    "side": "white" if color == chess.WHITE else "black",
                    "promotionRaceOk": wins_race,
                }
            )
    return out


def _opening_tempo_snapshot(board: chess.Board) -> dict[str, int]:
    w_home = sum(
        1
        for pt in (chess.KNIGHT, chess.BISHOP)
        for sq in board.pieces(pt, chess.WHITE)
        if chess.square_rank(sq) == 0
    )
    b_home = sum(
        1
        for pt in (chess.KNIGHT, chess.BISHOP)
        for sq in board.pieces(pt, chess.BLACK)
        if chess.square_rank(sq) == 7
    )
    developed = (4 - min(w_home, 4)) + (4 - min(b_home, 4))
    return {"minorPiecesStillHome": w_home + b_home, "developedScore": developed}


def _is_empty_for_drop(key: str | None, val: Any) -> bool:
    if key in CASTLING_KEYS:
        return False
    if key in KEEP_ZERO_METRICS and isinstance(val, (int, float)) and val == 0:
        return False
    if val is None:
        return True
    if val is False:
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    if val == [] or val == {}:
        return True
    return val == ""


def _drop_empty(obj: Any, *, _top_level_key: str | None = None) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _top_level_key is None and k in PROTECTED_TOP_LEVEL_KEYS:
                out[k] = _drop_empty(v, _top_level_key=k)
                continue
            nv = _drop_empty(v, _top_level_key=k)
            if _is_empty_for_drop(k, nv):
                continue
            out[k] = nv
        return out
    if isinstance(obj, list):
        items = [_drop_empty(x, _top_level_key=_top_level_key) for x in obj]
        items = [x for x in items if not _is_empty_for_drop(None, x)]
        return items
    return obj


def _build_board_overlay(
    files_info: dict[str, list[int]],
    white: dict[str, Any],
    black: dict[str, Any],
    contested: list[dict[str, Any]],
    pawn_structure: dict[str, Any],
    trapped_w: list[dict[str, Any]],
    trapped_b: list[dict[str, Any]],
) -> dict[str, Any]:
    ow = white.get("outposts") if isinstance(white.get("outposts"), dict) else {}
    ob = black.get("outposts") if isinstance(black.get("outposts"), dict) else {}

    def _files_to_letters(idx_list: list[int]) -> list[str]:
        return [FILE_NAMES[i] for i in idx_list]

    overlay: dict[str, Any] = {
        "openFiles": _files_to_letters(list(files_info.get("open", []))),
        "semiOpenWhite": _files_to_letters(list(files_info.get("semiOpenWhite", []))),
        "semiOpenBlack": _files_to_letters(list(files_info.get("semiOpenBlack", []))),
        "weakSquaresWhite": white.get("weakSquares"),
        "weakSquaresBlack": black.get("weakSquares"),
        "holesWhite": white.get("holes"),
        "holesBlack": black.get("holes"),
        "contestedSquares": contested,
        "outpostsWhite": (ow.get("occupied") or []) + (ow.get("available") or []),
        "outpostsBlack": (ob.get("occupied") or []) + (ob.get("available") or []),
        "passedPawnsWhite": [
            x["sq"] for x in white.get("passedPawns", []) if isinstance(x, dict)
        ],
        "passedPawnsBlack": [
            x["sq"] for x in black.get("passedPawns", []) if isinstance(x, dict)
        ],
        "doubledPawnsWhite": white.get("doubledPawns"),
        "doubledPawnsBlack": black.get("doubledPawns"),
        "pawnBreaks": pawn_structure.get("breaks")
        if isinstance(pawn_structure, dict)
        else [],
        "trappedPieces": [
            {"sq": x["sq"], "piece": x["piece"], "color": "w"} for x in trapped_w
        ]
        + [{"sq": x["sq"], "piece": x["piece"], "color": "b"} for x in trapped_b],
    }
    return overlay


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


def _side_features(
    board: chess.Board,
    color: chess.Color,
    files_info: dict[str, Any],
    is_endgame: bool,
) -> dict[str, Any]:
    """All positional features for one side (absolute, this-side perspective)."""
    side: dict[str, Any] = {}
    side["hasBishopPair"] = len(board.pieces(chess.BISHOP, color)) >= 2
    side["hasKnightPair"] = len(board.pieces(chess.KNIGHT, color)) >= 2
    side["hasRookPair"] = len(board.pieces(chess.ROOK, color)) >= 2
    side["hasQueen"] = len(board.pieces(chess.QUEEN, color)) >= 1

    bishops = list(board.pieces(chess.BISHOP, color))
    side["lightSquareBishops"] = sum(1 for b in bishops if _is_light_square(b))
    side["darkSquareBishops"] = sum(1 for b in bishops if not _is_light_square(b))

    side["canCastleKingSide"] = board.has_kingside_castling_rights(color)
    side["canCastleQueenSide"] = board.has_queenside_castling_rights(color)

    side["doubledPawns"] = _doubled_pawn_squares(board, color)
    side["isolatedPawns"] = _isolated_pawn_squares(board, color)
    side["passedPawns"] = _passed_pawn_records(board, color)
    side["backwardPawns"] = _backward_pawn_squares(board, color)
    side["weakSquares"] = _weak_squares_full_board(board, color)
    side["holes"] = _holes(board, color)
    side["pawnIslands"] = _pawn_islands(board, color)

    side.update(_attacked_and_attacking(board, color))
    side["mobility"] = _mobility(board, color)
    side["centerControl"] = _center_control(board, color)
    side.update(_king_safety(board, color))

    rooks = list(board.pieces(chess.ROOK, color))
    semi_key = "semiOpenWhite" if color == chess.WHITE else "semiOpenBlack"
    side["rooksOnOpenFiles"], side["rooksOnSemiOpenFiles"] = _rooks_on_files(
        rooks, set(files_info.get("open", [])), set(files_info.get(semi_key, []))
    )
    side["connectedRooks"] = _has_connected_rooks(board, rooks)

    side["outposts"] = _outposts(board, color)
    side.update(_bishop_quality(board, color))
    side["space"] = _space(board, color)
    side["centralization"] = _centralization(board, color)
    side["kingExposure"] = _king_exposure(board, color)
    side["batteries"] = _batteries(board, color)

    tropism = _king_pawn_tropism(board, color)
    if tropism is not None:
        side["kingPawnTropism"] = tropism
    side["connectivity"] = _connectivity(board, color)
    side["trappedPieces"] = _trapped_pieces(board, color, skip=is_endgame)
    return side


def _endgame_features(board: chess.Board, material: dict[str, dict]) -> dict[str, Any]:
    """Endgame-only flags (blockades, opposite bishops, wrong rook pawn, mop-up, …)."""
    eg: dict[str, Any] = {}
    blockades = _blockades(board)
    if blockades:
        eg["blockades"] = blockades
    if _opposite_color_bishops(board):
        eg["oppositeColorBishops"] = True
    wrong_rook_pawn = _wrong_rook_pawn_flags(board)
    if wrong_rook_pawn:
        eg["wrongRookPawn"] = wrong_rook_pawn
    mop_up = _mop_up_distance(board, material)
    if mop_up:
        eg["mopUpDistance"] = mop_up
    if _pawn_heavy_no_major(board):
        rule_of_square = _rule_of_square_flags(board)
        if rule_of_square:
            eg["ruleOfTheSquare"] = rule_of_square
    return eg


def _finalize_features(features: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values, but keep the evaluation skeleton sub-objects intact."""
    material = features.pop("material")
    open_files = features.pop("openFiles")
    pawn_structure = features.pop("pawnStructure")
    cleaned = _drop_empty(features)
    cleaned["material"] = material
    cleaned["openFiles"] = open_files
    cleaned["pawnStructure"] = pawn_structure
    return cleaned


def compute_hidden_features(
    board: chess.Board, *, in_opening_book_phase: bool = False
) -> dict[str, Any]:
    """Dense positional features for both sides + UI overlay. Numeric features are absolute per side."""
    files_info = _open_and_semi_open_files(board)
    material = _material_snapshot(board)
    pawn_structure = _pawn_structure_classification(board)
    is_endgame = _minor_major_count(board) < ENDGAME_MINOR_MAJOR_MAX
    contested = _contested_squares(board, cap=CONTESTED_SQUARES_CAP)

    features: dict[str, Any] = {
        "openFiles": files_info,
        "material": material,
        "pawnStructure": pawn_structure,
        "white": _side_features(board, chess.WHITE, files_info, is_endgame),
        "black": _side_features(board, chess.BLACK, files_info, is_endgame),
        "contestedSquares": contested,
    }
    if in_opening_book_phase:
        features["tempo"] = _opening_tempo_snapshot(board)
    if is_endgame:
        endgame = _endgame_features(board, material)
        if endgame:
            features["endgame"] = endgame

    white_side, black_side = features["white"], features["black"]
    features["boardOverlay"] = _build_board_overlay(
        files_info,
        white_side,
        black_side,
        contested,
        pawn_structure,
        white_side.get("trappedPieces") or [],
        black_side.get("trappedPieces") or [],
    )
    return _finalize_features(features)
