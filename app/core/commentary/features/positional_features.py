from __future__ import annotations

from typing import Dict, List, Set, Tuple

import chess


FILES = [
    [chess.square(file_idx, rank) for rank in range(8)] for file_idx in range(8)
]

CENTER_SQUARES = [
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
]

EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
]

FILE_NAMES = "abcdefgh"


def _sq_name(sq: int) -> str:
    """Return algebraic name of a square (e.g. 'e4')."""
    return chess.square_name(sq)


def _file_index(square: int) -> int:
    return chess.square_file(square)


def _is_light_square(square: int) -> bool:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return (file_idx + rank_idx) % 2 == 0


def _pawn_files(board: chess.Board, color: chess.Color) -> Dict[int, List[int]]:
    files: Dict[int, List[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        fi = _file_index(sq)
        files.setdefault(fi, []).append(sq)
    return files


def _is_passed_pawn(board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
    pawn_file = chess.square_file(pawn_sq)
    enemy = not color
    ranks = range(chess.square_rank(pawn_sq) + 1, 8) if color == chess.WHITE else range(chess.square_rank(pawn_sq) - 1, -1, -1)
    for rank in ranks:
        for f in (pawn_file - 1, pawn_file, pawn_file + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, rank)
                if sq in board.pieces(chess.PAWN, enemy):
                    return False
    return True


def _open_and_semi_open_files(board: chess.Board) -> Dict[str, List[int]]:
    open_files: List[int] = []
    semi_open_white: List[int] = []
    semi_open_black: List[int] = []
    for f in range(8):
        has_white_pawn = any(sq in board.pieces(chess.PAWN, chess.WHITE) for sq in FILES[f])
        has_black_pawn = any(sq in board.pieces(chess.PAWN, chess.BLACK) for sq in FILES[f])
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


def _side_squares(board: chess.Board, color: chess.Color) -> Set[int]:
    squares: Set[int] = set()
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        squares.update(board.pieces(pt, color))
    return squares


def _attacked_and_attacking(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    enemy = not color
    own_sqs = _side_squares(board, color)
    enemy_sqs = _side_squares(board, enemy)
    attacked_pieces = sum(1 for sq in own_sqs if board.is_attacked_by(enemy, sq))
    attackers: Set[int] = set()
    for sq in enemy_sqs:
        for atk_sq in board.attackers(color, sq):
            attackers.add(atk_sq)
    return {
        "attackedPieces": attacked_pieces,
        "attackingPieces": len(attackers),
    }


def _mobility(board: chess.Board, color: chess.Color) -> int:
    """Count legal moves for the given side as a proxy for activity."""
    original_turn = board.turn
    board.turn = color
    try:
        return sum(1 for _ in board.legal_moves)
    finally:
        board.turn = original_turn


def _center_control(board: chess.Board, color: chess.Color) -> int:
    """Count how many center squares are attacked by the side."""
    return sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(color, sq))


def _king_safety(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    """Compute simple king safety stats: pawn shield and open files near king."""
    king_sq = board.king(color)
    if king_sq is None:
        return {"kingShieldPawns": 0, "openFilesAdjacent": 0, "semiOpenFilesAdjacent": 0, "kingZoneAttacks": 0}

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    forward = 1 if color == chess.WHITE else -1

    shield_squares: List[int] = []
    for df in (-1, 0, 1):
        f = king_file + df
        if 0 <= f <= 7:
            for step in (1, 2):
                r = king_rank + forward * step
                if 0 <= r <= 7:
                    shield_squares.append(chess.square(f, r))
    king_shield_pawns = sum(
        1 for sq in shield_squares if board.piece_at(sq) == chess.Piece(chess.PAWN, color)
    )

    adjacent_files = {f for f in (king_file - 1, king_file, king_file + 1) if 0 <= f <= 7}
    open_adjacent = 0
    semi_open_adjacent = 0
    for f in adjacent_files:
        has_own_pawn = any(sq in board.pieces(chess.PAWN, color) for sq in FILES[f])
        has_enemy_pawn = any(sq in board.pieces(chess.PAWN, not color) for sq in FILES[f])
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


def _material_snapshot(board: chess.Board) -> Dict[str, Dict]:
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

    imbalance: List[str] = []
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


# ---------------------------------------------------------------------------
# NEW: Weak squares
# ---------------------------------------------------------------------------

def _is_weak_square(board: chess.Board, sq: int, for_color: chess.Color) -> bool:
    """A square is *weak for* ``for_color`` when no pawn of ``for_color`` can
    ever defend it (no friendly pawns on adjacent files that could advance to
    cover it)."""
    defender = for_color
    sq_file = chess.square_file(sq)
    sq_rank = chess.square_rank(sq)
    # Check adjacent files for pawns that could defend
    for f in (sq_file - 1, sq_file + 1):
        if not 0 <= f <= 7:
            continue
        # Look behind / at / ahead of the square for friendly pawns that could
        # advance to defend. Pawns defend diagonally forward.
        if defender == chess.WHITE:
            # White pawns defend squares one rank ahead-diagonally.
            # A white pawn on file f, rank r defends (f-1, r+1) and (f+1, r+1).
            # So to defend sq (sq_file, sq_rank), need pawn on (adj_file, sq_rank-1).
            # But pawn could also advance from further behind.
            for r in range(1, sq_rank):  # ranks below the target
                pawn_sq = chess.square(f, r)
                if pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
                    return False
        else:
            for r in range(sq_rank + 1, 7):  # ranks above the target
                pawn_sq = chess.square(f, r)
                if pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
                    return False
    return True


def _weak_squares(board: chess.Board, color: chess.Color) -> List[str]:
    """Return algebraic names of squares in the opponent's half that are weak
    for ``color`` (cannot be defended by color's pawns) and are on ranks 4-6
    for White / 3-5 for Black."""
    result: List[str] = []
    if color == chess.WHITE:
        rank_range = range(3, 6)  # ranks 4-6 (0-indexed 3-5)
    else:
        rank_range = range(2, 5)  # ranks 3-5 (0-indexed 2-4)
    for rank in rank_range:
        for file_idx in range(8):
            sq = chess.square(file_idx, rank)
            if _is_weak_square(board, sq, color):
                result.append(_sq_name(sq))
    return result


# ---------------------------------------------------------------------------
# NEW: Outposts
# ---------------------------------------------------------------------------

def _outposts(board: chess.Board, color: chess.Color) -> Dict[str, List[str]]:
    """Outposts are weak squares in the opponent's territory that are:
    (a) weak for the opponent (cannot be defended by enemy pawns), AND
    (b) protected by own pawn.
    Returns ``{"occupied": [...], "available": [...]}``.
    """
    enemy = not color
    own_pawns = board.pieces(chess.PAWN, color)
    knights = board.pieces(chess.KNIGHT, color)
    bishops = board.pieces(chess.BISHOP, color)

    if color == chess.WHITE:
        rank_range = range(3, 6)  # ranks 4-6
    else:
        rank_range = range(2, 5)  # ranks 3-5

    occupied: List[str] = []
    available: List[str] = []

    for rank in rank_range:
        for file_idx in range(8):
            sq = chess.square(file_idx, rank)
            # Must be weak for the enemy (they can't defend it with pawns)
            if not _is_weak_square(board, sq, enemy):
                continue
            # Must be protected by own pawn
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
            if piece and piece.color == color and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                occupied.append(name)
            else:
                available.append(name)

    return {"occupied": occupied, "available": available}


# ---------------------------------------------------------------------------
# NEW: Good / bad bishops
# ---------------------------------------------------------------------------

def _bishop_quality(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    """Evaluate bishop quality based on how many own pawns sit on the same
    square color as the bishop.  Bad bishop: >=3 own pawns on same color.
    Good bishop: <=1."""
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


# ---------------------------------------------------------------------------
# NEW: Pawn structure classification
# ---------------------------------------------------------------------------

def _pawn_structure_classification(board: chess.Board) -> Dict[str, object]:
    """Classify the center pawn structure and detect pawn tension / breaks.

    Returns:
        centerType: "open" | "closed" | "semi-open"
        tension: list of square-pair strings (e.g. ["e4-d5"])
        breaks: list of potential pawn break descriptions
    """
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    # Center files: c, d, e, f (indices 2-5)
    center_files = {2, 3, 4, 5}
    white_center = [sq for sq in white_pawns if chess.square_file(sq) in center_files and 2 <= chess.square_rank(sq) <= 5]
    black_center = [sq for sq in black_pawns if chess.square_file(sq) in center_files and 2 <= chess.square_rank(sq) <= 5]

    # Detect pawn tension: opposing pawns that can capture each other
    tension: List[str] = []
    for wp in white_pawns:
        wf, wr = chess.square_file(wp), chess.square_rank(wp)
        for df in (-1, 1):
            target_f = wf + df
            target_r = wr + 1
            if 0 <= target_f <= 7 and 0 <= target_r <= 7:
                target_sq = chess.square(target_f, target_r)
                if target_sq in black_pawns:
                    tension.append(f"{_sq_name(wp)}-{_sq_name(target_sq)}")

    # Detect locked center pawns (white pawn directly in front of black pawn)
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

    # Detect available pawn breaks: own pawns that could advance to create
    # tension with enemy pawns (one square push creates a capture opportunity)
    breaks: List[str] = []
    for color, own_pawns_set, enemy_pawns_set, forward in [
        (chess.WHITE, white_pawns, black_pawns, 1),
        (chess.BLACK, black_pawns, white_pawns, -1),
    ]:
        side = "White" if color == chess.WHITE else "Black"
        for pawn_sq in own_pawns_set:
            pf, pr = chess.square_file(pawn_sq), chess.square_rank(pawn_sq)
            advance_sq = chess.square(pf, pr + forward) if 0 <= pr + forward <= 7 else None
            if advance_sq is None:
                continue
            # Check if advancing creates tension with enemy pawns on adjacent files
            if board.piece_at(advance_sq) is not None:
                continue  # blocked
            for df in (-1, 1):
                diag_f = pf + df
                diag_r = pr + forward + forward  # enemy pawn one rank further
                if not (0 <= diag_f <= 7 and 0 <= diag_r <= 7):
                    continue
                diag_sq = chess.square(diag_f, diag_r)
                if diag_sq in enemy_pawns_set:
                    breaks.append(f"{side} {_sq_name(pawn_sq)}-{_sq_name(advance_sq)}")
                    break
    return {
        "centerType": center_type,
        "tension": tension[:6],  # cap for token economy
        "breaks": breaks[:6],
    }


# ---------------------------------------------------------------------------
# NEW: Space advantage
# ---------------------------------------------------------------------------

def _space(board: chess.Board, color: chess.Color) -> int:
    """Count squares in own half that are controlled (attacked) by the side.
    White's territory = ranks 1-4, Black's territory = ranks 5-8."""
    if color == chess.WHITE:
        territory = [chess.square(f, r) for r in range(4) for f in range(8)]
    else:
        territory = [chess.square(f, r) for r in range(4, 8) for f in range(8)]
    return sum(1 for sq in territory if board.is_attacked_by(color, sq))


# ---------------------------------------------------------------------------
# NEW: Piece centralization
# ---------------------------------------------------------------------------

def _centralization(board: chess.Board, color: chess.Color) -> float:
    """Average inverse Manhattan distance to the four center squares for all
    non-king, non-pawn pieces.  Higher = more centralized (0-4 scale)."""
    pieces = []
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        pieces.extend(board.pieces(pt, color))
    if not pieces:
        return 0.0

    total = 0.0
    for sq in pieces:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        # minimum Manhattan distance to any of the 4 center squares
        min_dist = min(abs(f - cf) + abs(r - cr) for cf, cr in [(3, 3), (3, 4), (4, 3), (4, 4)])
        total += max(0, 4 - min_dist)  # invert: 4 = on center, 0 = far away
    return round(total / len(pieces), 2)


# ---------------------------------------------------------------------------
# NEW: King exposure composite score (0-10)
# ---------------------------------------------------------------------------

def _king_exposure(board: chess.Board, color: chess.Color) -> float:
    """Composite king exposure score (0 = safe, 10 = very exposed)."""
    ks = _king_safety(board, color)
    score = 0.0
    # Fewer shield pawns = more exposed (max 6 shield squares, typically 2-3 pawns)
    score += max(0, 3 - ks["kingShieldPawns"]) * 1.5   # 0-4.5
    score += ks["openFilesAdjacent"] * 1.5                # 0-4.5
    score += ks["semiOpenFilesAdjacent"] * 0.5            # 0-1.5
    score += min(ks["kingZoneAttacks"], 6) * 0.3          # 0-1.8
    # No castling rights when king is still in center is dangerous
    king_sq = board.king(color)
    if king_sq is not None:
        king_file = chess.square_file(king_sq)
        if 2 <= king_file <= 5:  # king is in center files
            if not board.has_kingside_castling_rights(color) and not board.has_queenside_castling_rights(color):
                score += 2.0
    return round(min(10.0, score), 1)


# ---------------------------------------------------------------------------
# NEW: Piece coordination / batteries
# ---------------------------------------------------------------------------

def _batteries(board: chess.Board, color: chess.Color) -> List[str]:
    """Detect battery formations (queen+bishop on same diagonal, queen+rook
    on same file/rank)."""
    queens = list(board.pieces(chess.QUEEN, color))
    bishops = list(board.pieces(chess.BISHOP, color))
    rooks = list(board.pieces(chess.ROOK, color))

    def _clear_between_local(a: int, b: int) -> bool:
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

    result: List[str] = []
    for q in queens:
        qf, qr = chess.square_file(q), chess.square_rank(q)
        # Queen + Bishop on diagonal
        for b in bishops:
            bf, br = chess.square_file(b), chess.square_rank(b)
            if abs(qf - bf) == abs(qr - br) and _clear_between_local(q, b):
                result.append(f"Q{_sq_name(q)}+B{_sq_name(b)}")
        # Queen + Rook on file/rank
        for r_sq in rooks:
            rf, rr = chess.square_file(r_sq), chess.square_rank(r_sq)
            if (qf == rf or qr == rr) and _clear_between_local(q, r_sq):
                result.append(f"Q{_sq_name(q)}+R{_sq_name(r_sq)}")
    return result[:4]  # cap for token economy


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def compute_hidden_features(board: chess.Board) -> Dict:
    """Compute a comprehensive set of positional features for both sides.

    Returns a dict with keys: openFiles, material, pawnStructure, white, black
    """
    features: Dict = {}

    # Open/semi-open files
    files_info = _open_and_semi_open_files(board)
    features["openFiles"] = files_info
    features["material"] = _material_snapshot(board)
    features["pawnStructure"] = _pawn_structure_classification(board)

    def _clear_between(a: int, b: int) -> bool:
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

    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        side: Dict = {}

        # Piece counts / pairs
        num_bishops = len(board.pieces(chess.BISHOP, color))
        num_knights = len(board.pieces(chess.KNIGHT, color))
        num_rooks = len(board.pieces(chess.ROOK, color))
        num_queens = len(board.pieces(chess.QUEEN, color))

        side["hasBishopPair"] = num_bishops >= 2
        side["hasKnightPair"] = num_knights >= 2
        side["hasRookPair"] = num_rooks >= 2
        side["hasQueen"] = num_queens >= 1

        # Bishop square colors counts
        bishops = list(board.pieces(chess.BISHOP, color))
        side["lightSquareBishops"] = sum(1 for b in bishops if _is_light_square(b))
        side["darkSquareBishops"] = sum(1 for b in bishops if not _is_light_square(b))

        # Castling rights
        side["canCastleKingSide"] = board.has_kingside_castling_rights(color)
        side["canCastleQueenSide"] = board.has_queenside_castling_rights(color)

        # Pawn structure: doubled, isolated, passed
        pawn_files = _pawn_files(board, color)
        side["doubledPawns"] = sum(1 for _, lst in pawn_files.items() if len(lst) > 1)

        isolated = 0
        passed_pawns = 0
        for f, lst in pawn_files.items():
            has_left = (f - 1) in pawn_files
            has_right = (f + 1) in pawn_files
            if not has_left and not has_right:
                isolated += len(lst)
            for p_sq in lst:
                if _is_passed_pawn(board, p_sq, color):
                    passed_pawns += 1
        side["isolatedPawns"] = isolated
        side["passedPawns"] = passed_pawns

        # Attacked/attacking counts
        side.update(_attacked_and_attacking(board, color))

        # Mobility and center control
        side["mobility"] = _mobility(board, color)
        side["centerControl"] = _center_control(board, color)

        # King safety features
        side.update(_king_safety(board, color))

        # Rook features
        rooks = list(board.pieces(chess.ROOK, color))
        if color == chess.WHITE:
            semi_open_for_side = set(files_info.get("semiOpenWhite", []))
        else:
            semi_open_for_side = set(files_info.get("semiOpenBlack", []))
        open_files_set = set(files_info.get("open", []))

        rooks_on_open = 0
        rooks_on_semi = 0
        for r_sq in rooks:
            f = chess.square_file(r_sq)
            if f in open_files_set:
                rooks_on_open += 1
            if f in semi_open_for_side:
                rooks_on_semi += 1
        side["rooksOnOpenFiles"] = rooks_on_open
        side["rooksOnSemiOpenFiles"] = rooks_on_semi

        connected = False
        if len(rooks) >= 2:
            for i in range(len(rooks)):
                for j in range(i + 1, len(rooks)):
                    a, b = rooks[i], rooks[j]
                    if chess.square_file(a) == chess.square_file(b) or chess.square_rank(a) == chess.square_rank(b):
                        if _clear_between(a, b):
                            connected = True
                            break
                if connected:
                    break
        side["connectedRooks"] = connected

        # --- NEW strategic features ---
        side["weakSquares"] = _weak_squares(board, color)
        side.update({"outposts": _outposts(board, color)})
        side.update(_bishop_quality(board, color))
        side["space"] = _space(board, color)
        side["centralization"] = _centralization(board, color)
        side["kingExposure"] = _king_exposure(board, color)
        side["batteries"] = _batteries(board, color)

        features[label] = side

    return features


