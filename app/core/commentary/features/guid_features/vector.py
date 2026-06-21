"""The full White-POV feature vector for a position (the Knowledge Module)."""

from __future__ import annotations

import chess

from .geometry import (
    _center_ring_bonus,
    _chebyshev,
    _is_backward,
    _is_isolated,
    _is_passed,
    _pawn_files,
    _relative_rank,
)
from .weights import (
    CENTER_SQUARES,
    WEIGHTS,
    FeatureValue,
    FeatureVector,
    _passed_value,
    _w,
)


def compute_feature_vector(board: chess.Board) -> FeatureVector:
    """Full White-POV-signed feature vector for one position."""
    out: FeatureVector = {}

    def put(name: str, value_cp: float, flag: int | None = None) -> None:
        out[name] = FeatureValue(name=name, value_cp=round(value_cp), flag=flag)

    side_data: dict[chess.Color, dict[str, float]] = {}

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        prefix = "WHITE" if color == chess.WHITE else "BLACK"
        enemy = not color
        enemy_king = board.king(enemy)
        own_king = board.king(color)
        files = _pawn_files(board, color)
        pawns = list(board.pieces(chess.PAWN, color))
        knights = list(board.pieces(chess.KNIGHT, color))
        bishops = list(board.pieces(chess.BISHOP, color))
        rooks = list(board.pieces(chess.ROOK, color))
        queens = list(board.pieces(chess.QUEEN, color))

        # --- material ---
        material = (
            len(pawns) * _w("material_pawn")
            + len(knights) * _w("material_knight")
            + len(bishops) * _w("material_bishop")
            + len(rooks) * _w("material_rook")
            + len(queens) * _w("material_queen")
        )
        put(
            f"{prefix}_MATERIAL",
            sign * material,
            flag=len(pawns) + len(knights) + len(bishops) + len(rooks) + len(queens),
        )

        # --- pawn structure ---
        doubled = sum(len(sqs) - 1 for sqs in files.values() if len(sqs) > 1)
        isolated = sum(1 for sq in pawns if _is_isolated(files, sq))
        backward = sum(
            1
            for sq in pawns
            if not _is_isolated(files, sq) and _is_backward(board, sq, color)
        )
        passed = [sq for sq in pawns if _is_passed(board, sq, color)]
        duos = 0
        pawn_set = set(pawns)
        for sq in pawns:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            if f < 7 and chess.square(f + 1, r) in pawn_set:
                duos += 1
        advances = sum(max(0, _relative_rank(sq, color) - 1) for sq in pawns)

        put(f"{prefix}_PAWN_DOUBLED", sign * doubled * _w("pawn_doubled"), flag=doubled)
        put(
            f"{prefix}_PAWN_ISOLATED",
            sign * isolated * _w("pawn_isolated"),
            flag=isolated,
        )
        put(
            f"{prefix}_PAWN_BACKWARD",
            sign * backward * _w("pawn_backward"),
            flag=backward,
        )
        put(
            f"{prefix}_WEAK_PAWNS",
            sign * (isolated * _w("pawn_isolated") + backward * _w("pawn_backward")),
            flag=isolated + backward,
        )
        put(
            f"{prefix}_PAWN_PASSED",
            sign * sum(_passed_value(_relative_rank(sq, color)) for sq in passed),
            flag=len(passed),
        )
        put(f"{prefix}_PAWN_DUO", sign * duos * _w("pawn_duo"), flag=duos)
        put(
            f"{prefix}_PAWN_ADVANCES",
            sign * advances * _w("pawn_advance"),
            flag=len(pawns),
        )

        # --- knights ---
        outposts = 0
        centralization = 0
        for sq in knights:
            centralization += _center_ring_bonus(sq)
            rel = _relative_rank(sq, color)
            if 3 <= rel <= 5:
                defended = any(
                    board.piece_at(att) is not None
                    and board.piece_at(att).piece_type == chess.PAWN
                    and board.piece_at(att).color == color
                    for att in board.attackers(color, sq)
                )
                # can an enemy pawn ever kick it? (no enemy pawn on adjacent
                # file that is in front of or level with the knight)
                assailable = False
                f, r = chess.square_file(sq), chess.square_rank(sq)
                for esq in board.pieces(chess.PAWN, not color):
                    if abs(chess.square_file(esq) - f) != 1:
                        continue
                    er = chess.square_rank(esq)
                    if (color == chess.WHITE and er > r) or (
                        color == chess.BLACK and er < r
                    ):
                        assailable = True
                        break
                if defended and not assailable:
                    outposts += 1
        put(
            f"{prefix}_KNIGHTS_OUTPOSTS",
            sign * outposts * _w("knight_outpost"),
            flag=len(knights),
        )
        put(
            f"{prefix}_KNIGHTS_CENTRALIZATION",
            sign * centralization * _w("knight_centralization"),
            flag=len(knights),
        )

        # --- bishops ---
        pair = _w("bishop_pair") if len(bishops) >= 2 else 0
        put(f"{prefix}_BISHOP_PAIR", sign * pair, flag=len(bishops))

        pawns_on_color = 0
        bishop_mobility = 0
        bad_bishops = 0
        for sq in bishops:
            bishop_color_light = (
                chess.square_file(sq) + chess.square_rank(sq)
            ) % 2 == 1
            same_color_pawns = [
                p
                for p in pawns
                if ((chess.square_file(p) + chess.square_rank(p)) % 2 == 1)
                == bishop_color_light
            ]
            pawns_on_color += len(same_color_pawns)
            mob = len(board.attacks(sq))
            bishop_mobility += mob
            # Bad bishop (Ch.6 / Watson): own pawns on its color weighted by
            # centrality (d/e files x3, c/f x2, else x1), with low mobility.
            weighted = 0
            for p in same_color_pawns:
                pf = chess.square_file(p)
                if pf in (3, 4):
                    weighted += 3
                elif pf in (2, 5):
                    weighted += 2
                else:
                    weighted += 1
            # Strict on purpose: a loose threshold flags fianchetto bishops as
            # "bad" and floods comments with solved/created flicker.
            if weighted >= 7 and mob <= 3:
                bad_bishops += 1
        put(
            f"{prefix}_BISHOP_PLUS_PAWNS_ON_COLOR",
            sign * pawns_on_color * _w("bishop_pawn_on_color"),
            flag=len(bishops),
        )
        put(
            f"{prefix}_BISHOPS_MOBILITY",
            sign * bishop_mobility * _w("bishop_mobility"),
            flag=len(bishops),
        )
        put(
            f"{prefix}_BAD_BISHOP",
            sign * bad_bishops * _w("bad_bishop"),
            flag=bad_bishops,
        )

        # --- rooks ---
        open_files_cnt = 0
        half_open_cnt = 0
        seventh = 0
        behind_passer = 0
        enemy_files = _pawn_files(board, enemy)
        for sq in rooks:
            f = chess.square_file(sq)
            own_pawns_on_f = files.get(f, [])
            enemy_pawns_on_f = enemy_files.get(f, [])
            if not own_pawns_on_f and not enemy_pawns_on_f:
                open_files_cnt += 1
            elif not own_pawns_on_f:
                half_open_cnt += 1
            if _relative_rank(sq, color) == 6:
                seventh += 1
            for p in passed:
                if chess.square_file(p) != f:
                    continue
                rr, pr = chess.square_rank(sq), chess.square_rank(p)
                if (color == chess.WHITE and rr < pr) or (
                    color == chess.BLACK and rr > pr
                ):
                    behind_passer += 1
        connected = 0
        if len(rooks) >= 2:
            r1, r2 = rooks[0], rooks[1]
            if r2 in board.attacks(r1):
                connected = 1
        put(
            f"{prefix}_ROOK_OPEN_FILE",
            sign * open_files_cnt * _w("rook_open_file"),
            flag=open_files_cnt,
        )
        put(
            f"{prefix}_ROOK_HALF_OPEN_FILE",
            sign * half_open_cnt * _w("rook_half_open_file"),
            flag=half_open_cnt,
        )
        put(
            f"{prefix}_ROOK_ON_SEVENTH",
            sign * seventh * _w("rook_on_seventh"),
            flag=seventh,
        )
        put(
            f"{prefix}_ROOK_BEHIND_PASSED_PAWN",
            sign * behind_passer * _w("rook_behind_passed_pawn"),
            flag=behind_passer,
        )
        put(
            f"{prefix}_ROOKS_CONNECTED",
            sign * connected * _w("rooks_connected"),
            flag=connected,
        )

        # --- king safety ---
        shield = 0
        zone_attackers = 0
        back_rank = 0
        if own_king is not None:
            kf, kr = chess.square_file(own_king), chess.square_rank(own_king)
            step = 1 if color == chess.WHITE else -1
            for df in (-1, 0, 1):
                f = kf + df
                if f < 0 or f > 7:
                    continue
                for dr in (1, 2):
                    r = kr + step * dr
                    if r < 0 or r > 7:
                        continue
                    p = board.piece_at(chess.square(f, r))
                    if (
                        p is not None
                        and p.piece_type == chess.PAWN
                        and p.color == color
                    ):
                        shield += 1
                        break
            zone = [own_king] + [
                s for s in chess.SQUARES if _chebyshev(s, own_king) == 1
            ]
            attackers_seen = set()
            for zsq in zone:
                for att in board.attackers(enemy, zsq):
                    ap = board.piece_at(att)
                    if (
                        ap is not None
                        and ap.piece_type != chess.PAWN
                        and ap.piece_type != chess.KING
                    ):
                        attackers_seen.add(att)
            zone_attackers = len(attackers_seen)
            # back-rank weakness: king on back rank with no luft
            if _relative_rank(own_king, color) == 0:
                luft = False
                for df in (-1, 0, 1):
                    f = kf + df
                    if f < 0 or f > 7:
                        continue
                    r = kr + step
                    sq2 = chess.square(f, r)
                    p = board.piece_at(sq2)
                    if p is None and not board.is_attacked_by(enemy, sq2):
                        luft = True
                        break
                if not luft:
                    back_rank = 1
        put(
            f"{prefix}_KING_SHIELD", sign * shield * _w("king_shield_pawn"), flag=shield
        )
        put(
            f"{prefix}_KING_ZONE_ATTACKERS",
            sign * zone_attackers * _w("king_zone_attacker"),
            flag=zone_attackers,
        )
        put(
            f"{prefix}_BACK_RANK",
            sign * back_rank * _w("back_rank_weakness"),
            flag=back_rank,
        )

        # --- tropism (own pieces toward enemy king) ---
        tropism = 0.0
        if enemy_king is not None:
            for ptype, weight in WEIGHTS["king_tropism_piece"].items():
                for sq in board.pieces(ptype, color):
                    tropism += (7 - _chebyshev(sq, enemy_king)) * weight
        put(f"{prefix}_KING_TROPISM", sign * tropism, flag=None)

        # --- activity / space / center ---
        activity = 0
        for ptype in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for sq in board.pieces(ptype, color):
                activity += len(board.attacks(sq))
        center = sum(len(board.attackers(color, c)) for c in CENTER_SQUARES)
        space = 0
        enemy_half = range(4, 8) if color == chess.WHITE else range(0, 4)
        for r in enemy_half:
            for f in range(8):
                if board.is_attacked_by(color, chess.square(f, r)):
                    space += 1
        put(
            f"{prefix}_PIECE_ACTIVITY",
            sign * activity * _w("piece_activity"),
            flag=None,
        )
        put(f"{prefix}_CENTER_CONTROL", sign * center * _w("center_control"), flag=None)
        put(f"{prefix}_SPACE", sign * space * _w("space"), flag=None)

        # --- endgame pack (computed always; rules apply them in phase 'end') ---
        king_act = _center_ring_bonus(own_king) if own_king is not None else 0
        outside = 0
        if passed:
            pawn_files_all = sorted(
                {chess.square_file(p) for p in board.pieces(chess.PAWN, chess.WHITE)}
                | {chess.square_file(p) for p in board.pieces(chess.PAWN, chess.BLACK)}
            )
            if pawn_files_all:
                for p in passed:
                    pf = chess.square_file(p)
                    if pf <= min(pawn_files_all) or pf >= max(pawn_files_all):
                        # passer at the edge of the remaining pawn span
                        others = [x for x in pawn_files_all if x != pf]
                        if others and min(abs(pf - o) for o in others) >= 2:
                            outside += 1
        escort = 0
        if own_king is not None:
            for p in passed:
                escort += max(0, 3 - _chebyshev(own_king, p))
        put(f"{prefix}_KING_ACTIVITY", sign * king_act * _w("king_activity"), flag=None)
        put(
            f"{prefix}_OUTSIDE_PASSER",
            sign * outside * _w("outside_passer"),
            flag=outside,
        )
        put(
            f"{prefix}_PASSER_KING_ESCORT",
            sign * escort * _w("passer_king_escort"),
            flag=len(passed),
        )

        side_data[color] = {
            "pawn_struct": (
                out[f"{prefix}_PAWN_DOUBLED"].value_cp
                + out[f"{prefix}_PAWN_ISOLATED"].value_cp
                + out[f"{prefix}_PAWN_BACKWARD"].value_cp
                + out[f"{prefix}_PAWN_PASSED"].value_cp
                + out[f"{prefix}_PAWN_DUO"].value_cp
            ),
            "king_safety": (
                out[f"{prefix}_KING_SHIELD"].value_cp
                + out[f"{prefix}_KING_ZONE_ATTACKERS"].value_cp
                + out[f"{prefix}_BACK_RANK"].value_cp
            ),
            "tropism": out[f"{prefix}_KING_TROPISM"].value_cp,
        }

    # --- net composites (already White-POV signed, so plain sums) ---
    put(
        "EVALUATE_PAWNS",
        side_data[chess.WHITE]["pawn_struct"] + side_data[chess.BLACK]["pawn_struct"],
    )
    put(
        "EVALUATE_KING_SAFETY",
        side_data[chess.WHITE]["king_safety"] + side_data[chess.BLACK]["king_safety"],
    )
    put(
        "KING_TROPISM",
        side_data[chess.WHITE]["tropism"] + side_data[chess.BLACK]["tropism"],
    )
    put(
        "MATERIAL_BALANCE",
        out["WHITE_MATERIAL"].value_cp + out["BLACK_MATERIAL"].value_cp,
    )

    return out


def vector_to_plain(vec: FeatureVector) -> dict[str, dict[str, int | None]]:
    """JSON-friendly dump: name -> {v, flag}."""
    return {name: {"v": fv.value_cp, "flag": fv.flag} for name, fv in vec.items()}


def compute_feature_vector_fen(fen: str) -> FeatureVector:
    return compute_feature_vector(chess.Board(fen))
