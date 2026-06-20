"""Guid-style weighted positional feature vector (the Knowledge Module).

Implements the Crafty-inspired feature catalog from Guid's dissertation
(Ch. 5 Tables 5.1/5.2, Ch. 6) plus aggregate piece activity and an
endgame pack. Pure python-chess — no engine involved — so it can be
computed for every mainline ply, including opening-book moves, and for
every position along a principal variation.

Conventions
-----------
* Every value is an ``int`` in centipawns, **White-POV signed**: positive
  favors White, negative favors Black. A Black *bonus* (e.g. Black owning
  the bishop pair) is therefore negative; a Black *weakness* (e.g. a Black
  isolated pawn) is positive.
* ``flag`` carries the count behind the value (number of bishops, passed
  pawns, ...) so flag transitions like ``2 -> 1`` ("bishop pair
  eliminated") fall out of a simple diff.
* The diff vector ``after - before`` then reproduces the dissertation's
  Tables 5.1/5.2: positive entries are favorable changes for White,
  negative entries favorable changes for Black.

All weights live in ``WEIGHTS`` and are deliberately easy to retune
(Guid §5.4.2 calls manual threshold tuning a design requirement).
"""

from __future__ import annotations

import chess
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Tunable weights (centipawns)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "material_pawn": 100,
    "material_knight": 320,
    "material_bishop": 330,
    "material_rook": 500,
    "material_queen": 900,
    "bishop_pair": 25,
    "bishop_pawn_on_color": -4,  # per own pawn on own bishop's square color
    "bishop_mobility": 3,  # per attacked square
    "bad_bishop": -20,  # per bishop judged bad
    "knight_outpost": 20,  # per knight on a protected, unassailable square
    "knight_centralization": 4,  # per ring step toward the center, per knight
    "rook_open_file": 15,
    "rook_half_open_file": 8,
    "rook_on_seventh": 12,
    "rook_behind_passed_pawn": 12,
    "rooks_connected": 8,
    "pawn_doubled": -12,  # per extra pawn on a file
    "pawn_isolated": -10,
    "pawn_backward": -8,
    "pawn_duo": 4,  # per side-by-side pawn pair
    "pawn_advance": 2,  # per rank past the 2nd, per pawn
    "passed_pawn_by_rank": [0, 0, 10, 15, 25, 40, 60, 0],  # index = relative rank
    "king_shield_pawn": 8,  # per shield pawn in front of the king
    "king_zone_attacker": -10,  # per enemy piece eyeing the king zone
    "king_tropism_piece": {
        chess.KNIGHT: 2,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 3,
    },
    "back_rank_weakness": -15,
    "center_control": 4,  # per attack on d4/e4/d5/e5
    "space": 1,  # per safe square controlled in enemy half
    "piece_activity": 2,  # per weighted mobility unit
    "king_activity": 5,  # endgame: per ring step toward the center
    "outside_passer": 18,  # endgame: per outside passed pawn
    "passer_king_escort": 4,  # endgame: own king close to own passer
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)

_PIECE_VALUES = {
    chess.PAWN: "material_pawn",
    chess.KNIGHT: "material_knight",
    chess.BISHOP: "material_bishop",
    chess.ROOK: "material_rook",
    chess.QUEEN: "material_queen",
}


class FeatureValue(BaseModel):
    name: str
    value_cp: int
    flag: int | None = None


FeatureVector = dict[str, FeatureValue]


# ---------------------------------------------------------------------------
# Pawn-structure helpers
# ---------------------------------------------------------------------------


def _pawn_files(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    """file -> list of pawn squares for ``color``."""
    files: dict[int, list[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        files.setdefault(chess.square_file(sq), []).append(sq)
    return files


def _is_passed(board: chess.Board, sq: int, color: chess.Color) -> bool:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    enemy = not color
    for ef in (f - 1, f, f + 1):
        if ef < 0 or ef > 7:
            continue
        for esq in board.pieces(chess.PAWN, enemy):
            if chess.square_file(esq) != ef:
                continue
            er = chess.square_rank(esq)
            if color == chess.WHITE and er > r:
                return False
            if color == chess.BLACK and er < r:
                return False
    return True


def _is_isolated(files: dict[int, list[int]], sq: int) -> bool:
    f = chess.square_file(sq)
    return not files.get(f - 1) and not files.get(f + 1)


def _is_backward(board: chess.Board, sq: int, color: chess.Color) -> bool:
    """No friendly pawn on an adjacent file at or behind it, and its stop
    square is controlled by an enemy pawn."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    behind_ok = False
    for nsq in board.pieces(chess.PAWN, color):
        nf = chess.square_file(nsq)
        if abs(nf - f) != 1:
            continue
        nr = chess.square_rank(nsq)
        if (color == chess.WHITE and nr <= r) or (color == chess.BLACK and nr >= r):
            behind_ok = True
            break
    if behind_ok:
        return False
    step = 1 if color == chess.WHITE else -1
    stop_r = r + step
    if stop_r < 0 or stop_r > 7:
        return False
    enemy = not color
    # enemy pawn attack on the stop square
    for df in (-1, 1):
        ef = f + df
        if ef < 0 or ef > 7:
            continue
        er = stop_r + step
        if er < 0 or er > 7:
            continue
        ep = board.piece_at(chess.square(ef, er))
        if ep is not None and ep.piece_type == chess.PAWN and ep.color == enemy:
            return True
    return False


def _relative_rank(sq: int, color: chess.Color) -> int:
    r = chess.square_rank(sq)
    return r if color == chess.WHITE else 7 - r


def _center_ring_bonus(sq: int) -> int:
    """0 on the rim, 3 in the four center squares (ring distance toward center)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    df = min(f, 7 - f)
    dr = min(r, 7 - r)
    return min(df, dr)


def _chebyshev(a: int, b: int) -> int:
    return chess.square_distance(a, b)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _w(name: str) -> float:
    return float(WEIGHTS[name])


def _passed_value(rel_rank: int) -> int:
    table = WEIGHTS["passed_pawn_by_rank"]
    if 0 <= rel_rank < len(table):
        return int(table[rel_rank])
    return 0


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


# ---------------------------------------------------------------------------
# Square-level lookups for grounded claim texts ("passed pawn on e5",
# "doubled c-pawns", ...). Same definitions as the vector above.
# ---------------------------------------------------------------------------


def passed_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    return [
        chess.square_name(sq)
        for sq in board.pieces(chess.PAWN, color)
        if _is_passed(board, sq, color)
    ]


def doubled_pawn_files(board: chess.Board, color: chess.Color) -> list[str]:
    files = _pawn_files(board, color)
    return [chess.FILE_NAMES[f] for f, sqs in sorted(files.items()) if len(sqs) > 1]


def outpost_squares(board: chess.Board, color: chess.Color) -> list[str]:
    out: list[str] = []
    for sq in board.pieces(chess.KNIGHT, color):
        rel = _relative_rank(sq, color)
        if not (3 <= rel <= 5):
            continue
        defended = any(
            (p := board.piece_at(att)) is not None
            and p.piece_type == chess.PAWN
            and p.color == color
            for att in board.attackers(color, sq)
        )
        if not defended:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        assailable = any(
            abs(chess.square_file(esq) - f) == 1
            and (
                (color == chess.WHITE and chess.square_rank(esq) > r)
                or (color == chess.BLACK and chess.square_rank(esq) < r)
            )
            for esq in board.pieces(chess.PAWN, not color)
        )
        if not assailable:
            out.append(chess.square_name(sq))
    return out


def bad_bishop_squares(board: chess.Board, color: chess.Color) -> list[str]:
    out: list[str] = []
    pawns = list(board.pieces(chess.PAWN, color))
    for sq in board.pieces(chess.BISHOP, color):
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        weighted = 0
        for p in pawns:
            if ((chess.square_file(p) + chess.square_rank(p)) % 2 == 1) != light:
                continue
            pf = chess.square_file(p)
            weighted += 3 if pf in (3, 4) else 2 if pf in (2, 5) else 1
        if weighted >= 7 and len(board.attacks(sq)) <= 3:
            out.append(chess.square_name(sq))
    return out


def rooks_on_seventh_squares(board: chess.Board, color: chess.Color) -> list[str]:
    return [
        chess.square_name(sq)
        for sq in board.pieces(chess.ROOK, color)
        if _relative_rank(sq, color) == 6
    ]


# Feature names worth charting (order = display order in the UI grid).
CHART_FEATURES: list[str] = [
    "MATERIAL_BALANCE",
    "EVALUATE_PAWNS",
    "EVALUATE_KING_SAFETY",
    "KING_TROPISM",
    "WHITE_PIECE_ACTIVITY",
    "BLACK_PIECE_ACTIVITY",
    "WHITE_CENTER_CONTROL",
    "BLACK_CENTER_CONTROL",
    "WHITE_SPACE",
    "BLACK_SPACE",
    "WHITE_PAWN_PASSED",
    "BLACK_PAWN_PASSED",
    "WHITE_WEAK_PAWNS",
    "BLACK_WEAK_PAWNS",
    "WHITE_PAWN_DOUBLED",
    "BLACK_PAWN_DOUBLED",
    "WHITE_BISHOP_PAIR",
    "BLACK_BISHOP_PAIR",
    "WHITE_BISHOPS_MOBILITY",
    "BLACK_BISHOPS_MOBILITY",
    "WHITE_BAD_BISHOP",
    "BLACK_BAD_BISHOP",
    "WHITE_KNIGHTS_OUTPOSTS",
    "BLACK_KNIGHTS_OUTPOSTS",
    "WHITE_KNIGHTS_CENTRALIZATION",
    "BLACK_KNIGHTS_CENTRALIZATION",
    "WHITE_ROOK_OPEN_FILE",
    "BLACK_ROOK_OPEN_FILE",
    "WHITE_ROOK_ON_SEVENTH",
    "BLACK_ROOK_ON_SEVENTH",
    "WHITE_KING_SHIELD",
    "BLACK_KING_SHIELD",
    "WHITE_KING_TROPISM",
    "BLACK_KING_TROPISM",
    "WHITE_KING_ACTIVITY",
    "BLACK_KING_ACTIVITY",
    "WHITE_OUTSIDE_PASSER",
    "BLACK_OUTSIDE_PASSER",
]
