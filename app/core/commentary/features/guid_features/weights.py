"""Tunable Guid feature weights (centipawns) plus the feature value model and
the list of features worth charting."""

from __future__ import annotations

import chess
from pydantic import BaseModel

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
    "castling_rights": 6,  # per retained castling right (flexibility to castle)
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


def _w(name: str) -> float:
    return float(WEIGHTS[name])


def _passed_value(rel_rank: int) -> int:
    table = WEIGHTS["passed_pawn_by_rank"]
    if 0 <= rel_rank < len(table):
        return int(table[rel_rank])
    return 0


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
    "WHITE_CASTLING_RIGHTS",
    "BLACK_CASTLING_RIGHTS",
    "WHITE_KING_TROPISM",
    "BLACK_KING_TROPISM",
    "WHITE_KING_ACTIVITY",
    "BLACK_KING_ACTIVITY",
    "WHITE_OUTSIDE_PASSER",
    "BLACK_OUTSIDE_PASSER",
]
