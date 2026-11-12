"""Rules for catalog features that had none (added 1 Nov 2026 with the threshold analysis).

Thresholds come from the distribution of feature changes on the survey games
(see constants.THRESHOLDS); each rule reads the feature's natural count where the
centipawn sign convention would be ambiguous.
"""
from __future__ import annotations

import chess

from app.models.comment_facts import Claim
from app.core.commentary import pipeline_flags

from .constants import SIDES, THRESHOLDS
from .context import _benef, _benef_opp, _Ctx, _side_label, _toward


def _weight(value: int) -> int:
    """Claim weight: a fact that passed its rule threshold also passes the minimum weight."""
    return max(abs(value), THRESHOLDS["min_claim_cp"])


def _castled(ctx: _Ctx, color: chess.Color) -> bool:
    start, node = ctx.start_board.king(color), ctx.leaf_board.king(color)
    return (start is not None and node is not None and chess.square_file(start) == 4
            and chess.square_file(node) in (2, 6) and chess.square_rank(node) == chess.square_rank(start))


def rule_castling_rights(ctx: _Ctx) -> list[Claim]:
    """A side gives up its last castling right without castling."""
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out = []
    for side in SIDES:
        pair = ctx.flag_pair(f"{side}_CASTLING_RIGHTS")
        color = ctx.color(side)
        if pair and pair[0] > 0 and pair[1] == 0 and not _castled(ctx, color):
            out.append(Claim(rule_id="castling_rights_lost", beneficiary=_benef_opp(side),
                             text=f"{_side_label(side)} can no longer castle.",
                             text_state=f"{_side_label(side)} has lost the right to castle.",
                             features_involved=[f"{side}_CASTLING_RIGHTS"],
                             delta_cp=_weight(ctx.delta(f"{side}_CASTLING_RIGHTS"))))
    return out


def rule_king_zone_attacks(ctx: _Ctx) -> list[Claim]:
    """More of a side's pieces bear on the squares around the enemy king."""
    out = []
    for side in SIDES:
        pair = ctx.flag_pair(f"{side}_KING_ZONE_ATTACKS")
        if pair and pair[1] - pair[0] >= THRESHOLDS["king_zone_attacks"]:
            enemy = "BLACK" if side == "WHITE" else "WHITE"
            out.append(Claim(rule_id="king_zone_pressure", beneficiary=_benef(side),
                             text=f"More of {_side_label(side)}'s pieces bear on the squares around "
                                  f"{_side_label(enemy)}'s king.",
                             text_state=f"{_side_label(side)}'s pieces aim at the squares around "
                                        f"{_side_label(enemy)}'s king.",
                             features_involved=[f"{side}_KING_ZONE_ATTACKS"],
                             delta_cp=_weight(ctx.delta(f"{side}_KING_ZONE_ATTACKS")) * 10))
    return out


def rule_pawn_advances(ctx: _Ctx) -> list[Claim]:
    """A side's pawns advance markedly (pawn-advance score, not the space count)."""
    out = []
    for side in SIDES:
        d = _toward(side, ctx.delta(f"{side}_PAWN_ADVANCES"))
        if d >= THRESHOLDS["pawn_advances"]:
            out.append(Claim(rule_id="pawns_advanced", beneficiary=_benef(side),
                             text=f"{_side_label(side)}'s pawns advance and claim more of the board.",
                             text_state=f"{_side_label(side)}'s pawns stand further up the board.",
                             features_involved=[f"{side}_PAWN_ADVANCES"], delta_cp=_weight(d)))
    return out


def _pawns_on_bishop_colour(board: chess.Board, color: chess.Color) -> list[int]:
    """For each bishop of ``color``, the number of its own pawns on the bishop's colour."""
    def light(square: int) -> bool:
        return (chess.square_file(square) + chess.square_rank(square)) % 2 == 1

    pawns = list(board.pieces(chess.PAWN, color))
    return sorted(sum(1 for pawn in pawns if light(pawn) == light(bishop))
                  for bishop in board.pieces(chess.BISHOP, color))


def rule_bishop_color_complex(ctx: _Ctx) -> list[Claim]:
    """A side's bishop gets more (or fewer) of its own pawns on its colour."""
    out = []
    for side in SIDES:
        pair = ctx.flag_pair(f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR")
        if not pair:
            continue
        if (pipeline_flags.enabled("statement_wording") and ctx.start_board is not None
                and ctx.leaf_board is not None
                and _pawns_on_bishop_colour(ctx.start_board, ctx.color(side))
                == _pawns_on_bishop_colour(ctx.leaf_board, ctx.color(side))):
            # The score is a product (pawns on the colour x blocked central pawns). When only
            # the second factor moved (5.Nc3 in front of the c2 pawn), the sentence about the
            # pawns on the bishop's colour would describe something that did not change.
            continue
        change = pair[1] - pair[0]  # count of own pawns on the bishop's colour, weighted
        if change >= THRESHOLDS["bishop_color_complex"]:
            out.append(Claim(rule_id="bishop_hemmed_in", beneficiary=_benef_opp(side),
                             text=f"{_side_label(side)}'s bishop is more hemmed in by its own pawns on its colour.",
                             text_state=f"{_side_label(side)}'s own pawns stand on its bishop's colour.",
                             features_involved=[f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR"], delta_cp=_weight(change)))
        elif -change >= THRESHOLDS["bishop_color_complex"]:
            out.append(Claim(rule_id="bishop_freed", beneficiary=_benef(side),
                             text=f"{_side_label(side)}'s bishop is freed from its own pawns on its colour.",
                             text_state=f"Fewer of {_side_label(side)}'s pawns block its bishop's colour.",
                             features_involved=[f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR"], delta_cp=_weight(change)))
    return out


FEATURE_RULES = [rule_castling_rights, rule_king_zone_attacks, rule_pawn_advances, rule_bishop_color_complex]
