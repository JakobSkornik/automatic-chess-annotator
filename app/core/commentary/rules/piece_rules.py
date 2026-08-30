from __future__ import annotations

import logging

from app.core.commentary.features.guid_features import (
    bad_bishop_squares,
    outpost_squares,
    rooks_on_seventh_squares,
)
from app.models.comment_facts import Claim

import chess

from .constants import SIDES, THRESHOLDS
from .context import _benef, _benef_opp, _Ctx, _side_label, _toward

logger = logging.getLogger(__name__)


def rule_connected_rooks(ctx: _Ctx) -> list[Claim]:
    """Rooks become connected on the back rank / a file."""
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_ROOKS_CONNECTED"
        pair = ctx.flag_pair(name)
        if (
            pair
            and pair[1] > pair[0]
            and abs(ctx.delta(name)) >= THRESHOLDS["connected_rooks"]
        ):
            out.append(
                Claim(
                    rule_id="rooks_connected",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} connects the rooks.",
                    text_state=f"{_side_label(side)}'s rooks are connected.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_bishop_pair(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_BISHOP_PAIR"
        pair = ctx.flag_pair(name)
        if pair and pair[0] >= 2 and pair[1] < 2:
            out.append(
                Claim(
                    rule_id="bishop_pair_eliminated",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} no longer has the advantage of the bishop pair.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_strong_knight(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_KNIGHTS_OUTPOSTS"
        cent = f"{side}_KNIGHTS_CENTRALIZATION"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["strong_knight"]:
            sqs = ctx.new_squares(outpost_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(
                Claim(
                    rule_id="strong_knight_established",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} establishes a strong knight{where}.",
                    text_state=f"{_side_label(side)} has a strong knight{where}.",
                    features_involved=[name, cent],
                    delta_cp=ctx.claim_cp(name),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif -d >= THRESHOLDS["strong_knight"]:
            out.append(
                Claim(
                    rule_id="strong_knight_lost",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} no longer has a strong knight.",
                    features_involved=[name, cent],
                    delta_cp=ctx.claim_cp(name),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_bad_bishop(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_BAD_BISHOP"
        pair = ctx.flag_pair(name)
        # Require a meaningful swing — the flag flickers as pawns leave the
        # bishop's colour along best play, which over-reported "solved".
        if pair and abs(ctx.delta(name)) < THRESHOLDS["bad_bishop"]:
            continue
        if pair and pair[1] > pair[0]:
            sqs = ctx.new_squares(bad_bishop_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(
                Claim(
                    rule_id="bad_bishop_created",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} is left with a bad bishop{where}.",
                    text_state=f"{_side_label(side)}'s bishop{where} is now bad.",
                    features_involved=[
                        name,
                        f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR",
                        f"{side}_BISHOPS_MOBILITY",
                    ],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif pair and pair[1] < pair[0]:
            start_sqs = (
                bad_bishop_squares(ctx.start_board, ctx.color(side))
                if ctx.start_board is not None
                else []
            )
            where = f" on {start_sqs[0]}" if start_sqs else ""
            out.append(
                Claim(
                    rule_id="bad_bishop_solved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} solves the problem of the bad bishop{where}.",
                    text_state=f"{_side_label(side)}'s bad bishop{where} is no longer a problem.",
                    features_involved=[
                        name,
                        f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR",
                        f"{side}_BISHOPS_MOBILITY",
                    ],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_rook_activity(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        seventh = f"{side}_ROOK_ON_SEVENTH"
        pair7 = ctx.flag_pair(seventh)
        if pair7 and pair7[1] > pair7[0]:
            sqs = ctx.new_squares(rooks_on_seventh_squares, side)
            where = f" ({sqs[0]})" if sqs else ""
            out.append(
                Claim(
                    rule_id="rook_reaches_seventh",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s rook reaches the seventh rank{where}.",
                    text_state=f"{_side_label(side)} has a rook on the seventh rank{where}.",
                    features_involved=[seventh],
                    delta_cp=abs(ctx.delta(seventh)),
                    flag_note=ctx.flag_change(seventh),
                )
            )
            continue
        feats = [f"{side}_ROOK_OPEN_FILE", f"{side}_ROOK_HALF_OPEN_FILE"]
        d = sum(_toward(side, ctx.delta(f)) for f in feats)
        if d >= THRESHOLDS["rook_activity"]:
            out.append(
                Claim(
                    rule_id="rooks_activated",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s rooks become more active on the open files.",
                    text_state=f"{_side_label(side)}'s rooks are active on the open files.",
                    features_involved=feats,
                    delta_cp=ctx.claim_cp(feats[0]) + ctx.claim_cp(feats[1]),
                )
            )
    return out


def rule_king_safety(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_KING_DANGER"
        # KING_DANGER is stored negative for the attacked side, so a positive
        # mover-POV delta means that side's king got safer; negative = pressured.
        d = _toward(side, ctx.delta(name))
        opp = "BLACK" if side == "WHITE" else "WHITE"
        opp_tropism = _toward(opp, ctx.delta(f"{opp}_KING_TROPISM"))
        if (
            -d >= THRESHOLDS["king_safety"]
            and opp_tropism >= THRESHOLDS["king_tropism_corroborate"]
        ):
            out.append(
                Claim(
                    rule_id="king_under_pressure",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s king comes under pressure.",
                    text_state=f"{_side_label(side)}'s king is under pressure.",
                    features_involved=[name, f"{opp}_KING_TROPISM"],
                    delta_cp=-d,
                )
            )
        elif (
            d >= THRESHOLDS["king_safety"]
            and -opp_tropism >= THRESHOLDS["king_tropism_corroborate"]
        ):
            out.append(
                Claim(
                    rule_id="king_safer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s king is now safer.",
                    text_state=f"{_side_label(side)}'s king is safe.",
                    features_involved=[name, f"{opp}_KING_TROPISM"],
                    delta_cp=d,
                )
            )
        elif -d >= THRESHOLDS["king_safety"] or d >= THRESHOLDS["king_safety"]:
            # The king-danger swing alone cleared the threshold but the
            # corroborating tropism swing did not — the stacked AND condition
            # silently drops what would otherwise be a claim.
            logger.debug(
                "rule_king_safety: corroboration gate blocked a claim for %s "
                "(king_danger_swing=%d, threshold=%d, opp_tropism_swing=%d, "
                "corroborate_threshold=%d)",
                side,
                d if d >= THRESHOLDS["king_safety"] else -d,
                THRESHOLDS["king_safety"],
                opp_tropism,
                THRESHOLDS["king_tropism_corroborate"],
            )
    return out


def rule_back_rank(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_BACK_RANK"
        pair = ctx.flag_pair(name)
        if pair and pair[0] == 0 and pair[1] == 1:
            out.append(
                Claim(
                    rule_id="back_rank_weakness",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s back rank becomes vulnerable.",
                    text_state=f"{_side_label(side)}'s back rank is vulnerable.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_piece_activity(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    # When material changed, the mobility swing is mostly a side effect of the
    # capture, not a genuine activity gain — let rule_material speak instead.
    if abs(ctx.delta("MATERIAL_BALANCE")) >= 100:
        logger.debug(
            "rule_piece_activity: suppressed outright (material delta=%d cp >= 100) "
            "mover=%s",
            ctx.delta("MATERIAL_BALANCE"),
            ctx.mover,
        )
        return out
    for side in SIDES:
        name = f"{side}_PIECE_ACTIVITY"
        # `d` is now a mobility-square swing (PIECE_ACTIVITY is a natural count).
        # The claim's delta_cp is scaled to a cp-comparable importance so it ranks
        # and clears min_claim_cp alongside the centipawn-scaled features.
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["piece_activity"]:
            out.append(
                Claim(
                    rule_id="activity_improved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} has improved the activity of the pieces.",
                    text_state=f"{_side_label(side)}'s pieces are actively placed.",
                    features_involved=[name],
                    delta_cp=ctx.claim_cp(name),
                )
            )
        elif -d >= THRESHOLDS["piece_activity"]:
            out.append(
                Claim(
                    rule_id="activity_reduced",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s pieces become more passive.",
                    text_state=f"{_side_label(side)}'s pieces are passive.",
                    features_involved=[name],
                    delta_cp=ctx.claim_cp(name),
                )
            )
    return out


def rule_center_and_space(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        c = _toward(side, ctx.delta(f"{side}_CENTER_CONTROL"))
        s = _toward(side, ctx.delta(f"{side}_SPACE"))
        if c >= THRESHOLDS["center_control"]:
            out.append(
                Claim(
                    rule_id="center_control",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} takes control of the center.",
                    text_state=f"{_side_label(side)} controls the center.",
                    features_involved=[f"{side}_CENTER_CONTROL"],
                    delta_cp=c,
                )
            )
        # Independent of center — center no longer masks a real space gain.
        if s >= THRESHOLDS["space"]:
            out.append(
                Claim(
                    rule_id="space_gained",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} gains space.",
                    text_state=f"{_side_label(side)} has a space advantage.",
                    features_involved=[f"{side}_SPACE"],
                    delta_cp=s,
                )
            )
    return out


# --- endgame pack ---


def rule_king_activity(ctx: _Ctx) -> list[Claim]:
    if ctx.phase != "end":
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_KING_ACTIVITY"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["king_activity_endgame"]:
            out.append(
                Claim(
                    rule_id="king_activated",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s king becomes active.",
                    text_state=f"{_side_label(side)}'s king is active.",
                    features_involved=[name],
                    delta_cp=d,
                )
            )
    return out


def rule_intent(ctx: _Ctx) -> list[Claim]:
    """What a quiet move is *for*: newly attacked enemy piece, prepared pawn
    break, or file opened for the rook. Needs boards (not just the diff), so
    it reads them from the context; at most one intent claim per pass, for
    the mover only. Suppressed in decided positions — plans are noise there
    (Guid: positional claims dropped when the eval is decisive)."""
    if ctx.leaf_board is None:
        return []
    if ctx.eval_cp is not None and abs(ctx.eval_cp) > 500:
        logger.debug(
            "rule_intent: suppressed outright (|eval_cp|=%d > 500) mover=%s",
            abs(ctx.eval_cp),
            ctx.mover,
        )
        return []
    from app.core.commentary.features.intent import build_intent_claim

    claim = build_intent_claim(
        ctx.start_board or chess.Board(), ctx.leaf_board, _side_label(ctx.mover)
    )
    return [claim] if claim is not None else []
