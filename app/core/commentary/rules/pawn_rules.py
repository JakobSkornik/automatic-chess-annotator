from __future__ import annotations

from app.core.commentary.features.guid_features import (
    doubled_pawn_files,
    passed_pawn_squares,
)
from app.models.comment_facts import Claim

from .constants import SIDES, THRESHOLDS
from .context import _benef, _benef_opp, _Ctx, _side_label, _toward
from .material import _imbalance, describe_material


def rule_material(ctx: _Ctx) -> list[Claim]:
    """Material standing in concrete, whole-unit terms — the most important
    feature, so it leads the claim list. Fires when the line changes the
    material balance (a capture nets material), describing the actual
    difference / resulting imbalance."""
    if ctx.leaf_board is None:
        return []
    changed = ctx.start_board is None or _imbalance(ctx.start_board) != _imbalance(
        ctx.leaf_board
    )
    if not changed:
        return []
    described = describe_material(ctx.leaf_board, changed=True)
    if described is None:
        return []
    text, benef, delta_cp = described
    return [
        Claim(
            rule_id="material_won",
            beneficiary=benef,
            text=text,
            text_state=text,
            features_involved=["MATERIAL_BALANCE"],
            delta_cp=delta_cp,
        )
    ]


def rule_pawn_structure(ctx: _Ctx) -> list[Claim]:
    """The dissertation's pawn-structure rule (§5.4.1), both sides. Gated on the
    weighted EVALUATE_PAWNS aggregate (the individual pawn features are now raw
    counts in different units and cannot be summed directly)."""
    out: list[Claim] = []
    feat_names = ("PAWN_DOUBLED", "PAWN_ISOLATED", "PAWN_BACKWARD", "PAWN_PASSED")
    for side in SIDES:
        net = _toward(side, ctx.delta("EVALUATE_PAWNS"))
        involved = [f"{side}_{f}" for f in feat_names] + ["EVALUATE_PAWNS"]
        if net >= THRESHOLDS["pawn_structure_evaluate"]:
            out.append(
                Claim(
                    rule_id="pawn_structure_improved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} has improved the pawn structure.",
                    text_state=f"{_side_label(side)}'s pawn structure is now improved.",
                    features_involved=involved,
                    delta_cp=net,
                )
            )
        elif -net >= THRESHOLDS["pawn_structure_evaluate"]:
            out.append(
                Claim(
                    rule_id="pawn_structure_weakened",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s pawn structure has been weakened.",
                    text_state=f"{_side_label(side)}'s pawn structure is now weaker.",
                    features_involved=involved,
                    delta_cp=-net,
                )
            )
    return out


def rule_doubled_pawns(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PAWN_DOUBLED"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            new_files = ctx.new_squares(doubled_pawn_files, side)
            where = f" {new_files[0]}-pawns" if new_files else " pawns"
            out.append(
                Claim(
                    rule_id="doubled_pawns_accepted",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} is left with doubled{where}.",
                    text_state=f"{_side_label(side)} now has doubled{where}.",
                    features_involved=[name],
                    delta_cp=ctx.claim_cp(name),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif pair and pair[1] < pair[0]:
            out.append(
                Claim(
                    rule_id="doubled_pawns_resolved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} gets rid of the doubled pawns.",
                    text_state=f"{_side_label(side)}'s doubled pawns are gone.",
                    features_involved=[name],
                    delta_cp=ctx.claim_cp(name),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_rook_behind_passer(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_ROOK_BEHIND_PASSED_PAWN"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            out.append(
                Claim(
                    rule_id="rook_behind_passer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s rook gets behind the passed pawn.",
                    text_state=f"{_side_label(side)}'s rook is behind the passed pawn.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_passed_pawn(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PAWN_PASSED"
        pair = ctx.flag_pair(name)
        d = _toward(side, ctx.delta(name))
        if pair and pair[1] > pair[0]:
            sqs = ctx.new_squares(passed_pawn_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(
                Claim(
                    rule_id="passed_pawn_created",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} obtains a passed pawn{where}.",
                    text_state=f"{_side_label(side)} has a passed pawn{where}.",
                    features_involved=[name],
                    delta_cp=max(d, 0),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif (
            pair
            and pair[1] == pair[0]
            and pair[1] > 0
            and d >= THRESHOLDS["passer_advance"]
        ):
            out.append(
                Claim(
                    rule_id="passed_pawn_advances",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s passed pawn advances dangerously.",
                    text_state=f"{_side_label(side)}'s passed pawn is far advanced.",
                    features_involved=[name],
                    delta_cp=d,
                )
            )
    return out


def rule_outside_passer(ctx: _Ctx) -> list[Claim]:
    if ctx.phase != "end":
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_OUTSIDE_PASSER"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            out.append(
                Claim(
                    rule_id="outside_passer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} obtains an outside passed pawn.",
                    text_state=f"{_side_label(side)} has an outside passed pawn.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_passer_escort(ctx: _Ctx) -> list[Claim]:
    if ctx.phase != "end":
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PASSER_KING_ESCORT"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["passer_escort"]:
            out.append(
                Claim(
                    rule_id="king_escorts_passer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s king escorts the passed pawn forward.",
                    text_state=f"{_side_label(side)}'s king supports the passed pawn.",
                    features_involved=[name, f"{side}_KING_ACTIVITY"],
                    delta_cp=d,
                )
            )
    return out
