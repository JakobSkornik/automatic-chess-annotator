"""Threat rules: concrete pressure on enemy pieces (hanging / weak)."""

from __future__ import annotations

from app.models.comment_facts import Claim

from .constants import SIDES
from .context import _benef, _Ctx, _side_label


def rule_threats(ctx: _Ctx) -> list[Claim]:
    """Fires when a side leaves an enemy piece hanging — a concrete threat the
    opponent must answer. (``{side}_HANGING`` counts the enemy pieces hanging to
    ``side``, so a rise means the other side has a piece in danger.)"""
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_HANGING"
        pair = ctx.flag_pair(name)
        if not (pair and pair[1] > pair[0]):
            continue
        victim = _side_label("BLACK" if side == "WHITE" else "WHITE")
        out.append(
            Claim(
                rule_id="hanging_piece",
                beneficiary=_benef(side),
                text=f"{victim} leaves a piece hanging.",
                text_state=f"{victim} has a hanging piece.",
                features_involved=[name, f"{side}_WEAK_ENEMIES"],
                delta_cp=ctx.claim_cp(name),
                flag_note=ctx.flag_change(name),
            )
        )
    return out


def rule_pin(ctx: _Ctx) -> list[Claim]:
    """Fires when a side wins a pin against an enemy piece — a concrete bind that
    holds even when the evaluation is already decided."""
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PINS"
        pair = ctx.flag_pair(name)
        if not (pair and pair[1] > pair[0]):
            continue
        victim = _side_label("BLACK" if side == "WHITE" else "WHITE")
        out.append(
            Claim(
                rule_id="piece_pinned",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} pins a {victim.lower()} piece.",
                text_state=f"{_side_label(side)} has a piece pinned.",
                features_involved=[name],
                delta_cp=ctx.claim_cp(name),
                flag_note=ctx.flag_change(name),
            )
        )
    return out
