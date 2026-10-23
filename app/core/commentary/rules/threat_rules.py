"""Threat rules: concrete pressure on enemy pieces (hanging / weak)."""

from __future__ import annotations

import chess

from app.core.commentary.features.guid_features.features.threats import hanging
from app.models.comment_facts import Claim

from .constants import SIDES
from .context import _benef, _Ctx, _side_label


def rule_threats(ctx: _Ctx) -> list[Claim]:
    """Fires on a rise in hanging-piece pressure only with concrete board proof.

    ``{side}_HANGING`` counts the enemy pieces hanging to ``side``; the rule
    also requires a new undefended non-king target with a legal capture.
    """
    out: list[Claim] = []
    if ctx.start_board is None or ctx.leaf_board is None:
        return out
    for side in SIDES:
        name = f"{side}_HANGING"
        pair = ctx.flag_pair(name)
        if not (pair and pair[1] > pair[0]):
            continue
        # The feature tracks a broad attack/defence heuristic. Only make the
        # concrete "hanging" claim when a new, undefended non-king target has
        # at least one legal capture in the leaf position.
        new_hanging = ctx.new_squares(hanging.squares, side)
        if not new_hanging:
            continue
        square = chess.parse_square(new_hanging[0])
        piece_name = chess.piece_name(ctx.leaf_board.piece_type_at(square))
        victim = _side_label("BLACK" if side == "WHITE" else "WHITE")
        text = f"{victim}'s {piece_name} on {new_hanging[0]} is undefended and attacked."
        out.append(
            Claim(
                rule_id="hanging_piece",
                beneficiary=_benef(side),
                # The feature diff says the count of hanging pieces increased;
                # the opponent's move may have created the threat, so avoid
                # attributing the mistake to the victim side.
                text=text,
                text_state=text,
                features_involved=[name, f"{side}_WEAK_ENEMIES"],
                delta_cp=ctx.claim_cp(name),
                flag_note=ctx.flag_change(name),
            )
        )
    return out


def rule_pin(ctx: _Ctx) -> list[Claim]:
    """Fires when a side wins a pin against an enemy piece — a concrete bind that
    holds even when the evaluation is already decided."""
    from .tactical_evidence import absolute_pins

    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PINS"
        color = side == "WHITE"
        for attacker, victim, king in sorted(absolute_pins(ctx.leaf_board, color) - absolute_pins(ctx.start_board, color)):
            text = (
                f"{_side_label(side)}'s {chess.piece_name(ctx.leaf_board.piece_type_at(attacker))} "
                f"on {chess.square_name(attacker)} pins the "
                f"{chess.piece_name(ctx.leaf_board.piece_type_at(victim))} on {chess.square_name(victim)} "
                f"to the king on {chess.square_name(king)}."
            )
            out.append(
                Claim(
                    rule_id="piece_pinned",
                    beneficiary=_benef(side),
                    text=text,
                    text_state=text,
                    features_involved=[name],
                    delta_cp=ctx.claim_cp(name),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out
