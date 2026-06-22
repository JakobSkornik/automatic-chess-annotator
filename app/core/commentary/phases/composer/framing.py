"""Move framing: the commentary archetype (engine choice / brilliancy /
inaccuracy) and the better-alternative gap helpers that decide whether the
engine's move is "better" or merely "comparable"."""

from __future__ import annotations

from app.models.comment_facts import Claim, CommentFacts

# Below this mover-POV gap the engine's move is a comparable option, not a miss.
ALT_MATERIAL_GAP_CP = 50


def alt_gap_cp(facts: CommentFacts) -> int | None:
    """Mover-POV centipawns the alternative gains over the played move."""
    alt = facts.better_alternative
    if alt is None or alt.eval_cp is None or facts.eval_cp is None:
        return None
    gap = alt.eval_cp - facts.eval_cp  # White-POV
    return gap if facts.mover == "White" else -gap


def _is_alt_materially_better(facts: CommentFacts) -> bool:
    gap = alt_gap_cp(facts)
    return gap is None or gap >= ALT_MATERIAL_GAP_CP


def _missed_noun(claims: list[Claim]) -> str:
    """'chance' if the alternative's gain lands at once, 'potential' if it only
    develops in the line (judged on the ordered-strongest claim)."""
    return "potential" if claims and claims[0].realization == "envisioned" else "chance"


def comment_archetype(key_moment_type: str | None) -> str:
    """Map the key-moment classification to a commentary framing.

    engine_choice      the played move IS the engine's pick — confirm, don't fault
    brilliant_sacrifice a material sacrifice that holds/improves the eval
    inaccuracy_missed  an imprecise move — lead with the opportunity passed up
    neutral            the default verdict + claims shape
    """
    kmt = key_moment_type or ""
    if kmt == "brilliant":
        return "brilliant_sacrifice"
    if kmt in ("best_move", "great_move"):
        return "engine_choice"
    if kmt in ("inaccuracy", "missed_opportunity"):
        return "inaccuracy_missed"
    return "neutral"


# NB: openers must not contain any phrase in forbidden_phrases.FORBIDDEN_REGEX,
# or the scrub pass strips it mid-sentence and leaves debris ("is the ,").
_ARCHETYPE_OPENER: dict[str, str] = {
    "engine_choice": "is the strongest move here,",
    "brilliant_sacrifice": "is a brilliant sacrifice,",
}
