"""LLM prompt assets and the per-move user prompt builder.

Per the house rules, the large text blocks (system rules, audience registers,
archetype rules, response schema) live here in their own module, kept out of the
orchestration logic that consumes them."""

from __future__ import annotations

from typing import Any

from app.core.commentary.rules.constants import INFERIOR_ALT_WEAKER_CP
from app.models.comment_facts import Claim, CommentFacts

from .framing import ALT_MATERIAL_GAP_CP, alt_gap_cp
from .tokens import _alt_pv_token, _move_label, eval_token, pv_token

SINGLE_LEVEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}

GUID_COMPOSER_SYSTEM = (
    "You are a chess annotator. You are given INVIOLABLE FACTS about one move: "
    "a verdict, a variation token, an evaluation token, and positional claims "
    "produced by a rule-based expert system (each claim may come in a change-form "
    "and a state-form — use whichever reads naturally). Write ONE rendering of "
    "the annotation for the audience specified below.\n\n"
    "HARD RULES:\n"
    "- Copy the EVAL token and every PV token into the text VERBATIM, unchanged.\n"
    "- State the verdict and convey EVERY claim (rephrase fluently; merging "
    "related claims into one sentence is encouraged).\n"
    "- NEVER add a position-specific assertion that is not among the claims: no "
    "new squares, files, piece placements, threats, plans, tactics, or judgments "
    "about THIS position. General chess knowledge about a named concept (what a "
    "doubled pawn is, why the bishop pair usually matters) is allowed only where "
    "the audience rules below say so — and only about features named in claims.\n"
    "- Express the evaluation ONLY through the verdict words and the eval token; "
    "never convert centipawns into 'pawns up' language.\n"
    "- MOVER PERSPECTIVE: explain why the move serves the side that played it. "
    "Claims listed as MERITS are what the move achieves — lead with them. Claims "
    "listed as CONCESSIONS favor the opponent: phrase them strictly as trade-offs "
    "('in return', 'at the cost of') for sound moves, or as the move's drawbacks "
    "('now the opponent ...') when the concession mode says 'consequence'. NEVER "
    "present a concession as an achievement of the move.\n"
    "- TIMING: every claim is tagged <immediate> or <in the line>. State "
    "<immediate> claims as fact. A claim tagged <in the line> is NOT yet true "
    "after the move — it only develops deeper in the variation, so you MUST hedge "
    "it ('may lead to', 'risks', 'could leave', 'is set to') and never assert it "
    "as already achieved.\n"
    "- If REFUTATION is present, the text MUST name that move as what punishes "
    "the played move (it is the board-level reason the move fails).\n"
    "- If BETTER ALTERNATIVE is present, end with one sentence naming it. When its "
    "Significance is 'materially better', frame its claims as the opportunity the "
    "mover passed up: 'missed the chance to ...' for <immediate> claims, 'missed "
    "the potential to ...' for <in the line> claims. When 'roughly equal', present "
    "it neutrally as a comparable option, never as a miss. Always include its "
    "verdict and PV token.\n"
    "- If INFERIOR ALTERNATIVE is present instead, the played move was the best: "
    "add one sentence contrasting it with that weaker runner-up (why the played "
    "move was better), using its verdict and PV token. Never frame it as a miss.\n"
    "- One paragraph. No lists, no headers, no engine-worship.\n"
)

AUDIENCE_BLOCKS: dict[str, str] = {
    "expert": (
        "AUDIENCE — expert: register of Chess Informant / grandmaster game "
        "collections. Dry, terse, declarative; ~25-55 words besides tokens. Do "
        "NOT explain features, concepts or terms — the reader is a strong "
        "player. No flavor, no narrative, no rhetorical questions. Merge the "
        "claims into compact compound sentences.\n"
    ),
    "intermediate": (
        "AUDIENCE — intermediate: club player. ~50-85 words besides tokens. "
        "Standard terms used, never defined. You MAY add one short clause on "
        "why the single most important claimed feature generally matters. "
        "Plain, practical tone.\n"
    ),
    "beginner": (
        "AUDIENCE — beginner: learning player. ~70-115 words besides tokens. "
        "Explain in a simple way each targeted positional feature (e.g. what a "
        "passed pawn is) the first time it is named; avoid jargon or define it "
        "inline; friendly instructive tone; you MAY close with one short "
        "general takeaway tied to a claimed concept. Never patronize and never "
        "invent new analysis.\n"
    ),
}

ARCHETYPE_RULES: dict[str, str] = {
    "engine_choice": (
        "ARCHETYPE — best move: the played move is the strongest available. Open "
        "by affirming it as the best/most accurate move (do NOT mention the engine "
        "or 'engine's choice') and say what it leads to (from the verdict and "
        "claims). There is no BETTER alternative, so do not invent a flaw, a "
        "downside, or a 'but' — though if an INFERIOR ALTERNATIVE is provided you "
        "may contrast it to show why the played move is stronger.\n"
    ),
    "brilliant_sacrifice": (
        "ARCHETYPE — brilliant sacrifice: the move gives up material yet the "
        "side to move comes out clearly better. Lead by calling it a sacrifice "
        "and explain why it works using the claims and verdict. You MAY mark it "
        "'!!'. Do NOT name the specific sacrificed piece — you are not told "
        "which piece is given up, so write 'a sacrifice' or 'sacrifices "
        "material', never e.g. 'bishop sacrifice' or 'rook sacrifice'. "
        "Do not call it dubious.\n"
    ),
    "inaccuracy_missed": (
        "ARCHETYPE — inaccuracy / missed opportunity: open with what the move "
        "passed up. Lead with the better alternative framed as the opportunity "
        "missed (per the BETTER ALTERNATIVE rules), then the eval consequence.\n"
    ),
}

ENRICHMENT_RULES = (
    "ENRICHMENT block (optional): intermediate and beginner MAY weave in at most "
    "one element (opening background or what this leads to later in the game) as "
    "scene-setting; expert must ignore it entirely.\n"
    "Do NOT open with or restate the opening name/ECO (e.g. 'In this <Opening>...'); "
    "the reader already sees it. Reference an opening idea only if it directly "
    "explains THIS move.\n"
)


def _claim_prompt_line(c: Claim) -> str:
    line = f"- {c.text}"
    if c.text_state:
        line += f" | state-form: {c.text_state}"
    if c.flag_note:
        line += f" [{c.flag_note}]"
    if c.features_involved:
        line += f" (features: {', '.join(c.features_involved[:3])})"
    line += " <in the line>" if c.realization == "envisioned" else " <immediate>"
    return line


def _alternative_block(facts: CommentFacts) -> list[str]:
    alt = facts.better_alternative
    if alt is None:
        return []
    if alt.is_inferior:
        gap = alt_gap_cp(facts)  # mover-POV; <= 0 for a runner-up
        standing = (
            "clearly weaker — contrast it to show why the played move is stronger"
            if gap is None or -gap >= INFERIOR_ALT_WEAKER_CP
            else "roughly as good — present as a comparable option, not a mistake"
        )
        return [
            "",
            "RUNNER-UP ALTERNATIVE (the played move was the best; this is the "
            "second-best — use it only as a contrast, NEVER as a move that was "
            "missed):",
            f"Move: {alt.san}",
            f"Verdict: {alt.verdict}",
            f"Standing: {standing}",
            f"PV token (copy verbatim): {_alt_pv_token(alt)}",
            "Claims (what this line would have given):",
            *([_claim_prompt_line(c) for c in alt.claims] or ["- (none)"]),
        ]
    gap = alt_gap_cp(facts)
    significance = (
        "roughly equal — present as a comparable option, NOT as a miss"
        if gap is not None and gap < ALT_MATERIAL_GAP_CP
        else "materially better — the mover passed up this opportunity"
    )
    return [
        "",
        "BETTER ALTERNATIVE:",
        f"Move: {alt.san}",
        f"Verdict: {alt.verdict}",
        f"Significance: {significance}",
        f"PV token (copy verbatim): {_alt_pv_token(alt)}",
        "Claims (the gains the mover forwent):",
        *([_claim_prompt_line(c) for c in alt.claims] or ["- (none)"]),
    ]


def _enrichment_block(enrichment: dict[str, Any] | None) -> list[str]:
    if not enrichment:
        return []
    lines = [
        f"{key}: {enrichment[key]}"
        for key in ("opening", "what_happens_later")
        if enrichment.get(key)
    ]
    if not lines:
        return []
    return ["", "ENRICHMENT (intermediate/beginner only, optional):", *lines]


def build_facts_user_prompt(
    facts: CommentFacts,
    *,
    enrichment: dict[str, Any] | None = None,
    archetype: str | None = None,
) -> str:
    merit_lines = [_claim_prompt_line(c) for c in facts.claims if not c.is_concession]
    concession_lines = [_claim_prompt_line(c) for c in facts.claims if c.is_concession]
    if not merit_lines and not concession_lines:
        merit_lines = [
            "- (no positional claims fired; comment on verdict and line only)"
        ]

    blocks: list[str] = [
        "INVIOLABLE FACTS:",
        f"Move: {_move_label(facts)} (played by {facts.mover}, {facts.phase}game)",
        f"Verdict: this move {facts.verdict}",
        f"EVAL token (copy verbatim): {eval_token(facts)}",
        f"PV token (copy verbatim): {pv_token(facts)}",
    ]
    if archetype and archetype != "neutral":
        blocks.append(f"Move character: {archetype} (see ARCHETYPE rule)")
    if facts.refutation_san:
        blocks.append(f"REFUTATION (must be named): {facts.refutation_san}")
    blocks += [
        f"MERITS (what the move achieves for {facts.mover}):",
        *(merit_lines or ["- (none)"]),
    ]
    if concession_lines:
        blocks += [
            f"CONCESSIONS (favor the opponent; mode: {facts.concession_mode}):",
            *concession_lines,
        ]
    blocks += _alternative_block(facts)
    blocks += _enrichment_block(enrichment)
    return "\n".join(blocks)
