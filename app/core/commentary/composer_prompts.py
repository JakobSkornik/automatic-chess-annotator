"""System prompts, text blocks, and JSON schemas for the LLM composer.

All move-commentary prompt text lives here (kept out of the orchestration
logic) so prompts can be reviewed, tuned, and reused on their own. The
``AdvancedCommentService`` imports the assembled prompts and schemas from this
module; it never inlines prompt text.
"""

from __future__ import annotations

from typing import Any

from app.models.chess_events import MoveCategory

# ---------------------------------------------------------------------------
# Strategic-archetype background ideas (woven into the system prompt as texture)
# ---------------------------------------------------------------------------
ARCHETYPE_IDEAS: dict[str, list[str]] = {
    "opposite_side_race": [
        "Tempo on the attacker's wing beats material.",
        "Open lines toward the enemy king decide races.",
        "Central breaks can defuse a wing attack.",
    ],
    "iqp": [
        "The IQP side seeks piece activity and open files.",
        "Blockading the IQP is a key defensive plan.",
    ],
    "minority_attack": [
        "Create a weakness on the minority wing with pawn levers.",
        "Rooks belong on the open file toward the target.",
    ],
    "hedgehog": [
        "Elastic pawn chain; breaks with b5 or f5 come later.",
        "Pieces re-route behind the pawns before the rupture.",
    ],
    "closed_maneuvering": [
        "Regroup knights to better squares; avoid pawn tension.",
        "Probe for weaknesses before committing a break.",
    ],
    "endgame_technique": [
        "Activate the king; create passed pawns with tempo.",
        "Opposition and zugzwang motifs decide many endings.",
    ],
    "king_hunt": [
        "Forcing checks drive the king into the open.",
        "Sacrifices clear escape squares.",
    ],
    "simplification_endgame": [
        "Trade into a won or holdable ending.",
        "Remove the opponent's active pieces first.",
    ],
    "maroczy_bind": [
        "Space clamp on d5; knights hop to ideal squares.",
        "Black seeks c5 or f5 breaks under restraint.",
    ],
    "carlsbad": [
        "Minority attack on the queenside is a main plan.",
        "The exchange variation changes pawn structure goals.",
    ],
    "other": [
        "Connect rooks and improve the worst piece.",
        "Match plans to the pawn structure center type.",
    ],
}


def composer_role_key(category_value: str | None) -> str:
    """Collapse MoveCategory routing to four composer roles."""
    c = (category_value or MoveCategory.POSITIONAL.value).strip().lower()
    if c == MoveCategory.BOOK.value:
        return "book"
    if c in (
        MoveCategory.TACTICAL.value,
        MoveCategory.FORCING.value,
        MoveCategory.DEFENSIVE.value,
    ):
        return "tactical_forcing"
    if c in (MoveCategory.INACCURACY.value, MoveCategory.CRITICAL.value):
        return "mistake_explainer"
    return "positional_plan"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EPISODE_NARRATIVE_PROMPT = (
    "You are a grandmaster chess commentator narrating the story of a game phase.\n"
    "Input begins with STRATEGIC THEMES aggregated for the episode; then each move line is:\n"
    "SAN | eval | move_quality | key_moment | tactical_motifs (strategic motifs are omitted per-move).\n"
    "Write 2-4 sentences on plans, how the phase evolved, and the critical idea. Name motifs only when "
    "they change the story.\n"
    "Open with a concrete board observation (a piece, square, file, or pawn break)—do NOT begin with "
    '"In this phase", "During this", "This phase is marked by", or "In this complex".\n'
    "Cite only squares and pieces verifiable from the move list. Do NOT invent specific diagonals, "
    "files, pins, or piece placements not implied by the SAN sequence.\n"
    'Output strictly valid JSON: {"commentary": "2-4 sentences"}'
)

STYLE_GUIDE_BLOCK = (
    "AUDIENCE ~1500–2000: explain WHY in plain chess language, not motif laundry lists.\n"
    "VOICE — coach, not engine dump.\n"
    '- Prefer at most one explicit eval number in the whole comment; otherwise use words (~"about a pawn").\n'
    "- Motifs: only name a motif key from MOTIF HINTS if you can state the exact attacker square, target square, "
    "and (for pin/skewer) the piece behind; otherwise use named_motifs=[] and plain English.\n"
    "- For captures, lead with what was captured (piece + square). Do NOT frame your own capture as "
    '"eliminating a threat" unless the POSITION block explicitly states the captured piece was attacking '
    "one of your pieces.\n"
    "- On captures or checks, include a short forcing line as [pv:san1 san2 ...] (2-3 plies).\n"
    '- Opponent-aware phrasing ("Black can now ...").\n'
    "- In openings still in theory, cite opening name/code and typical plans when known.\n"
    "- Endgames: name the decisive element (king, passed pawn, bishop colour, etc.).\n"
    "- Keep prose tight: ~50 words for low-effort tiers; up to ~90 for medium.\n"
    '- One short paragraph only; no labeled lists. Never start with "Immediate:", "Future:", "Counterfactual:".\n'
)

EXPLAIN_WHY_BLOCK = (
    "WORKFLOW\n"
    "1) Identify the concrete idea this move embodies (attacks, frees a line, fixes a weakness, prepares a break).\n"
    "   NEVER describe the opponent's refutation/PV continuation as if it were the played move.\n"
    "2) Say why this is desirable or costly for side to move AFTER the ply, referencing alternatives when useful.\n"
    "   For mistakes: name the refutation idea IN WORDS before the [pv:...] snippet.\n"
    "3) Compare briefly to engine best-move when PLAYED differs from BEST (POSITION block), unless BOOK tier.\n"
    "4) Only after (1)-(3): optionally attach ONE motif hint from MOTIF HINTS if genuinely expressed with the "
    "attacker/target/(piece behind) squares named in prose.\n\n"
    "RAG MASTER ANNOTATIONS block (when present):\n"
    "- If one idea obviously applies here, summarize it once in prose (do NOT quote verbatim).\n"
    "- Populate rag_applied=true and rag_idea_used with ONE sentence naming that idea.\n"
    '- If none apply, rag_applied=false and rag_idea_used="".\n'
)

PLAIN_OUTPUT_INSTRUCTIONS = (
    "\nOUTPUT: JSON only (no markdown).\n"
    '- "named_motifs": array of 0–1 motif keys you anchored in prose (subset of MOTIF HINTS).\n'
    '- "text": plain prose paragraph (SAN tokens plain; forcing lines as [pv:san san ...]).\n'
    '  When the user message includes "Forcing line ready:" with a [pv:...] token, copy that '
    '[pv:...] snippet verbatim into "text".\n'
    '- "better_alternative": one sentence-ready clause (Better was … / Instead … with reason), or "" if irrelevant.\n'
    "  REQUIRED for inaccuracies/mistakes/blunders/critical classifications; optional otherwise.\n"
    '- "rag_idea_used": one sentence naming the reused master-note idea (or "").\n'
    '- "rag_applied": boolean — true iff MASTER ANNOTATIONS influenced your explanation.\n'
)


def _archetype_texture_system_block(archetype: str) -> str:
    arch = (archetype or "other").strip() or "other"
    ideas = ARCHETYPE_IDEAS.get(arch, ARCHETYPE_IDEAS["other"])
    return (
        "STRATEGIC ARCHETYPE TEXTURE (background only; do not quote verbatim; "
        "ground plans in these ideas):\n"
        f"Archetype: {arch}\n" + "\n".join(f"- {line}" for line in ideas)
    )


COMPOSER_ROLE_BODIES: dict[str, str] = {
    "book": (
        "ROLE: BOOK / theory — openings where development and tabiya still matter.\n"
        "- 1 crisp sentence tying the SAN to typical plans.\n"
        "- Reference opening name/code when MOTIF hints or POSITION mention them.\n"
        '- better_alternative "" unless theory clearly rejects the move.\n'
    ),
    "tactical_forcing": (
        "ROLE: TACTICAL / FORCING / DEFENCE — captures, checks, tactical shots, tactical defence.\n"
        "- Lead with forcing consequences and concrete squares.\n"
        "- Mention defence only when STOPPING a direct threat matters.\n"
    ),
    "positional_plan": (
        "ROLE: POSITIONAL PLAN / PROPHYLAXIS — structure, slow manoeuvres, quiet improvements.\n"
        "- Name files, pawn breaks, weaknesses, timing — not jargon stacks.\n"
    ),
    "mistake_explainer": (
        "ROLE: MISTAKE & CRITICAL MOMENTS — eval swings, dubious choices, decisive branches.\n"
        "- Mandatory better_alternative whenever BEST_MOVE differs materially from PLAYED and quality is dubious.\n"
        "- Tie alternatives to measurable plans (outposts, breaks, king safety).\n"
    ),
}


def composer_system_prompt(role_key: str, *, archetype: str | None = None) -> str:
    """Assemble the full system prompt for a composer role (+ optional archetype texture)."""
    role_body = COMPOSER_ROLE_BODIES.get(
        role_key, COMPOSER_ROLE_BODIES["positional_plan"]
    )
    mid = role_body.strip() + "\n"
    if archetype is not None:
        mid += "\n" + _archetype_texture_system_block(archetype) + "\n"
    return (
        STYLE_GUIDE_BLOCK
        + "\n\n"
        + EXPLAIN_WHY_BLOCK
        + "\n\n"
        + mid
        + PLAIN_OUTPUT_INSTRUCTIONS
    )


BOOK_COMPOSER_PROMPT = composer_system_prompt("book")
TACTICAL_FORCING_COMPOSER_PROMPT = composer_system_prompt("tactical_forcing")
POSITIONAL_PLAN_COMPOSER_PROMPT = composer_system_prompt("positional_plan")
MISTAKE_EXPLAINER_COMPOSER_PROMPT = composer_system_prompt("mistake_explainer")

COMPOSER_ROLE_PROMPTS: dict[str, str] = {
    "book": BOOK_COMPOSER_PROMPT,
    "tactical_forcing": TACTICAL_FORCING_COMPOSER_PROMPT,
    "positional_plan": POSITIONAL_PLAN_COMPOSER_PROMPT,
    "mistake_explainer": MISTAKE_EXPLAINER_COMPOSER_PROMPT,
}

# ---------------------------------------------------------------------------
# JSON schemas for OpenAI Structured Outputs (strict)
# ---------------------------------------------------------------------------
COMPOSER_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "named_motifs": {"type": "array", "items": {"type": "string"}},
        "text": {"type": "string"},
        "better_alternative": {"type": "string"},
        "rag_idea_used": {"type": "string"},
        "rag_applied": {"type": "boolean"},
    },
    "required": [
        "named_motifs",
        "text",
        "better_alternative",
        "rag_idea_used",
        "rag_applied",
    ],
    "additionalProperties": False,
}

EPISODE_COMMENTARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"commentary": {"type": "string"}},
    "required": ["commentary"],
    "additionalProperties": False,
}
