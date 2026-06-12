"""Guid-format comment composition with LLM enrichment.

The shape of every mid/end-phase comment follows Fig. 5.2 of the dissertation:

    {move} {verdict} after {shortened variation} ({eval}, Stockfish:{depth}).
    {fired-rule claims}. [Better was {best move}: {its claims}.]

``HUMANIZATION_LEVEL`` selects how the facts are rendered:

  0  deterministic template join (zero hallucination baseline)
  1  LLM restates the facts fluently; no additions
  2  LLM may additionally weave in ONE flavor/context clause from the
     supplied enrichment (opening lore, game context, foreshadowing)

All levels share identical facts. The LLM contract is enforced after the
call: the eval token and the PV token must survive verbatim, otherwise the
deterministic rendering is used instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.models.comment_facts import BestAlternative, CommentFacts

logger = logging.getLogger(__name__)


def humanization_level() -> int:
    try:
        lvl = int(os.environ.get("HUMANIZATION_LEVEL", "2"))
    except ValueError:
        lvl = 2
    return max(0, min(2, lvl))


# ---------------------------------------------------------------------------
# Shared fact tokens
# ---------------------------------------------------------------------------

def eval_token(facts: CommentFacts) -> str:
    if facts.eval_mate is not None:
        return f"(#{abs(facts.eval_mate)}, {facts.engine}:{facts.depth})"
    if facts.eval_cp is None:
        return ""
    return f"({facts.eval_cp / 100:+.2f}, {facts.engine}:{facts.depth})"


def pv_token(facts: CommentFacts) -> str:
    line = facts.display_line
    if not line or not line.line_san:
        return ""
    return "[pv:" + " ".join(line.line_san) + "]"


def _alt_pv_token(alt: BestAlternative) -> str:
    if not alt.display_line or not alt.display_line.line_san:
        return ""
    return "[pv:" + " ".join(alt.display_line.line_san) + "]"


def _move_label(facts: CommentFacts) -> str:
    move_no = (facts.ply + 1) // 2
    dots = "." if facts.mover == "White" else "..."
    return f"{move_no}{dots}{facts.san}"


# ---------------------------------------------------------------------------
# Level 0 — deterministic template
# ---------------------------------------------------------------------------

def render_facts_template(facts: CommentFacts) -> str:
    parts: List[str] = []
    head = f"{_move_label(facts)} {facts.verdict}"
    pv = pv_token(facts)
    ev = eval_token(facts)
    if pv:
        head += f" after {pv}"
    if ev:
        head += f" {ev}"
    parts.append(head + ".")

    if facts.claims:
        parts.append(" ".join(c.text for c in facts.claims))

    alt = facts.better_alternative
    if alt is not None:
        alt_pv = _alt_pv_token(alt)
        s = f"Better was {alt.san}"
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " " + " ".join(c.text for c in alt.claims)
        parts.append(s)
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Levels 1/2 — LLM enrichment under a fact contract
# ---------------------------------------------------------------------------

FACTS_COMMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}

GUID_COMPOSER_SYSTEM = (
    "You are a chess annotator writing in the style of classic annotated game "
    "collections: concrete, declarative, instructive.\n\n"
    "You are given INVIOLABLE FACTS about one move: a verdict, a variation token, "
    "an evaluation token, and zero or more positional claims produced by a "
    "rule-based expert system. Your job is ONLY to phrase them well.\n\n"
    "HARD RULES:\n"
    "- Copy the EVAL token and the PV token into the text VERBATIM, unchanged.\n"
    "- State the verdict and restate EVERY claim (you may rephrase fluently, "
    "merge related claims into one sentence, and vary word choice).\n"
    "- NEVER add a positional assertion that is not among the claims: no new "
    "squares, files, motifs, threats, plans, or piece judgments.\n"
    "- Express the evaluation ONLY through the verdict words and the eval token; "
    "never convert centipawns into 'pawns up' language.\n"
    "- If BETTER ALTERNATIVE is present, end with one sentence: 'Better was "
    "{move}...' using its verdict, claims and PV token under the same rules.\n"
    "- One paragraph. No lists, no headers, no engine-worship phrasing.\n"
)

_LEVEL_RULES = {
    1: (
        "STYLE: terse and factual, ~40-70 words. Connective tissue only — "
        "no flavor, no context beyond the facts.\n"
    ),
    2: (
        "STYLE: ~60-110 words. You MAY additionally use AT MOST ONE short "
        "clause or sentence drawn from the ENRICHMENT block (opening "
        "background, the game's broader arc, or what this leads to later). "
        "Enrichment may set the scene or foreshadow, but never adds new "
        "positional analysis of the current move. If the enrichment is "
        "irrelevant, omit it.\n"
    ),
}


def build_facts_user_prompt(
    facts: CommentFacts,
    *,
    level: int,
    enrichment: Optional[Dict[str, Any]] = None,
) -> str:
    """User message: the fact block plus (level 2) the enrichment block."""
    claims_lines = [
        f"- {c.text}" + (f" [{c.flag_note}]" if c.flag_note else "")
        for c in facts.claims
    ] or ["- (no positional claims fired; comment on verdict and line only)"]

    blocks: List[str] = [
        "INVIOLABLE FACTS:",
        f"Move: {_move_label(facts)} (played by {facts.mover}, {facts.phase}game)",
        f"Verdict: this move {facts.verdict}",
        f"EVAL token (copy verbatim): {eval_token(facts)}",
        f"PV token (copy verbatim): {pv_token(facts)}",
        "Claims:",
        *claims_lines,
    ]
    alt = facts.better_alternative
    if alt is not None:
        alt_claims = [f"- {c.text}" for c in alt.claims] or ["- (none)"]
        blocks += [
            "",
            "BETTER ALTERNATIVE:",
            f"Move: {alt.san}",
            f"Verdict: {alt.verdict}",
            f"PV token (copy verbatim): {_alt_pv_token(alt)}",
            "Claims:",
            *alt_claims,
        ]
    if level >= 2 and enrichment:
        enr_lines: List[str] = []
        for key in ("opening", "episode_theme", "game_so_far", "what_happens_later", "master_note"):
            val = enrichment.get(key)
            if val:
                enr_lines.append(f"{key}: {val}")
        if enr_lines:
            blocks += ["", "ENRICHMENT (use at most one element, optional):", *enr_lines]
    return "\n".join(blocks)


def build_facts_system_prompt(level: int) -> str:
    return GUID_COMPOSER_SYSTEM + "\n" + _LEVEL_RULES.get(level, _LEVEL_RULES[1])


# ---------------------------------------------------------------------------
# Post-validation
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def validate_facts_comment(text: str, facts: CommentFacts) -> bool:
    """The fact contract: eval and PV tokens must survive verbatim."""
    t = _normalize(text)
    if not t:
        return False
    ev = eval_token(facts)
    if ev and ev not in t:
        return False
    pv = pv_token(facts)
    if pv and _normalize(pv) not in t:
        return False
    alt = facts.better_alternative
    if alt is not None and alt.san and alt.san not in t:
        return False
    return True


def parse_facts_response(raw: str) -> str:
    try:
        obj = json.loads(raw)
        return str(obj.get("text") or "").strip()
    except Exception:
        return (raw or "").strip()


async def compose_facts_comment(
    service: Any,
    facts: CommentFacts,
    *,
    model: Optional[str],
    effort: str,
    enrichment: Optional[Dict[str, Any]] = None,
    level: Optional[int] = None,
) -> Dict[str, Any]:
    """Render the facts at the configured humanization level.

    Returns {"text": ..., "rendering": "template"|"llm", "level": int,
    "contract_ok": bool}. Falls back to the deterministic template whenever
    the LLM output violates the fact contract.
    """
    lvl = humanization_level() if level is None else max(0, min(2, int(level)))
    template = render_facts_template(facts)
    configured = True
    try:
        configured = bool(service._provider.is_configured())
    except Exception:
        configured = False
    if lvl == 0 or not configured:
        return {"text": template, "rendering": "template", "level": lvl, "contract_ok": True}

    system = build_facts_system_prompt(lvl)
    user = build_facts_user_prompt(facts, level=lvl, enrichment=enrichment)
    try:
        raw = await service._llm_call_json_schema(
            system,
            user,
            model=model,
            effort=effort,
            schema=FACTS_COMMENT_SCHEMA,
            schema_name="facts_comment",
            max_output_tokens=500,
        )
        text = parse_facts_response(raw)
    except Exception as e:
        logger.warning("facts composer LLM call failed (ply %s): %s", facts.ply, e)
        text = ""

    if text and validate_facts_comment(text, facts):
        return {"text": text, "rendering": "llm", "level": lvl, "contract_ok": True}
    if text:
        logger.info(
            "facts comment failed contract at ply %s — using template", facts.ply
        )
    return {"text": template, "rendering": "template", "level": lvl, "contract_ok": False}
