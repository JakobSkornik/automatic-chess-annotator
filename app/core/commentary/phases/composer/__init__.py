"""Guid-format comment composition.

Every comment follows the dissertation's shape (Fig. 5.2):

    {move} {verdict} after {variation} ({eval}, Stockfish:{depth}).
    {fired-rule claims}. [Better was {best move}: {its claims}.]

The **commentary language level** is an audience register, not a licence to
invent. Position-specific claims come exclusively from the rule engine at
every level; what scales with the level is how much general chess knowledge
is explained around them:

  expert        Informant/Chessbase register for strong players. Dry,
                declarative, no didactics, claims merged into compact
                compound sentences. Matej's dissertation voice.
  intermediate  Club player (~1500-2000). Same facts; one brief clause on
                why the key feature matters in general. Standard terms used,
                never defined.
  beginner      Learning player. Each named feature explained in plain words
                (what a passed pawn IS), jargon avoided or defined inline,
                one short takeaway tied to a fired claim's concept.

A single LLM call renders the level chosen at submit time; the text is
validated against the fact contract (eval + PV tokens verbatim, alternative
named) and any failure falls back to the deterministic template. With no LLM
configured (or ``HUMANIZATION_LEVEL=0``) the output is always the template.

This package splits that work into cohesive modules:
  tokens       verbatim eval/PV strings and move labels
  framing      archetype + better-alternative gap helpers
  template     the deterministic (no-LLM / fallback) renderer
  prompts      LLM text blocks + the per-move user prompt builder
  validation   fact-contract check + provider-response parsing
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.models.comment_facts import CommentFacts

from .framing import ALT_MATERIAL_GAP_CP, alt_gap_cp, comment_archetype
from .prompts import (
    ARCHETYPE_RULES,
    AUDIENCE_BLOCKS,
    ENRICHMENT_RULES,
    GUID_COMPOSER_SYSTEM,
    SINGLE_LEVEL_SCHEMA,
    build_facts_user_prompt,
)
from .template import render_facts_template
from .tokens import eval_token, pv_token
from .validation import _parse_text, validate_facts_comment

logger = logging.getLogger(__name__)

LEVELS = ("expert", "intermediate", "beginner")
DEFAULT_LEVEL = "intermediate"

__all__ = [
    "ALT_MATERIAL_GAP_CP",
    "DEFAULT_LEVEL",
    "LEVELS",
    "SINGLE_LEVEL_SCHEMA",
    "alt_gap_cp",
    "build_facts_user_prompt",
    "comment_archetype",
    "compose_facts_comment",
    "eval_token",
    "humanization_level",
    "is_llm_rendering_enabled",
    "pv_token",
    "render_facts_template",
    "validate_facts_comment",
]


def humanization_level() -> int:
    """Legacy switch: 0 disables the LLM entirely (template-only output)."""
    try:
        lvl = int(os.environ.get("HUMANIZATION_LEVEL", "2"))
    except ValueError:
        lvl = 2
    return max(0, min(2, lvl))


def is_llm_rendering_enabled() -> bool:
    return humanization_level() > 0


COMPOSER_MAX_OUTPUT_TOKENS = 500


def _result(text: str, rendering: str, contract_ok: bool, level: str) -> dict[str, Any]:
    return {
        "text": text,
        "rendering": rendering,
        "contract_ok": contract_ok,
        "level": level,
    }


def _is_provider_configured(service: Any) -> bool:
    try:
        return bool(service._provider.is_configured())
    except Exception:
        return False


def _build_system_prompt(level: str, archetype: str) -> str:
    system = (
        GUID_COMPOSER_SYSTEM + "\n" + AUDIENCE_BLOCKS[level] + "\n" + ENRICHMENT_RULES
    )
    if archetype in ARCHETYPE_RULES:
        system += "\n" + ARCHETYPE_RULES[archetype]
    return system


async def _call_composer(
    service: Any, system: str, user: str, model: str | None, effort: str, ply: int
) -> str:
    """One fact-contract LLM call; returns the parsed text, or "" on failure."""
    try:
        raw = await service._llm_call_json_schema(
            system,
            user,
            model=model,
            effort=effort,
            schema=SINGLE_LEVEL_SCHEMA,
            schema_name="facts_comment",
            max_output_tokens=COMPOSER_MAX_OUTPUT_TOKENS,
        )
        return _parse_text(raw)
    except Exception as e:
        logger.warning("facts composer LLM call failed (ply %s): %s", ply, e)
        return ""


async def compose_facts_comment(
    service: Any,
    facts: CommentFacts,
    *,
    model: str | None,
    effort: str,
    enrichment: dict[str, Any] | None = None,
    level: str = DEFAULT_LEVEL,
    key_moment_type: str | None = None,
) -> dict[str, Any]:
    """Render the facts at the audience level chosen at submit time.

    Returns ``{"text": str, "rendering": "llm"|"template", "contract_ok": bool}``;
    an LLM text violating the fact contract falls back to the template.
    """
    lvl = level if level in LEVELS else DEFAULT_LEVEL
    archetype = comment_archetype(key_moment_type)
    template = render_facts_template(facts, archetype=archetype)

    if not is_llm_rendering_enabled() or not _is_provider_configured(service):
        return _result(template, "template", True, lvl)

    system = _build_system_prompt(lvl, archetype)
    user = build_facts_user_prompt(facts, enrichment=enrichment, archetype=archetype)
    candidate = await _call_composer(service, system, user, model, effort, facts.ply)

    if candidate and validate_facts_comment(candidate, facts):
        return _result(candidate, "llm", True, lvl)
    if candidate:
        logger.info(
            "facts comment failed contract at ply %s — using template", facts.ply
        )
    return _result(template, "template", False, lvl)
