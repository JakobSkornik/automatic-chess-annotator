"""Evidence-grounded composition with versioned comparison modes.

The deterministic fallback and legacy modes follow this rigid shape:

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

By default a writer owns the whole annotation with deterministic validation.
Set COMMENTARY_LLM_REVIEW_ENABLED=1 to add an isolated semantic screening call. The application supplies facts, protected values and variations.
``grounded_rewrite_v2`` preserves the earlier fragment-rewriting comparison.
Rejection falls back to a deterministic template. This screening is not proof of
chess correctness. ``COMMENTARY_RENDER_MODE=bounded`` retains the original
one-call ID/order/wording editor for comparisons. With no configured provider or
``HUMANIZATION_LEVEL=0`` the output is always the template.

This package splits that work into cohesive modules:
  tokens       verbatim eval/PV strings and move labels
  framing      archetype + better-alternative gap helpers
  template     the deterministic (no-LLM / fallback) renderer
  prompts      LLM text blocks + the per-move user prompt builder
  validation   fact-contract check + provider-response parsing
"""

from __future__ import annotations

import logging
import json
import os
from typing import Any

from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine

from .framing import ALT_MATERIAL_GAP_CP, alt_gap_cp, comment_archetype
from .prompts import (
    SINGLE_LEVEL_SCHEMA,
    build_facts_user_prompt,
)
from .template import render_facts_template
from .tokens import eval_token, pv_token
from .validation import _parse_text, validate_facts_comment
from .grounded import (
    EDITOR_SCHEMA, EDITOR_SYSTEM, EDITOR_VERSION, SESSION_SYSTEM, editor_prompt,
    has_editorial_choice, render_editor_response,
)

logger = logging.getLogger(__name__)

LEVELS = ("expert", "intermediate", "beginner")
DEFAULT_LEVEL = "intermediate"

# Beginner comments stay focused: fewer, most fundamental claims (Guid:
# what to comment on depends on the audience's strength).
BEGINNER_MAX_MERITS = 2
BEGINNER_MAX_CONCESSIONS = 1

# Legacy experiment settings retained for importing research scripts. The
# current renderer preserves evidence endpoints; these caps are not applied.
LEVEL_BASE_PLIES: dict[str, int] = {
    "beginner": 3,
    "intermediate": 5,
    "expert": 12,  # effectively "no extra trim" (max_display_plies default)
}

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
        "prompt_version": EDITOR_VERSION,
    }


def _is_provider_configured(service: Any) -> bool:
    try:
        return bool(service._provider.is_configured())
    except Exception:
        return False


def _audience_claims(facts: CommentFacts, level: str) -> list[Claim]:
    """Trim the claim list to the audience's budget (strongest kept).

    Beginner/intermediate: two merits and one concession; expert: everything
    fired. The stored CommentFacts stay complete — this only shapes
    what the comment itself conveys."""
    if level == "expert":
        return facts.claims
    merits = [c for c in facts.claims if not c.is_concession]
    concessions = [c for c in facts.claims if c.is_concession]
    return merits[:BEGINNER_MAX_MERITS] + concessions[:BEGINNER_MAX_CONCESSIONS]


def _audience_line(line: EnvisionedLine | None, level: str) -> EnvisionedLine | None:
    """Keep the evidence endpoint: audience adaptation changes prose, not facts.

    Shortening here would require recomputing every claim and feature series.
    The facts builder has already selected a bounded supporting variation.
    """
    return line


def _audience_view(facts: CommentFacts, level: str) -> CommentFacts:
    """Apply the audience's claim budget while preserving evidence endpoints."""
    updates: dict[str, Any] = {"claims": _audience_claims(facts, level)}
    if facts.display_line is not None:
        updates["display_line"] = _audience_line(facts.display_line, level)
    alt = facts.better_alternative
    if alt is not None and alt.display_line is not None:
        updates["better_alternative"] = alt.model_copy(
            update={"display_line": _audience_line(alt.display_line, level)}
        )
    return facts.model_copy(update=updates)


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
    audience_facts = _audience_view(facts, lvl)
    template = render_facts_template(audience_facts, archetype=archetype)

    if (audience_facts.terminal_result or not is_llm_rendering_enabled()
            or not _is_provider_configured(service)):
        return dict(_result(template, "template", True, lvl),
                    commentary_status="template",
                    commentary_reason="terminal_result") if audience_facts.terminal_result else dict(
                        _result(template, "template", True, lvl),
                        commentary_status="template_disabled" if not is_llm_rendering_enabled()
                        else "template_provider_failed",
                        commentary_reason="llm_disabled" if not is_llm_rendering_enabled()
                        else "provider_not_configured")

    mode = os.environ.get("COMMENTARY_RENDER_MODE", "grounded_rewrite")
    if mode in {"grounded_rewrite", "grounded_rewrite_v2"}:
        # The writer selects the explanation, so it must see the complete ranked
        # evidence. A pre-LLM two-merit cap could remove the decisive pin/reply.
        return await _compose_rewrite(service, facts, lvl, archetype,
                                      template, model=model, effort=effort,
                                      legacy=mode == "grounded_rewrite_v2")
    if mode != "bounded":
        raise ValueError(f"Unknown COMMENTARY_RENDER_MODE: {mode}")

    if not has_editorial_choice(audience_facts):
        # Preserve the same beginner explanations as an approved model response.
        # There is only one effective ordering/wording, so no call is necessary.
        choices = {"main": [{"id": i, "form": "change"} for i in range(len(audience_facts.claims))],
                   "alternative": [{"id": i, "form": "change"} for i in range(
                       len(audience_facts.better_alternative.claims)
                       if audience_facts.better_alternative else 0)]}
        text = render_editor_response(json.dumps(choices), audience_facts, lvl, archetype)
        return _result(text, "template", True, lvl)

    try:
        context_profile = os.environ.get("COMMENTARY_CONTEXT_PROFILE", "board_grounded")
        system = EDITOR_SYSTEM
        if getattr(service, "is_commentary_session", False):
            system += SESSION_SYSTEM
        raw = await service._llm_call_json_schema(
            system, editor_prompt(audience_facts, lvl, context_profile=context_profile), model=model,
            effort=effort, schema=EDITOR_SCHEMA, schema_name="grounded_comment",
            max_output_tokens=COMPOSER_MAX_OUTPUT_TOKENS,
        )
        text = render_editor_response(raw, audience_facts, lvl, archetype)
        return _result(text, "llm", True, lvl)
    except Exception as exc:
        logger.warning("Grounded composer rejected at ply %s: %s", facts.ply, type(exc).__name__)
    return _result(template, "template", False, lvl)


async def _compose_rewrite(service, facts, level, archetype, template, *, model, effort, legacy=False):
    if legacy:
        from . import rewrite
    else:
        from . import narrative as rewrite

    result = {
        "text": template, "rendering": "template", "contract_ok": True, "level": level,
        "prompt_version": rewrite.VERSION, "rendering_contract": rewrite.VERSION,
        "semantic_review": "not_needed", "review_scope": "model_screening_not_chess_ground_truth",
    }
    if legacy and not facts.claims and not (facts.better_alternative and facts.better_alternative.claims):
        return result
    review_enabled = os.environ.get("COMMENTARY_LLM_REVIEW_ENABLED", "0") == "1"
    result["review_scope"] = ("model_screening_not_chess_ground_truth" if review_enabled
                              else "deterministic_validation_only")
    stage = "validation"
    try:
        user, evidence = rewrite.prompt(facts, level)
        result["claim_catalog"] = evidence
        stage = "provider"
        raw = await service._llm_call_json_schema(
            rewrite.SYSTEM, user, model=model, effort=effort,
            schema=rewrite.SCHEMA, schema_name="grounded_rewrite", max_output_tokens=1200,
        )
        result["raw_proposal"] = raw
        stage = "validation"
        proposal, sentences = rewrite.parse_proposal(raw, evidence)
        result["selected_claims"] = proposal
        if not review_enabled:
            result.update(text=rewrite.render(proposal, evidence, facts, archetype),
                          rendering="llm", semantic_review="disabled",
                          commentary_status="llm_unreviewed")
            return result
        stage = "review"
        review = await service.review_json(
            rewrite.REVIEW_SYSTEM, json.dumps({"sentences": sentences}, ensure_ascii=False),
            model=model, effort=effort, schema=rewrite.REVIEW_SCHEMA,
            schema_name="grounded_rewrite_review", max_output_tokens=1500,
        )
        review_data = json.loads(review)
        if isinstance(review_data, dict):
            result["review_decisions"] = review_data.get("sentences")
        decisions = rewrite.validate_review(review, len(sentences))
        result.update(text=rewrite.render(proposal, evidence, facts, archetype),
                      rendering="llm", semantic_review="accepted", review_decisions=decisions,
                      commentary_status="llm_approved")
    except Exception as exc:
        logger.warning("Grounded rewrite rejected at ply %s: %s", facts.ply, type(exc).__name__)
        result.update(contract_ok=False,
                      semantic_review="rejected_or_unavailable" if stage == "review"
                      else "not_run" if review_enabled else "disabled",
                      fallback_reason=str(exc)[:240],
                      commentary_status={"validation": "template_validation_failed",
                                         "provider": "template_provider_failed",
                                         "review": "template_review_rejected"}[stage])
    return result
