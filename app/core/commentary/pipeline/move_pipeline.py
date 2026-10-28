"""Per-move commentary as explicit ordered stages over a shared context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from app.core.commentary.llm_policy import resolve_model
from app.models.chess_events import (
    AnalyzedMoveData,
    GameAnalysisContext,
    MoveEvent,
)
from app.core.commentary.rendering_contracts import WHOLE_ANNOTATION_CONTRACTS

if TYPE_CHECKING:
    from app.core.commentary.advanced_comment_service import AdvancedCommentService

logger = logging.getLogger(__name__)


@dataclass
class MoveCommentaryContext:
    move_event: MoveEvent
    game_context: GameAnalysisContext
    analyzed_row: AnalyzedMoveData | None
    service: AdvancedCommentService
    composer_effort: str
    key_moment_type: str | None
    skip: bool = False
    llm_debug: dict[str, Any] = field(default_factory=dict)
    final_text: str = ""
    fallback_used: str | None = None
    composer_pass_label: str | None = None
    # Audience level the comment is generated at (job parameter)
    commentary_level: str = "intermediate"


class MoveStage(Protocol):
    name: str

    async def run(self, ctx: MoveCommentaryContext) -> None: ...


class KeyMomentGateStage:
    name = "key_moment_gate"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if not ctx.move_event.key_moment_type:
            ctx.skip = True
            logger.debug(
                "KeyMomentGateStage: ply %s (%s) is not a key moment — no "
                "comment will be generated for this move",
                ctx.move_event.ply,
                ctx.move_event.san,
            )


class FactsComposeStage:
    """Guid path: render CommentFacts (template or fact-contract LLM call)."""

    name = "facts_compose"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        facts = ctx.move_event.comment_facts
        if facts is None:
            return
        from app.core.commentary.forbidden_phrases import scrub_forbidden, forbidden_hit_strings
        from app.core.commentary.phases.composer import compose_facts_comment
        from app.core.commentary.phases.composer.framing import alt_gap_cp, ALT_MATERIAL_GAP_CP

        pl = ctx.composer_pass_label or (
            "key_moment" if ctx.move_event.key_moment_type else "teaching"
        )
        model = resolve_model(ctx.service.provider_name, "composer", pass_label=pl)
        result = await compose_facts_comment(
            ctx.service,
            facts,
            model=model,
            effort=ctx.composer_effort,
            level=ctx.commentary_level,
            key_moment_type=ctx.move_event.key_moment_type,
        )
        text = str(result.get("text") or "")
        if result.get("rendering_contract") in WHOLE_ANNOTATION_CONTRACTS and result.get("semantic_review") in {"accepted", "disabled"}:
            # Do not alter screened prose (including spaces before ...SAN).
            # Historical phrase flags remain observable, without deleting words.
            forbidden_hits = forbidden_hit_strings(text)
        else:
            text, forbidden_hits = scrub_forbidden(text)
        ctx.final_text = text
        record_annotation = getattr(ctx.service, "record_annotation", None)
        if record_annotation is not None:
            await record_annotation({
                "ply": ctx.move_event.ply,
                "san": ctx.move_event.san,
                "uci": ctx.move_event.uci,
                "fen_after": ctx.move_event.fen_after,
                "text": text,
                "rendering": result.get("rendering"),
                "contract_ok": result.get("contract_ok"),
            })
        ctx.llm_debug.update(
            {
                "facts_renderings": {ctx.commentary_level: result.get("rendering")},
                "facts_contract_ok": result.get("contract_ok"),
                "rendering_contract": result.get("rendering_contract", "claim_selection_v1"),
                "prompt_version": result.get("prompt_version"),
                "semantic_review": result.get("semantic_review", "not_applicable"),
                "review_scope": result.get("review_scope"),
                "selected_claims": result.get("selected_claims"),
                "raw_proposal": result.get("raw_proposal"),
                "claim_catalog": result.get("claim_catalog"),
                "review_decisions": result.get("review_decisions"),
                "fallback_reason": result.get("fallback_reason"),
                "commentary_reason": result.get("commentary_reason"),
                "commentary_status": result.get("commentary_status",
                    "llm_unreviewed" if result.get("rendering") == "llm" else "template"),
                "selection_metadata": facts.selection_metadata,
                "missed_improvement": {
                    "gap_cp": alt_gap_cp(facts),
                    "threshold_cp": ALT_MATERIAL_GAP_CP,
                    "detected": bool(
                        facts.better_alternative and not facts.better_alternative.is_inferior
                        and alt_gap_cp(facts) is not None
                        and alt_gap_cp(facts) >= ALT_MATERIAL_GAP_CP
                    ),
                    "scope": "renderer_comparison_not_primary_key_moment_class",
                },
                "forbidden_phrase_hits": len(forbidden_hits),
                "claims": [c.text for c in facts.claims],
                "feature_refs": facts.feature_refs(),
                "conversation_mode": "game_session" if getattr(
                    ctx.service, "is_commentary_session", False
                ) else "independent",
            }
        )


class FallbackStage:
    name = "fallback"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        if (ctx.final_text or "").strip():
            return
        ctx.llm_debug["composer_empty"] = True
        ctx.final_text = "Commentary temporarily unavailable."
        ctx.fallback_used = "unavailable"
        ctx.llm_debug["fallback_used"] = "unavailable"
        logger.debug(
            "FallbackStage: ply %s (%s, key_moment=%s) reached the fallback "
            "stage with empty composer output — comment replaced with the "
            "'unavailable' placeholder",
            ctx.move_event.ply,
            ctx.move_event.san,
            ctx.move_event.key_moment_type,
        )


class FinalizeStage:
    """No-op: row / WS assembly stays in game pipeline; debug is already on ctx."""

    name = "finalize"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        return


class MoveCommentaryPipeline:
    def __init__(self) -> None:
        self.stages: list[MoveStage] = cast(
            list[MoveStage],
            [
                KeyMomentGateStage(),
                FactsComposeStage(),
                FallbackStage(),
                FinalizeStage(),
            ],
        )

    async def run(self, ctx: MoveCommentaryContext) -> MoveCommentaryContext:
        for st in self.stages:
            await st.run(ctx)
        return ctx
