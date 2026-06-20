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
    # Reverse-order generation: what the game already "knows" about its future
    future_context: str | None = None
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


class FactsComposeStage:
    """Guid path: render CommentFacts (template or fact-contract LLM call)."""

    name = "facts_compose"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        facts = ctx.move_event.comment_facts
        if facts is None:
            return
        from app.core.commentary.forbidden_phrases import scrub_forbidden
        from app.core.commentary.phases.composer import compose_facts_comment

        enrichment: dict[str, Any] = {}
        gc = ctx.game_context
        # Opening scene-setting is only relevant while still in the opening;
        # passing it on every key moment makes the model restate the opening
        # line in every comment (Guid feedback).
        if (gc.opening_name or gc.opening_eco) and ctx.move_event.phase == "opening":
            enrichment["opening"] = (
                f"{gc.opening_name or ''} ({gc.opening_eco or ''})".strip()
            )
        if ctx.future_context:
            enrichment["what_happens_later"] = ctx.future_context

        pl = ctx.composer_pass_label or (
            "key_moment" if ctx.move_event.key_moment_type else "teaching"
        )
        model = resolve_model(ctx.service.provider_name, "composer", pass_label=pl)
        result = await compose_facts_comment(
            ctx.service,
            facts,
            model=model,
            effort=ctx.composer_effort,
            enrichment=enrichment,
            level=ctx.commentary_level,
        )
        text, forbidden_hits = scrub_forbidden(str(result.get("text") or ""))
        ctx.final_text = text
        ctx.llm_debug.update(
            {
                "facts_renderings": {ctx.commentary_level: result.get("rendering")},
                "facts_contract_ok": result.get("contract_ok"),
                "forbidden_phrase_hits": len(forbidden_hits),
                "claims": [c.text for c in facts.claims],
                "feature_refs": facts.feature_refs(),
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
