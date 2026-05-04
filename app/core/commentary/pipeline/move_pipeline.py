"""Per-move commentary as explicit ordered stages over a shared context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, cast

from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.rag_retriever import RAGResult, build_rag_query
from app.models.chess_events import AnalyzedMoveData, Episode, GameAnalysisContext, MoveEvent, MoveRationale

if TYPE_CHECKING:
    from app.core.commentary.advanced_comment_service import AdvancedCommentService

logger = logging.getLogger(__name__)


@dataclass
class MoveCommentaryContext:
    move_event: MoveEvent
    episode: Optional[Episode]
    game_context: GameAnalysisContext
    analyzed_row: Optional[AnalyzedMoveData]
    service: AdvancedCommentService
    composer_effort: str
    key_moment_type: Optional[str]
    skip: bool = False
    rag_results: List[RAGResult] = field(default_factory=list)
    rationale: Optional[MoveRationale] = None
    user_prompt: Optional[str] = None
    llm_debug: Dict[str, Any] = field(default_factory=dict)
    final_text: str = ""
    fallback_used: Optional[str] = None


class MoveStage(Protocol):
    name: str

    async def run(self, ctx: MoveCommentaryContext) -> None: ...


class KeyMomentGateStage:
    name = "key_moment_gate"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if not (ctx.move_event.key_moment_type or ctx.move_event.teaching_moment):
            ctx.skip = True


class RagRetrievalStage:
    name = "rag_retrieval"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        from app.core.commentary.advanced_comment_service import (
            compute_rag_top_k,
            _detail_level_for_key_moment,
        )

        query = build_rag_query(ctx.move_event, ctx.episode)
        detail_pre = _detail_level_for_key_moment(ctx.move_event.key_moment_type)
        rag_top_k = compute_rag_top_k(detail_pre, query.phase)
        ctx.rag_results = await ctx.service._rag.retrieve(query, top_k=rag_top_k)
        logger.info(
            "RAG query: phase=%s fen=%.80s",
            query.phase,
            (query.fen or "")[:80],
        )


class RationaleStage:
    name = "rationale"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        from app.core.commentary.move_rationale import build_rationale

        ctx.rationale = build_rationale(ctx.move_event, ctx.move_event.future_line)


class PromptBuildStage:
    name = "prompt_build"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        user_prompt, rag_results, dbg = await ctx.service.build_event_llm_input(
            ctx.move_event,
            ctx.episode,
            ctx.game_context,
            analyzed_row=ctx.analyzed_row,
            composer_effort=ctx.composer_effort,
            rag_results=ctx.rag_results,
            rationale_override=ctx.rationale,
        )
        ctx.user_prompt = user_prompt
        ctx.rag_results = rag_results
        ctx.llm_debug = dbg


class LlmCallStage:
    name = "llm_call"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        model = resolve_model(ctx.service.provider_name, "composer")
        ctx.final_text = await ctx.service.analyze_and_compose_raw_text(
            ctx.user_prompt or "",
            model=model,
            effort=ctx.composer_effort,
            key_moment_type=ctx.key_moment_type or ctx.move_event.key_moment_type,
            move_category=ctx.move_event.move_category.value if ctx.move_event.move_category else None,
            llm_debug=ctx.llm_debug,
            fen_before=ctx.move_event.fen_before,
            fen_after=ctx.move_event.fen_after,
        )


class FallbackStage:
    name = "fallback"

    async def run(self, ctx: MoveCommentaryContext) -> None:
        if ctx.skip:
            return
        if (ctx.final_text or "").strip():
            return
        from app.core.engine.analysis_retriever import _heuristic_llm_fallback_comment

        ctx.llm_debug["composer_empty"] = True
        fb = _heuristic_llm_fallback_comment(ctx.move_event)
        if fb:
            ctx.final_text = fb
            ctx.fallback_used = "heuristic"
            ctx.llm_debug["fallback_used"] = "heuristic"
        else:
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
        self.stages: List[MoveStage] = cast(
            List[MoveStage],
            [
                KeyMomentGateStage(),
                RagRetrievalStage(),
                RationaleStage(),
                PromptBuildStage(),
                LlmCallStage(),
                FallbackStage(),
                FinalizeStage(),
            ],
        )

    async def run(self, ctx: MoveCommentaryContext) -> MoveCommentaryContext:
        for st in self.stages:
            await st.run(ctx)
        return ctx
