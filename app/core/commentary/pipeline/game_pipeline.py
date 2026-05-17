"""Full-game annotation LLM phases: digest, per-move commentary, episode + game narratives."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.core.commentary import llm_call_log
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.annotation_tokens import resolve_tokens_for_comment
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.pipeline.move_pipeline import MoveCommentaryContext, MoveCommentaryPipeline
from app.core.commentary.rag_retriever import rag_results_to_ws_refs
from app.core.commentary.features.future_line_compare import compare_played_vs_best_future_lines
from app.core.commentary.features.move_category import classify_move_event
from app.core.engine.analysis_retriever import (
    EnginePipelineState,
    _apply_back_to_back_key_moment_suppression,
    _bm25_pv_san_from_fen_after,
    _mark_teaching_moments_per_episode,
    _pv_line_for_ai_payload,
)
from app.models.chess_events import Episode

logger = logging.getLogger(__name__)


class GameAnnotationPipeline:
    """Single entrypoint for post-engine LLM commentary (mutates ``EnginePipelineState``)."""

    async def run_llm_phases(
        self,
        state: EnginePipelineState,
        advanced_commenter: AdvancedCommentService,
        *,
        progress_callback: Optional[Callable[[float, str], Awaitable[None]]] = None,
        commentary_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
        llm_effort: Optional[str] = None,
    ) -> None:
        eff = llm_effort or os.environ.get("LLM_DEFAULT_EFFORT", "medium")
        analyzed_rows = state.analyzed_rows
        move_events = state.move_events
        episodes = state.episodes
        context = state.context
        ply_to_episode = state.ply_to_episode
        digest_model = resolve_model(advanced_commenter.provider_name, "digest")

        gid_token = llm_call_log.set_game_context(state.metadata.id)
        try:
            digest_ctx = llm_call_log.set_move_context(pass_label="digest")
            try:
                try:
                    game_digest = await advanced_commenter.generate_game_digest(
                        context, model=digest_model, effort=eff
                    )
                except Exception as e:
                    logger.warning("game digest failed: %s", e)
                    game_digest = {}
            finally:
                llm_call_log.reset_move_context(digest_ctx)
            context.game_digest = game_digest
            if commentary_callback and game_digest:
                await commentary_callback("GAME_SUMMARY", {"digest": game_digest})

            arch_raw = game_digest.get("strategic_archetype") if isinstance(game_digest, dict) else None
            if isinstance(arch_raw, str) and arch_raw.strip():
                a = arch_raw.strip()
                state.metadata = state.metadata.model_copy(update={"strategic_archetype": a})
                md = dict(context.metadata or {})
                md["strategic_archetype"] = a
                context.metadata = md

            tps = (game_digest or {}).get("turning_points") or []
            ply_to_mi = {me.ply: mi for mi, me in enumerate(move_events)}
            for tp in tps:
                if not isinstance(tp, dict):
                    continue
                try:
                    p = int(tp.get("ply", 0))
                except (TypeError, ValueError):
                    continue
                mi = ply_to_mi.get(p)
                if mi is None:
                    continue
                me = move_events[mi]
                if me.key_moment_type:
                    continue
                promoted = me.model_copy(update={"key_moment_type": "critical_decision"})
                move_events[mi] = promoted
                context.move_events[mi] = promoted
                for ep in episodes:
                    for ej, ev in enumerate(ep.move_events):
                        if ev.ply == promoted.ply:
                            ep.move_events[ej] = promoted
                            break

            _apply_back_to_back_key_moment_suppression(move_events, episodes, context)
            _mark_teaching_moments_per_episode(
                move_events, episodes, context, state.metadata.result
            )

            audit_coverage: List[float] = []
            audit_evals: List[int] = []
            audit_forbidden = 0
            rag_usage_num = 0
            rag_usage_den = 0

            key_moment_list = [
                me for me in move_events if me.key_moment_type or me.teaching_moment
            ]
            n_key_moments = max(len(key_moment_list), 1)
            key_moment_idx = 0

            for mi, me in enumerate(move_events):
                if not (me.key_moment_type or me.teaching_moment):
                    continue
                ep = next((e for e in episodes if e.episode_index == ply_to_episode.get(me.ply)), None)
                composer_pass_label = "key_moment" if me.key_moment_type else "teaching"
                composer_model = resolve_model(
                    advanced_commenter.provider_name, "composer", pass_label=composer_pass_label
                )
                move_ctx = llm_call_log.set_move_context(
                    ply=me.ply,
                    episode_index=ply_to_episode.get(me.ply),
                    pass_label="key_moment" if me.key_moment_type else "teaching",
                )
                try:
                    pct = 95.0 + (key_moment_idx / n_key_moments) * 3.0
                    key_moment_idx += 1
                    if progress_callback:
                        label = (
                            "teaching" if me.teaching_moment and not me.key_moment_type else "key moment"
                        )
                        await progress_callback(pct, f"LLM: {label} {me.san} (ply {me.ply})")
                    row = analyzed_rows[me.move_index] if 0 <= me.move_index < len(analyzed_rows) else None
                    depth_bm25 = int(os.environ.get("RAG_BM25_STOCKFISH_DEPTH", "14"))
                    pv_san_bm25: List[str] = []
                    if row:
                        pv_san_bm25 = _bm25_pv_san_from_fen_after(
                            state.retriever.engine_connector, row.fen_after, depth_bm25
                        )
                    future_delta = None
                    if row and me.best_move_uci and me.uci != me.best_move_uci:
                        depth_fl = int(os.environ.get("FUTURE_LINE_DEPTH", "18"))
                        n_plies = int(os.environ.get("FUTURE_LINE_PLIES", "6"))
                        try:
                            future_delta = compare_played_vs_best_future_lines(
                                state.retriever.engine_connector,
                                me.fen_before,
                                row.fen_after,
                                me.best_move_uci,
                                me.uci,
                                depth=depth_fl,
                                n_plies=n_plies,
                            )
                        except Exception as e:
                            logger.warning("future_line_compare failed ply %s: %s", me.ply, e)
                    me_for_rag = me.model_copy(
                        update={
                            "pv_san": pv_san_bm25,
                            "future_line": future_delta,
                            "move_category": classify_move_event(
                                me.model_copy(update={"future_line": future_delta}),
                                future_delta,
                            ),
                        }
                    )
                    move_events[mi] = me_for_rag
                    context.move_events[mi] = me_for_rag
                    for epi in episodes:
                        for ej, ev in enumerate(epi.move_events):
                            if ev.ply == me_for_rag.ply:
                                epi.move_events[ej] = me_for_rag
                                break
                    move_id = me.move_index + 1
                    if commentary_callback:
                        await commentary_callback(
                            "AI_GENERATION_STATUS",
                            {
                                "moveId": move_id,
                                "context": "mainline",
                                "status": "start",
                                "startedAt": time.time(),
                                "model": composer_model,
                                "effort": eff,
                            },
                        )
                    effort_move = "low" if me.teaching_moment and not me.key_moment_type else eff
                    try:
                        mctx = MoveCommentaryContext(
                            move_event=me_for_rag,
                            episode=ep,
                            game_context=context,
                            analyzed_row=row,
                            service=advanced_commenter,
                            composer_effort=effort_move,
                            key_moment_type=me.key_moment_type,
                            composer_pass_label=composer_pass_label,
                        )
                        mctx = await MoveCommentaryPipeline().run(mctx)
                        text = mctx.final_text
                        rag_results = mctx.rag_results
                        llm_debug = dict(mctx.llm_debug or {})

                        if text:
                            persist_rag = (
                                os.environ.get("RAG_PERSIST_REFS", "1").strip().lower()
                                not in ("0", "false", "")
                            )
                            for r in analyzed_rows:
                                if r.ply == me.ply:
                                    if isinstance(r.analyzed_move.hiddenFeatures, dict):
                                        r.analyzed_move.hiddenFeatures.setdefault("_llm", {})
                                        _slot = r.analyzed_move.hiddenFeatures["_llm"]
                                        _slot["comment"] = text
                                        _slot["named_motifs"] = llm_debug.get("composer_named_motifs", [])
                                        rat = llm_debug.get("rationale") or {}
                                        _slot["primary_motif_label"] = rat.get("primary_motif_label", "")
                                        if persist_rag:
                                            _slot["rag_refs"] = rag_results_to_ws_refs(rag_results or [])
                                        _slot["composer_rag_applied"] = llm_debug.get("composer_rag_applied")
                                    break
                            arch_snip = str((context.game_digest or {}).get("strategic_archetype") or "")
                            motif_hint = ", ".join(llm_debug.get("composer_named_motifs") or []) or (
                                "heuristic-fallback"
                                if llm_debug.get("fallback_used") == "heuristic"
                                else "—"
                            )
                            hints = list(context.prior_context_snippets or [])
                            hints.append(
                                f"After {me.san} (ply {me.ply}): {motif_hint}; archetype={arch_snip}"
                            )
                            context.prior_context_snippets = hints[-6:]
                        if llm_debug:
                            ca = llm_debug.get("commentary_audit")
                            if isinstance(ca, dict):
                                try:
                                    audit_coverage.append(float(ca.get("motif_coverage_ratio", 0)))
                                    audit_evals.append(int(ca.get("eval_token_count", 0)))
                                    audit_forbidden += int(ca.get("forbidden_phrase_hits", 0))
                                except (TypeError, ValueError):
                                    pass
                                if ca.get("rag_had_hits"):
                                    rag_usage_den += 1
                                    if ca.get("rag_applied"):
                                        rag_usage_num += 1
                        if commentary_callback and text:
                            resolved_tokens = resolve_tokens_for_comment(text, me.fen_before, me.fen_after)
                            await commentary_callback(
                                "AI_COMMENT_UPDATE",
                                {
                                    "moveId": move_id,
                                    "context": "mainline",
                                    "data": {
                                        "summary": text,
                                        "commentary": text,
                                        "pv_line": _pv_line_for_ai_payload(row),
                                        "resolved_tokens": resolved_tokens,
                                        "rag_refs": rag_results_to_ws_refs(rag_results),
                                        "llm_debug": llm_debug,
                                    },
                                },
                            )
                    except Exception as e:
                        logger.error(f"LLM move commentary failed at ply {me.ply}: {e}")
                    finally:
                        if commentary_callback:
                            await commentary_callback(
                                "AI_GENERATION_STATUS",
                                {
                                    "moveId": move_id,
                                    "context": "mainline",
                                    "status": "end",
                                    "endedAt": time.time(),
                                    "model": composer_model,
                                    "effort": eff,
                                },
                            )
                finally:
                    llm_call_log.reset_move_context(move_ctx)

            context.critical_moments = [e for e in move_events if e.is_critical]

            if progress_callback:
                await progress_callback(98.5, "LLM: episode narratives...")

            episode_model = resolve_model(advanced_commenter.provider_name, "episode")
            narrative_model = resolve_model(advanced_commenter.provider_name, "narrative")

            async def _episode_narrative(ep: Episode) -> None:
                ep_ctx = llm_call_log.set_move_context(
                    episode_index=ep.episode_index, pass_label="episode"
                )
                try:
                    try:
                        ep.narrative_summary = await advanced_commenter.generate_episode_commentary(
                            ep, model=episode_model, effort=eff
                        )
                        if commentary_callback and ep.narrative_summary:
                            await commentary_callback(
                                "EPISODE_NARRATIVE",
                                {
                                    "episode_index": ep.episode_index,
                                    "title": ep.title,
                                    "narrative": ep.narrative_summary,
                                },
                            )
                    except Exception as e:
                        logger.error("Episode commentary failed: %s", e)
                finally:
                    llm_call_log.reset_move_context(ep_ctx)

            await asyncio.gather(*[_episode_narrative(ep) for ep in episodes])

            narr_ctx = llm_call_log.set_move_context(pass_label="narrative")
            try:
                try:
                    context.game_narrative = await advanced_commenter.generate_game_narrative(
                        context, model=narrative_model, effort=eff
                    )
                    if commentary_callback and context.game_narrative:
                        await commentary_callback(
                            "GAME_NARRATIVE",
                            {"narrative": context.game_narrative},
                        )
                except Exception as e:
                    logger.error(f"Game narrative failed: {e}")
            finally:
                llm_call_log.reset_move_context(narr_ctx)

            if audit_coverage:
                logger.info(
                    "game %s commentary_audit avg_motif_coverage=%.3f avg_eval_tokens=%.3f "
                    "forbidden_hits=%d n_llm_moves=%d",
                    state.metadata.id,
                    sum(audit_coverage) / len(audit_coverage),
                    sum(audit_evals) / max(len(audit_evals), 1),
                    audit_forbidden,
                    len(audit_coverage),
                )
            if rag_usage_den > 0:
                logger.info(
                    "game %s rag_quality moves_with_rag_hits=%d rag_applied=%d ratio=%.3f",
                    state.metadata.id,
                    rag_usage_den,
                    rag_usage_num,
                    rag_usage_num / rag_usage_den,
                )

            if progress_callback:
                await progress_callback(99.0, "Commentary complete.")
        finally:
            llm_call_log.reset_game_context(gid_token)
