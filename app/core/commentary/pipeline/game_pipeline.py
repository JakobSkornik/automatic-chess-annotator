"""Full-game annotation LLM phases: digest, per-move commentary, episode + game narratives."""

from __future__ import annotations

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


def compute_turning_points(
    move_events: List[Any],
    *,
    min_swing_cp: int = 150,
    max_points: int = 3,
) -> List[Dict[str, Any]]:
    """Heuristic turning points from the eval series (replaces the LLM game digest).

    A turning point is a large White-POV eval swing, ranked higher when the
    advantage flips sign or a balanced position becomes decisive.
    """
    candidates: List[tuple] = []
    for me in move_events:
        if me.eval_swing_cp is None or me.eval_before_cp is None or me.eval_after_cp is None:
            continue
        swing = abs(int(me.eval_swing_cp))
        if swing < min_swing_cp:
            continue
        before = int(me.eval_before_cp)
        after = int(me.eval_after_cp)
        sign_flip = (before >= 50 and after <= -50) or (before <= -50 and after >= 50)
        becomes_decisive = abs(before) < 100 <= abs(after)
        rank = swing + (200 if sign_flip else 0) + (100 if becomes_decisive else 0)
        candidates.append((rank, me))
    candidates.sort(key=lambda t: -t[0])
    picked = [
        {
            "ply": me.ply,
            "san": me.san,
            "swing_cp": int(me.eval_swing_cp),
            "eval_before_cp": int(me.eval_before_cp),
            "eval_after_cp": int(me.eval_after_cp),
        }
        for _rank, me in candidates[:max_points]
    ]
    return sorted(picked, key=lambda d: d["ply"])


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
        commentary_level: Optional[str] = None,
        comment_side: Optional[str] = None,
    ) -> None:
        eff = llm_effort or os.environ.get("LLM_DEFAULT_EFFORT", "medium")
        level = (commentary_level or "intermediate").strip().lower()
        if level not in ("beginner", "intermediate", "expert"):
            level = "intermediate"
        side = (comment_side or "both").strip().lower()
        if side not in ("white", "black", "both"):
            side = "both"
        state.metadata = state.metadata.model_copy(
            update={"commentary_level": level, "comment_side": side}
        )
        analyzed_rows = state.analyzed_rows
        move_events = state.move_events
        episodes = state.episodes
        context = state.context
        ply_to_episode = state.ply_to_episode

        gid_token = llm_call_log.set_game_context(state.metadata.id)
        try:
            # Turning points are computed heuristically from the eval series;
            # the whole-game LLM digest (and its game_summary output) is gone.
            turning_points = compute_turning_points(move_events)
            context.game_digest = {"turning_points": turning_points}

            ply_to_mi = {me.ply: mi for mi, me in enumerate(move_events)}
            for tp in turning_points:
                p = int(tp["ply"])
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

            # --- Reverse-order generation -------------------------------------
            # Comments are written from the LAST move backwards, so each prompt
            # can see what happens later in the game (foreshadowing). Episodes
            # are processed last-to-first; an episode's narrative is generated
            # right after its moves so earlier episodes can reference it.
            result_str = str(state.metadata.result or "*")
            winner_note = {
                "1-0": "White went on to win",
                "0-1": "Black went on to win",
                "1/2-1/2": "the game ended in a draw",
            }.get(result_str, "the result was unrecorded")
            future_comments: List[Dict[str, Any]] = []  # nearest-later first
            future_episode_notes: List[str] = []

            def _future_context_text() -> str:
                parts = [f"Game outcome: {result_str} ({winner_note})."]
                for fc in future_comments[:3]:
                    parts.append(
                        f"Later, at move {(fc['ply'] + 1) // 2} ({fc['san']}): {fc['text'][:160]}"
                    )
                parts.extend(future_episode_notes[:2])
                return "\n".join(parts)[:1200]

            def _mover_matches_side(ply: int) -> bool:
                if side == "both":
                    return True
                is_white_move = ply % 2 == 1
                return (side == "white") == is_white_move

            episodes_desc = sorted(episodes, key=lambda e: -e.episode_index)
            commented_mis = [
                mi for mi, me in enumerate(move_events)
                if (me.key_moment_type or me.teaching_moment) and _mover_matches_side(me.ply)
            ]
            mis_by_episode: Dict[Optional[int], List[int]] = {}
            for mi in commented_mis:
                ep_key = ply_to_episode.get(move_events[mi].ply)
                mis_by_episode.setdefault(ep_key, []).append(mi)

            ordered_units: List[tuple] = []  # ("move", mi, episode) / ("episode", ep)
            for ep in episodes_desc:
                for mi in sorted(
                    mis_by_episode.get(ep.episode_index, []),
                    key=lambda i: -move_events[i].ply,
                ):
                    ordered_units.append(("move", mi, ep))
                ordered_units.append(("episode", ep, None))
            for mi in sorted(
                mis_by_episode.get(None, []), key=lambda i: -move_events[i].ply
            ):
                ordered_units.append(("move", mi, None))

            episode_model = resolve_model(advanced_commenter.provider_name, "episode")

            for unit_kind, unit_a, unit_b in ordered_units:
                if unit_kind == "episode":
                    ep_obj: Episode = unit_a
                    ep_ctx = llm_call_log.set_move_context(
                        episode_index=ep_obj.episode_index, pass_label="episode"
                    )
                    try:
                        try:
                            ep_obj.narrative_summary = (
                                await advanced_commenter.generate_episode_commentary(
                                    ep_obj, model=episode_model, effort=eff
                                )
                            )
                        except Exception as e:
                            logger.error("Episode commentary failed: %s", e)
                            ep_obj.narrative_summary = ep_obj.narrative_summary or None
                        if ep_obj.narrative_summary:
                            future_episode_notes.insert(
                                0,
                                f"Then ('{ep_obj.title}'): {ep_obj.narrative_summary[:140]}",
                            )
                            if commentary_callback:
                                await commentary_callback(
                                    "EPISODE_NARRATIVE",
                                    {
                                        "episode_index": ep_obj.episode_index,
                                        "title": ep_obj.title,
                                        "narrative": ep_obj.narrative_summary,
                                    },
                                )
                    finally:
                        llm_call_log.reset_move_context(ep_ctx)
                    continue

                mi = unit_a
                me = move_events[mi]
                ep = unit_b
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
                    has_facts = me.comment_facts is not None
                    pv_san_bm25: List[str] = []
                    if row and not has_facts:
                        # Legacy path only: facts moves carry their own lines.
                        depth_bm25 = int(os.environ.get("RAG_BM25_STOCKFISH_DEPTH", "14"))
                        pv_san_bm25 = _bm25_pv_san_from_fen_after(
                            state.retriever.engine_connector, row.fen_after, depth_bm25
                        )
                    elif row and has_facts and me.comment_facts.display_line:
                        pv_san_bm25 = list(me.comment_facts.display_line.line_san[:5])
                    future_delta = None
                    if row and not has_facts and me.best_move_uci and me.uci != me.best_move_uci:
                        # Legacy path only: facts.better_alternative supersedes this
                        # (and saves a depth-18 search per key moment).
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
                            future_context=_future_context_text(),
                            commentary_level=level,
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
                                        if mctx.level_texts:
                                            _slot["comments"] = dict(mctx.level_texts)
                                        if llm_debug.get("facts_renderings"):
                                            _slot["facts_renderings"] = llm_debug["facts_renderings"]
                                        if "facts_contract_ok" in llm_debug:
                                            _slot["facts_contract_ok"] = llm_debug["facts_contract_ok"]
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
                            # Reverse order: this comment is "the future" for
                            # every move still to be commented.
                            future_comments.insert(
                                0, {"ply": me.ply, "san": me.san, "text": text}
                            )
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

            # Episode narratives were generated inline during the backward sweep;
            # the whole-game narrative stage is gone (the Summary panel was cut).
            state.llm_done = True
            if commentary_callback:
                await commentary_callback("COMMENTARY_COMPLETE", {})

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
