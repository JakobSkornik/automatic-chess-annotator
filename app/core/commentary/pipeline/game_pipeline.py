"""Full-game annotation LLM phase: heuristic turning points + per-move commentary."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.commentary import llm_call_log
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.annotation_tokens import resolve_tokens_for_comment
from app.core.commentary.features.move_category import classify_move_event
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.pipeline.move_pipeline import (
    MoveCommentaryContext,
    MoveCommentaryPipeline,
)
from app.core.engine.analysis_retriever import (
    EnginePipelineState,
    _apply_back_to_back_key_moment_suppression,
    _pv_line_for_ai_payload,
)

logger = logging.getLogger(__name__)


def compute_turning_points(
    move_events: list[Any],
    *,
    min_swing_cp: int = 150,
    max_points: int = 3,
) -> list[dict[str, Any]]:
    """Heuristic turning points from the eval series (replaces the LLM game digest).

    A turning point is a large White-POV eval swing, ranked higher when the
    advantage flips sign or a balanced position becomes decisive.
    """
    candidates: list[tuple] = []
    for me in move_events:
        if (
            me.eval_swing_cp is None
            or me.eval_before_cp is None
            or me.eval_after_cp is None
        ):
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
        progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
        commentary_callback: Callable[[str, dict[str, Any]], Awaitable[None]]
        | None = None,
        llm_effort: str | None = None,
        commentary_level: str | None = None,
        comment_side: str | None = None,
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
        context = state.context

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
                promoted = me.model_copy(
                    update={"key_moment_type": "critical_decision"}
                )
                move_events[mi] = promoted
                context.move_events[mi] = promoted

            _apply_back_to_back_key_moment_suppression(move_events, context)

            audit_coverage: list[float] = []
            audit_evals: list[int] = []
            audit_forbidden = 0

            key_moment_list = [me for me in move_events if me.key_moment_type]
            n_key_moments = max(len(key_moment_list), 1)
            key_moment_idx = 0

            # --- Reverse-order generation -------------------------------------
            # Comments are written from the LAST move backwards, so each prompt
            # can see what happens later in the game (foreshadowing).
            result_str = str(state.metadata.result or "*")
            winner_note = {
                "1-0": "White went on to win",
                "0-1": "Black went on to win",
                "1/2-1/2": "the game ended in a draw",
            }.get(result_str, "the result was unrecorded")
            future_comments: list[dict[str, Any]] = []  # nearest-later first

            def _future_context_text() -> str:
                parts = [f"Game outcome: {result_str} ({winner_note})."]
                for fc in future_comments[:3]:
                    parts.append(
                        f"Later, at move {(fc['ply'] + 1) // 2} ({fc['san']}): {fc['text'][:160]}"
                    )
                return "\n".join(parts)[:1200]

            def _mover_matches_side(ply: int) -> bool:
                if side == "both":
                    return True
                is_white_move = ply % 2 == 1
                return (side == "white") == is_white_move

            ordered_mis = sorted(
                [
                    mi
                    for mi, me in enumerate(move_events)
                    if me.key_moment_type and _mover_matches_side(me.ply)
                ],
                key=lambda i: -move_events[i].ply,
            )

            for mi in ordered_mis:
                me = move_events[mi]
                composer_model = resolve_model(
                    advanced_commenter.provider_name,
                    "composer",
                    pass_label="key_moment",
                )
                move_ctx = llm_call_log.set_move_context(
                    ply=me.ply,
                    pass_label="key_moment",
                )
                try:
                    pct = 95.0 + (key_moment_idx / n_key_moments) * 3.0
                    key_moment_idx += 1
                    if progress_callback:
                        await progress_callback(
                            pct, f"LLM: key moment {me.san} (ply {me.ply})"
                        )
                    row = (
                        analyzed_rows[me.move_index]
                        if 0 <= me.move_index < len(analyzed_rows)
                        else None
                    )
                    me_classified = me.model_copy(
                        update={"move_category": classify_move_event(me)}
                    )
                    move_events[mi] = me_classified
                    context.move_events[mi] = me_classified
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
                    try:
                        mctx = MoveCommentaryContext(
                            move_event=me_classified,
                            game_context=context,
                            analyzed_row=row,
                            service=advanced_commenter,
                            composer_effort=eff,
                            key_moment_type=me.key_moment_type,
                            composer_pass_label="key_moment",
                            future_context=_future_context_text(),
                            commentary_level=level,
                        )
                        mctx = await MoveCommentaryPipeline().run(mctx)
                        text = mctx.final_text
                        llm_debug = dict(mctx.llm_debug or {})

                        if text:
                            for r in analyzed_rows:
                                if r.ply == me.ply:
                                    if isinstance(r.analyzed_move.hiddenFeatures, dict):
                                        r.analyzed_move.hiddenFeatures.setdefault(
                                            "_llm", {}
                                        )
                                        _slot = r.analyzed_move.hiddenFeatures["_llm"]
                                        _slot["comment"] = text
                                        if mctx.level_texts:
                                            _slot["comments"] = dict(mctx.level_texts)
                                        if llm_debug.get("facts_renderings"):
                                            _slot["facts_renderings"] = llm_debug[
                                                "facts_renderings"
                                            ]
                                        if "facts_contract_ok" in llm_debug:
                                            _slot["facts_contract_ok"] = llm_debug[
                                                "facts_contract_ok"
                                            ]
                                    break
                            arch_snip = str(
                                (context.game_digest or {}).get("strategic_archetype")
                                or ""
                            )
                            motif_hint = ", ".join(
                                llm_debug.get("composer_named_motifs") or []
                            ) or (
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
                                    audit_coverage.append(
                                        float(ca.get("motif_coverage_ratio", 0))
                                    )
                                    audit_evals.append(
                                        int(ca.get("eval_token_count", 0))
                                    )
                                    audit_forbidden += int(
                                        ca.get("forbidden_phrase_hits", 0)
                                    )
                                except (TypeError, ValueError):
                                    pass
                        if commentary_callback and text:
                            resolved_tokens = resolve_tokens_for_comment(
                                text, me.fen_before, me.fen_after
                            )
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

            if progress_callback:
                await progress_callback(99.0, "Commentary complete.")
        finally:
            llm_call_log.reset_game_context(gid_token)
