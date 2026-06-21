"""Full-game annotation LLM phase: heuristic turning points + per-move commentary."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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


# Turning-point ranking.
SIGN_FLIP_CP = 50  # eval crossing +-this counts as an advantage swing
SIGN_FLIP_RANK_BONUS = 200
DECISIVE_RANK_BONUS = 100

# LLM commentary progress milestones (percent).
LLM_PROGRESS_START = 95.0
LLM_PROGRESS_SPAN = 3.0
LLM_DONE_PROGRESS = 99.0


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
    for move_event in move_events:
        if (
            move_event.eval_swing_cp is None
            or move_event.eval_before_cp is None
            or move_event.eval_after_cp is None
        ):
            continue
        swing = abs(int(move_event.eval_swing_cp))
        if swing < min_swing_cp:
            continue
        before = int(move_event.eval_before_cp)
        after = int(move_event.eval_after_cp)
        sign_flip = (before >= SIGN_FLIP_CP and after <= -SIGN_FLIP_CP) or (
            before <= -SIGN_FLIP_CP and after >= SIGN_FLIP_CP
        )
        becomes_decisive = abs(before) < 100 <= abs(after)
        rank = (
            swing
            + (SIGN_FLIP_RANK_BONUS if sign_flip else 0)
            + (DECISIVE_RANK_BONUS if becomes_decisive else 0)
        )
        candidates.append((rank, move_event))
    candidates.sort(key=lambda t: -t[0])
    picked = [
        {
            "ply": move_event.ply,
            "san": move_event.san,
            "swing_cp": int(move_event.eval_swing_cp),
            "eval_before_cp": int(move_event.eval_before_cp),
            "eval_after_cp": int(move_event.eval_after_cp),
        }
        for _rank, move_event in candidates[:max_points]
    ]
    return sorted(picked, key=lambda d: d["ply"])


def _resolve_commentary_params(
    llm_effort: str | None, commentary_level: str | None, comment_side: str | None
) -> tuple[str, str, str]:
    eff = llm_effort or os.environ.get("LLM_DEFAULT_EFFORT", "medium")
    level = (commentary_level or "intermediate").strip().lower()
    if level not in ("beginner", "intermediate", "expert"):
        level = "intermediate"
    side = (comment_side or "both").strip().lower()
    if side not in ("white", "black", "both"):
        side = "both"
    return eff, level, side


def _game_outcome(result: str | None) -> tuple[str, str]:
    result_str = str(result or "*")
    winner_note = {
        "1-0": "White went on to win",
        "0-1": "Black went on to win",
        "1/2-1/2": "the game ended in a draw",
    }.get(result_str, "the result was unrecorded")
    return result_str, winner_note


def _promote_turning_points(move_events: list[Any], context: Any) -> None:
    """Mark heuristic turning points as critical_decision key moments."""
    turning_points = compute_turning_points(move_events)
    context.game_digest = {"turning_points": turning_points}
    ply_to_mi = {move_event.ply: mi for mi, move_event in enumerate(move_events)}
    for tp in turning_points:
        mi = ply_to_mi.get(int(tp["ply"]))
        if mi is None or move_events[mi].key_moment_type:
            continue
        promoted = move_events[mi].model_copy(
            update={"key_moment_type": "critical_decision"}
        )
        move_events[mi] = promoted
        context.move_events[mi] = promoted


def _key_moment_order(move_events: list[Any], side: str) -> list[int]:
    """Indices of side-relevant key moments, latest ply first (reverse-order pass)."""

    def mover_matches(ply: int) -> bool:
        return side == "both" or (side == "white") == (ply % 2 == 1)

    return sorted(
        [
            mi
            for mi, move_event in enumerate(move_events)
            if move_event.key_moment_type and mover_matches(move_event.ply)
        ],
        key=lambda i: -move_events[i].ply,
    )


def _record_audit(run: _LlmRun, llm_debug: dict) -> None:
    """Accumulate the per-move commentary-audit metrics onto the run."""
    ca = llm_debug.get("commentary_audit")
    if not isinstance(ca, dict):
        return
    try:
        run.audit_coverage.append(float(ca.get("motif_coverage_ratio", 0)))
        run.audit_evals.append(int(ca.get("eval_token_count", 0)))
        run.audit_forbidden += int(ca.get("forbidden_phrase_hits", 0))
    except (TypeError, ValueError):
        pass


def _write_llm_comment(
    analyzed_rows: list, ply: int, text: str, llm_debug: dict
) -> None:
    """Store the LLM comment + debug onto the matching row's hiddenFeatures._llm."""
    for r in analyzed_rows:
        if r.ply != ply:
            continue
        hidden_features = r.analyzed_move.hiddenFeatures
        if isinstance(hidden_features, dict):
            slot = hidden_features.setdefault("_llm", {})
            slot["comment"] = text
            if llm_debug.get("facts_renderings"):
                slot["facts_renderings"] = llm_debug["facts_renderings"]
            if "facts_contract_ok" in llm_debug:
                slot["facts_contract_ok"] = llm_debug["facts_contract_ok"]
        return


def _log_audit(run: _LlmRun) -> None:
    if not run.audit_coverage:
        return
    logger.info(
        "game %s commentary_audit avg_motif_coverage=%.3f avg_eval_tokens=%.3f "
        "forbidden_hits=%d n_llm_moves=%d",
        run.state.metadata.id,
        sum(run.audit_coverage) / len(run.audit_coverage),
        sum(run.audit_evals) / max(len(run.audit_evals), 1),
        run.audit_forbidden,
        len(run.audit_coverage),
    )


@dataclass
class _LlmRun:
    """Mutable state shared across the reverse-order commentary sweep."""

    state: EnginePipelineState
    service: AdvancedCommentService
    eff: str
    level: str
    side: str
    progress_callback: Callable[[float, str], Awaitable[None]] | None
    commentary_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None
    result_str: str
    winner_note: str
    future_comments: list[dict[str, Any]] = field(default_factory=list)
    audit_coverage: list[float] = field(default_factory=list)
    audit_evals: list[int] = field(default_factory=list)
    audit_forbidden: int = 0

    def future_context_text(self) -> str:
        parts = [f"Game outcome: {self.result_str} ({self.winner_note})."]
        for fc in self.future_comments[:3]:
            parts.append(
                f"Later, at move {(fc['ply'] + 1) // 2} ({fc['san']}): {fc['text'][:160]}"
            )
        return "\n".join(parts)[:1200]


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
        eff, level, side = _resolve_commentary_params(
            llm_effort, commentary_level, comment_side
        )
        state.metadata = state.metadata.model_copy(
            update={"commentary_level": level, "comment_side": side}
        )
        gid_token = llm_call_log.set_game_context(state.metadata.id)
        try:
            _promote_turning_points(state.move_events, state.context)
            _apply_back_to_back_key_moment_suppression(state.move_events, state.context)

            result_str, winner_note = _game_outcome(state.metadata.result)
            run = _LlmRun(
                state=state,
                service=advanced_commenter,
                eff=eff,
                level=level,
                side=side,
                progress_callback=progress_callback,
                commentary_callback=commentary_callback,
                result_str=result_str,
                winner_note=winner_note,
            )
            # Reverse order: each comment can see what happens later (foreshadowing).
            ordered_mis = _key_moment_order(state.move_events, side)
            n_total = max(
                sum(
                    1 for move_event in state.move_events if move_event.key_moment_type
                ),
                1,
            )
            for pos, mi in enumerate(ordered_mis):
                await self._comment_one_key_moment(run, mi, pos, n_total)

            state.context.critical_moments = [
                e for e in state.move_events if e.is_critical
            ]
            state.llm_done = True
            if commentary_callback:
                await commentary_callback("COMMENTARY_COMPLETE", {})
            _log_audit(run)
            if progress_callback:
                await progress_callback(LLM_DONE_PROGRESS, "Commentary complete.")
        finally:
            llm_call_log.reset_game_context(gid_token)

    async def _comment_one_key_moment(
        self, run: _LlmRun, mi: int, pos: int, n_total: int
    ) -> None:
        """Generate and broadcast the LLM comment for one key-moment move."""
        move_events = run.state.move_events
        context = run.state.context
        analyzed_rows = run.state.analyzed_rows
        advanced_commenter = run.service
        eff = run.eff
        level = run.level
        progress_callback = run.progress_callback
        commentary_callback = run.commentary_callback
        future_comments = run.future_comments
        move_event = move_events[mi]
        composer_model = resolve_model(
            advanced_commenter.provider_name,
            "composer",
            pass_label="key_moment",
        )
        move_ctx = llm_call_log.set_move_context(
            ply=move_event.ply,
            pass_label="key_moment",
        )
        try:
            pct = LLM_PROGRESS_START + (pos / n_total) * LLM_PROGRESS_SPAN
            if progress_callback:
                await progress_callback(
                    pct, f"LLM: key moment {move_event.san} (ply {move_event.ply})"
                )
            row = (
                analyzed_rows[move_event.move_index]
                if 0 <= move_event.move_index < len(analyzed_rows)
                else None
            )
            me_classified = move_event.model_copy(
                update={"move_category": classify_move_event(move_event)}
            )
            move_events[mi] = me_classified
            context.move_events[mi] = me_classified
            move_id = move_event.move_index + 1
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
                    key_moment_type=move_event.key_moment_type,
                    composer_pass_label="key_moment",
                    future_context=run.future_context_text(),
                    commentary_level=level,
                )
                mctx = await MoveCommentaryPipeline().run(mctx)
                text = mctx.final_text
                llm_debug = dict(mctx.llm_debug or {})

                if text:
                    _write_llm_comment(analyzed_rows, move_event.ply, text, llm_debug)
                    arch_snip = str(
                        (context.game_digest or {}).get("strategic_archetype") or ""
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
                        f"After {move_event.san} (ply {move_event.ply}): {motif_hint}; archetype={arch_snip}"
                    )
                    context.prior_context_snippets = hints[-6:]
                    # Reverse order: this comment is "the future" for
                    # every move still to be commented.
                    future_comments.insert(
                        0, {"ply": move_event.ply, "san": move_event.san, "text": text}
                    )
                _record_audit(run, llm_debug)
                if commentary_callback and text:
                    resolved_tokens = resolve_tokens_for_comment(
                        text, move_event.fen_before, move_event.fen_after
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
                logger.error(f"LLM move commentary failed at ply {move_event.ply}: {e}")
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
