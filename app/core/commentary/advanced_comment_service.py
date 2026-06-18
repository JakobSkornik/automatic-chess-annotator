from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import chess

from app.core.commentary.annotation_tokens import auto_tokenize
from app.core.commentary.composer_prompts import (
    COMPOSER_OUTPUT_SCHEMA,
    COMPOSER_ROLE_PROMPTS,
    EPISODE_COMMENTARY_SCHEMA,
    EPISODE_NARRATIVE_PROMPT,
    POSITIONAL_PLAN_COMPOSER_PROMPT,
    composer_role_key,
    composer_system_prompt,
)
from app.core.commentary.composer_tiers import (
    KEY_MOMENT_TIERS,
    TIER_SINGLE,
    compute_rag_top_k,
    detail_level_for_key_moment,
)
from app.core.commentary.forbidden_phrases import (
    forbidden_hit_count,
    forbidden_hit_strings,
)
from app.core.commentary.json_utils import strip_json_fence
from app.core.commentary.llm_call_log import append_postcheck
from app.core.commentary.llm_call_log import log_call as log_llm_call
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.llm_providers import (
    DYNAMIC_SECTION_SENTINEL,
    LlmProvider,
    make_llm_provider,
)
from app.core.commentary.motif_phrases import glossary_phrase_for
from app.core.commentary.move_rationale import build_rationale, prompt_projection
from app.core.commentary.rag_filtering import (
    rag_annotation_passes_filters,
    rag_idea_overlap_tokens,
    rag_min_score,
    truncate_annotation_at_sentence,
)
from app.core.commentary.rag_retriever import (
    RAGResult,
    RAGRetriever,
    build_rag_query,
)
from app.core.commentary.tantivy_positional_retriever import get_default_retriever
from app.models.chess_events import (
    AnalyzedMoveData,
    Episode,
    GameAnalysisContext,
    MoveCategory,
    MoveEvent,
    MoveRationale,
)

logger = logging.getLogger(__name__)


def _log_llm_prompts_enabled() -> bool:
    return os.environ.get("LOG_LLM_PROMPTS", "").strip().lower() in ("1", "true")


def _debug_log_prompt(name: str, system: str, user: str) -> None:
    """Full system + user text for dev tracing (LOG_LLM_PROMPTS). No truncation."""
    if not _log_llm_prompts_enabled():
        return
    logger.info(
        "LLM_DEBUG_PROMPT name=%s\n---\nSYSTEM:\n%s\n---\nUSER:\n%s",
        name,
        system,
        user,
    )


# ---------------------------------------------------------------------------
# Text sanitisation – replace Unicode typography with ASCII equivalents
# to prevent mojibake when the frontend or terminal uses a non-UTF-8 encoding.
# ---------------------------------------------------------------------------
_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2018", "'"),  # left single quote
    ("\u2019", "'"),  # right single quote
    ("\u201c", '"'),  # left double quote
    ("\u201d", '"'),  # right double quote
    ("\u2014", " -- "),  # em dash
    ("\u2013", " - "),  # en dash
    ("\u2011", "-"),  # non-breaking hyphen
    ("\u2010", "-"),  # hyphen
    ("\u2026", "..."),  # horizontal ellipsis
    ("\u00a0", " "),  # non-breaking space
    ("\u200b", ""),  # zero-width space
]


def sanitize_text(text: str) -> str:
    """Replace common Unicode typographic characters with ASCII equivalents."""
    for src, dst in _UNICODE_REPLACEMENTS:
        text = text.replace(src, dst)
    return text


def _composer_forcing_pv_bracket(move_event: MoveEvent) -> str | None:
    """Bracket snippet for prompts / post-check when captures or checks need a short PV."""
    try:
        b = chess.Board(move_event.fen_before)
        m = chess.Move.from_uci(move_event.uci)
        is_cap = b.is_capture(m)
        b.push(m)
        is_chk = b.is_check()
    except Exception:
        is_cap = False
        is_chk = False
    km = move_event.key_moment_type or ""
    if not (is_cap or is_chk or km in ("blunder", "critical_decision", "brilliant")):
        return None
    for pv in move_event.pv_lines[:1]:
        line = pv.get("line_san") or []
        if len(line) >= 2:
            frag = " ".join(line[:3])
            return f"[pv:{frag}]"
    return None


STRATEGIC_MOTIF_CONTRADICTIONS: tuple[frozenset, ...] = (
    frozenset({"bad_bishop", "bishop_pair_advantage"}),
)


def _collapse_contradictory_strategic(labels: list[str], primary: str) -> list[str]:
    """Drop redundant conflicting strategic tags; prefer rationale primary when tied."""
    out = list(dict.fromkeys(labels))
    pri = primary.strip()
    for group in STRATEGIC_MOTIF_CONTRADICTIONS:
        present = [x for x in out if x in group]
        if len(present) <= 1:
            continue
        keep = pri if pri in present else present[0]
        out = [x for x in out if x not in group or x == keep]
    return out


def cap_motifs_for_prompt(move_event: MoveEvent, primary_motif: str = "") -> list[str]:
    """Expose at most 3 tactical + 2 strategic motifs, deduped, primary first."""
    tact = [m.value for m in move_event.tactical_motifs]
    strat = _collapse_contradictory_strategic(
        [m.value for m in move_event.strategic_motifs],
        primary_motif,
    )
    tact_u = list(dict.fromkeys(tact))
    strat_u = list(dict.fromkeys(strat))
    pri = primary_motif.strip()

    tact_ord: list[str] = []
    if pri and pri in tact_u:
        tact_ord.append(pri)
    for t in tact_u:
        if t not in tact_ord:
            tact_ord.append(t)
    tact_ord = tact_ord[:3]

    strat_ord: list[str] = []
    if pri and pri in strat_u:
        strat_ord.append(pri)
    for s in strat_u:
        if s not in strat_ord:
            strat_ord.append(s)
    strat_ord = strat_ord[:2]

    return list(dict.fromkeys(tact_ord + strat_ord))


def format_motif_digest_lines(keys: list[str]) -> str:
    if not keys:
        return "(none)"
    return "\n".join(f"- {k}: {glossary_phrase_for(k)}" for k in keys)


def compact_game_context_for_move(
    digest: dict[str, Any],
    current_ply: int,
    *,
    current_phase: str = "",
) -> dict[str, Any]:
    """Subset of digest for per-move prompts: no spoilers, prior turning points only."""
    arch = digest.get("strategic_archetype", "")
    out: dict[str, Any] = {"strategic_archetype": arch}
    if current_ply <= 20:
        oc = digest.get("opening_character", "")
        if isinstance(oc, str) and oc.strip():
            out["opening_character"] = oc.strip()

    tps = digest.get("turning_points") or []
    past: list[dict[str, Any]] = []
    for tp in tps:
        if not isinstance(tp, dict):
            continue
        try:
            p = int(tp.get("ply", 0))
        except (TypeError, ValueError):
            continue
        if p < current_ply:
            past.append(tp)
    out["turning_points"] = past[:8]

    phase_story_full = digest.get("phase_story") or []
    ph_norm = (current_phase or "").strip().lower()
    phase_rows: list[dict[str, Any]] = []
    for row in phase_story_full:
        if not isinstance(row, dict):
            continue
        ph = str(row.get("phase") or "").strip().lower()
        if ph_norm and ph == ph_norm:
            phase_rows.append(row)
    out["phase_story"] = phase_rows[:2]

    def approx_tokens(d: dict[str, Any]) -> int:
        return max(1, len(json.dumps(d, ensure_ascii=False)) // 4)

    while approx_tokens(out) > 250 and len(out.get("turning_points") or []) > 2:
        out["turning_points"] = out["turning_points"][1:]
    while approx_tokens(out) > 250 and out.get("phase_story"):
        out["phase_story"] = []
    return out


def compute_commentary_audit(
    text: str,
    detected_motifs: list[str],
    *,
    rag_had_hits: bool = False,
    rag_applied: bool | None = None,
) -> dict[str, Any]:
    eval_tokens = len(re.findall(r"\[eval:[^\]]+\]", text))
    forbidden = forbidden_hit_count(text)
    if detected_motifs:
        lowered = text.lower()
        covered = sum(
            1
            for m in detected_motifs
            if m.lower() in lowered or m.replace("_", " ").lower() in lowered
        )
        coverage = covered / max(len(detected_motifs), 1)
    else:
        coverage = 1.0
    audit: dict[str, Any] = {
        "motif_coverage_ratio": round(coverage, 3),
        "eval_token_count": eval_tokens,
        "forbidden_phrase_hits": forbidden,
        "rag_had_hits": rag_had_hits,
        "rag_applied": rag_applied,
        "rag_usage_ratio": None,
    }
    if rag_had_hits and rag_applied is not None:
        audit["rag_usage_ratio"] = 1.0 if rag_applied else 0.0
    return audit


def build_planned_llm_passes_and_system_prompts(
    key_moment_type: str | None,
    move_category: str | None,
    tier_effort: str,
    *,
    detail_level: str = "minimal",
    strategic_archetype: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Planned passes and system prompts for WS debug (mirrors analyze_and_compose_raw_text)."""
    km = key_moment_type or ""
    tier = KEY_MOMENT_TIERS.get(km, TIER_SINGLE)
    cat_key = move_category or MoveCategory.POSITIONAL.value
    role = composer_role_key(cat_key)
    if detail_level == "book":
        composer_prompt = COMPOSER_ROLE_PROMPTS.get(
            role, POSITIONAL_PLAN_COMPOSER_PROMPT
        )
    else:
        arch = (strategic_archetype or "other").strip() or "other"
        composer_prompt = composer_system_prompt(role, archetype=arch)
    composer_name = f"composer_{role}"
    passes: list[dict[str, Any]] = [
        {
            "name": "composer_single",
            "effort": tier_effort,
            "tier_max_output_tokens": tier.get("max_tokens"),
        }
    ]
    system_prompts: list[dict[str, str]] = [
        {"name": composer_name, "text": composer_prompt}
    ]
    return passes, system_prompts


# Character budget for the RAG annotations block, by prose detail level.
RAG_MAX_CHARS_BY_DETAIL = {"full": 1600, "minimal": 500, "book": 500}
DEFAULT_RAG_MAX_CHARS = 800

# Composer pass budgeting / messages.
LLM_TOKEN_WARN_THRESHOLD = 4000
_LLM_DISABLED_MESSAGE = (
    "AI comments are disabled. Set OPENAI_API_KEY (OpenAI) or "
    "ANTHROPIC_API_KEY (Anthropic)."
)
_FORBIDDEN_RETRY_SUFFIX = (
    "\n\nRevise your commentary JSON: remove these banned phrases entirely: "
)
COMPOSER_MAX_ATTEMPTS = 2
COMPOSER_RAW_PREVIEW_CHARS = 500
_UNDER_WORD_LIMIT_PREFIX = "Keep the answer under 70 words.\n\n"
_PV_TOKEN_RE = re.compile(r"\[pv\s*:")


def _empty_composer_extras() -> dict[str, Any]:
    return {"better_alternative": "", "rag_idea_used": "", "rag_applied": False}


def _is_forbidden_retry_enabled() -> bool:
    return os.environ.get("FORBIDDEN_PHRASE_RETRY", "").strip().lower() in ("1", "true")


def _as_str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


@dataclass
class _ComposerInputs:
    """Everything a composer pass needs besides the user text."""

    composer_prompt: str
    model: str | None
    effort: str | None
    max_output_tokens: int
    fen_before: str | None
    fen_after: str | None
    rag_snippets: list[str]
    motif_hint_allowlist: list[str]
    forcing_pv_bracket: str | None


@dataclass
class _EventCtx:
    """Shared per-move inputs threaded through the prompt-building submethods."""

    move_event: MoveEvent
    game_context: GameAnalysisContext
    analyzed_row: AnalyzedMoveData | None
    detail: str
    rationale: MoveRationale
    motif_hint_keys: list[str]
    played_move_block: str
    digest: dict[str, Any]
    digest_archetype: str


@dataclass
class _ComposedPrompt:
    """The assembled user prompt plus the bits the debug trace needs back."""

    text: str
    rag_snippets_used: list[str] = field(default_factory=list)
    rag_hit_count: int = 0
    compact_context: dict[str, Any] | None = None


class AdvancedCommentService:
    def __init__(
        self,
        rag_retriever: RAGRetriever | None = None,
        *,
        provider: LlmProvider | None = None,
        provider_key: str | None = None,
    ) -> None:
        self._provider: LlmProvider = provider or make_llm_provider(provider_key)
        self._rag: RAGRetriever = rag_retriever or get_default_retriever()
        self._last_token_usage: int = 0
        self._last_llm_log_seq: int | None = None

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @staticmethod
    def _format_move_event_block(
        move_event: MoveEvent, motif_hint_keys: list[str]
    ) -> str:
        fe = move_event
        parts: list[str] = []
        parts.append(f"Move: {fe.san} (ply {fe.ply})")
        mover = "?"
        to_move_after = "?"
        try:
            b = chess.Board(fe.fen_before)
            mover = "White" if b.turn == chess.WHITE else "Black"
            to_move_after = "Black" if b.turn == chess.WHITE else "White"
        except Exception:
            pass
        parts.append(f"Move played by: {mover}")
        parts.append(f"Side to move after: {to_move_after}")
        parts.append(f"Phase: {fe.phase}")
        pb = fe.eval_before_cp / 100.0 if fe.eval_before_cp is not None else None
        pa = fe.eval_after_cp / 100.0 if fe.eval_after_cp is not None else None
        sw = fe.eval_swing_cp / 100.0 if fe.eval_swing_cp is not None else None
        if pb is not None:
            parts.append(f"Eval before (White POV): {pb:+.2f} pawns")
        if pa is not None:
            parts.append(f"Eval after (White POV): {pa:+.2f} pawns")
        if sw is not None:
            parts.append(f"Eval swing: {sw:+.2f} pawns")
        if fe.best_move_san:
            bev = (
                fe.best_move_eval_cp / 100.0
                if fe.best_move_eval_cp is not None
                else None
            )
            parts.append(
                f"Best engine move: {fe.best_move_san}"
                + (f" (eval {bev:+.2f})" if bev is not None else "")
            )
        km = fe.key_moment_type or ""
        pv_cap = 3 if km in ("brilliant", "blunder") else 2
        for pv in fe.pv_lines[:pv_cap]:
            line = pv.get("line_san") or []
            rnk = pv.get("rank")
            ev = pv.get("eval_cp")
            evp = ev / 100.0 if isinstance(ev, (int, float)) else None
            parts.append(
                f"PV #{rnk}: {' '.join(line[:8])}"
                + (f" (root eval {evp:+.2f})" if evp is not None else "")
            )
        parts.append(f"Move quality: {fe.move_quality.value}")
        parts.append(f"Event type: {fe.event_type.value}")
        if fe.move_category:
            parts.append(f"Move category: {fe.move_category.value}")
        if fe.pawn_structure_type:
            parts.append(f"Pawn structure (center): {fe.pawn_structure_type}")
        if fe.opening_name or fe.opening_eco:
            parts.append(
                f"Opening: {fe.opening_name or ''} ({fe.opening_eco or ''})".strip()
            )
        if fe.eval_instability_cp is not None:
            parts.append(
                f"Eval instability across depths 8/12/16 (cp): {fe.eval_instability_cp}"
            )
        parts.append(f"FEN after (side to move = {to_move_after}): {fe.fen_after}")
        return "\n".join(parts)

    @staticmethod
    def _format_pv_position_comparison(
        move_event: MoveEvent,
        analyzed_row: AnalyzedMoveData | None,
    ) -> str:
        """One-line engine line vs played position (no feature tables)."""
        if not analyzed_row or not analyzed_row.pvs or not analyzed_row.pvs[0]:
            return ""
        if move_event.best_move_uci and move_event.uci == move_event.best_move_uci:
            return ""
        pv_seq = analyzed_row.pvs[0]
        first = pv_seq[0]
        uci = getattr(first, "move", None)
        first_san = ""
        if uci:
            try:
                b = chess.Board(move_event.fen_before)
                first_san = b.san(chess.Move.from_uci(str(uci)))
            except Exception:
                first_san = ""
        xp = (
            move_event.eval_after_cp / 100.0
            if move_event.eval_after_cp is not None
            else None
        )
        yp = None
        sc = getattr(first, "score", None)
        if isinstance(sc, (int, float)):
            yp = sc / 100.0
        if xp is None and yp is None and not first_san:
            return ""
        parts = []
        if first_san:
            parts.append(f"PV diverges after {first_san}")
        if xp is not None:
            parts.append(f"eval after played (White POV): {xp:+.2f} pawns")
        if yp is not None:
            parts.append(f"root eval if best first move (White POV): {yp:+.2f} pawns")
        return "; ".join(parts) + "."

    @staticmethod
    def _format_future_line_block(move_event: MoveEvent) -> str:
        """Deep line comparison after N plies along played vs engine PV (critical moves)."""
        fl = move_event.future_line
        if not fl:
            return ""
        parts = ["FUTURE LINE COMPARISON (after engine PV continuation, White POV cp):"]
        if fl.played_line_san:
            parts.append(
                f"Played-line SAN (first plies): {' '.join(fl.played_line_san)}"
            )
        if fl.best_line_san:
            parts.append(f"Best-line SAN (first plies): {' '.join(fl.best_line_san)}")
        if fl.played_leaf_eval_cp is not None:
            parts.append(
                f"Eval at played-line leaf: {fl.played_leaf_eval_cp / 100.0:+.2f} pawns"
            )
        if fl.best_leaf_eval_cp is not None:
            parts.append(
                f"Eval at best-line leaf: {fl.best_leaf_eval_cp / 100.0:+.2f} pawns"
            )
        if fl.eval_gap_cp is not None:
            parts.append(
                f"Leaf eval gap (best - played): {fl.eval_gap_cp / 100.0:+.2f} pawns"
            )
        if fl.feature_deltas:
            fd = ", ".join(
                f"{k}={v:+.3f}" for k, v in list(fl.feature_deltas.items())[:8]
            )
            parts.append(f"Feature deltas (best vs played leaf): {fd}")
        if fl.played_targets:
            parts.append(
                f"Recurring destination squares (played line): {', '.join(fl.played_targets)}"
            )
        if fl.best_targets:
            parts.append(
                f"Recurring destination squares (best line): {', '.join(fl.best_targets)}"
            )
        return "\n".join(parts)

    @staticmethod
    def _format_rag_block(
        results: list[RAGResult], *, max_chars: int = 1600
    ) -> tuple[str, list[str]]:
        if not results:
            return "", []
        cap = max(80, int(max_chars))
        min_sc = rag_min_score()
        lines = [
            "MASTER ANNOTATIONS — these excerpts come from strongly annotated GM/IM games "
            "in tactically or structurally similar positions.",
            "If an idea plainly applies here, weave it into your prose (ONE short paraphrase). "
            "Do NOT quote verbatim.",
        ]
        kept_snippets: list[str] = []
        shown = 0
        for r in results:
            sc = r.similarity_score
            if sc is not None and sc < min_sc:
                continue
            ann_raw = (r.annotation_text or "").strip()
            if not rag_annotation_passes_filters(ann_raw):
                continue
            ann = truncate_annotation_at_sentence(ann_raw, cap)
            if len(ann.split()) < 12:
                continue
            shown += 1
            tags = r.relevance_tags or {}
            ph = (tags.get("phase") or "").strip()
            on = (tags.get("opening_name") or tags.get("opening") or "").strip()
            eco = (tags.get("opening_eco") or tags.get("eco") or "").strip()
            mats = (tags.get("material_signature") or "").strip()
            score_txt = f"score={sc:.3f}" if sc is not None else "score=?"
            head_bits = [
                f"[{shown}]",
                score_txt,
                f"phase={ph}" if ph else "",
                f"opening={on}" if on else "",
                f"eco={eco}" if eco else "",
                f"material={mats}" if mats else "",
            ]
            lines.append(" ".join(b for b in head_bits if b))
            lines.append(f"    {ann}")
            kept_snippets.append(ann)
        if not kept_snippets:
            return "", []
        return "\n".join(lines), kept_snippets

    @staticmethod
    def _format_move_event_minimal(
        move_event: MoveEvent, motif_hint_keys: list[str]
    ) -> str:
        """Reduced engine/position block for low-tier moves (no PV dump, no feature laundry list)."""
        fe = move_event
        parts: list[str] = []
        parts.append(f"Move: {fe.san} (ply {fe.ply})")
        try:
            b = chess.Board(fe.fen_before)
            mover = "White" if b.turn == chess.WHITE else "Black"
        except Exception:
            mover = "?"
        parts.append(f"Move played by: {mover}")
        parts.append(f"Phase: {fe.phase}")
        pb = fe.eval_before_cp / 100.0 if fe.eval_before_cp is not None else None
        pa = fe.eval_after_cp / 100.0 if fe.eval_after_cp is not None else None
        sw = fe.eval_swing_cp / 100.0 if fe.eval_swing_cp is not None else None
        if pb is not None:
            parts.append(f"Eval before (White POV): {pb:+.2f} pawns")
        if pa is not None:
            parts.append(f"Eval after (White POV): {pa:+.2f} pawns")
        if sw is not None:
            parts.append(f"Eval swing: {sw:+.2f} pawns")
        if fe.best_move_san:
            bev = (
                fe.best_move_eval_cp / 100.0
                if fe.best_move_eval_cp is not None
                else None
            )
            parts.append(
                f"Best engine move: {fe.best_move_san}"
                + (f" (eval {bev:+.2f})" if bev is not None else "")
            )
        parts.append(f"Move quality: {fe.move_quality.value}")
        if fe.move_category:
            parts.append(f"Move category: {fe.move_category.value}")
        if fe.pawn_structure_type:
            parts.append(f"Pawn structure (center): {fe.pawn_structure_type}")
        if fe.opening_name or fe.opening_eco:
            parts.append(
                f"Opening: {fe.opening_name or ''} ({fe.opening_eco or ''})".strip()
            )
        parts.append(f"FEN after: {fe.fen_after}")
        br = _composer_forcing_pv_bracket(fe)
        if br:
            parts.append(f"Forcing line ready: {br}")
        return "\n".join(parts)

    async def build_event_llm_input(
        self,
        move_event: MoveEvent,
        episode: Episode | None,
        game_context: GameAnalysisContext,
        analyzed_row: AnalyzedMoveData | None = None,
        composer_effort: str | None = None,
        *,
        rag_results: list[RAGResult] | None = None,
        rationale_override: MoveRationale | None = None,
    ) -> tuple[str, list[RAGResult], dict[str, Any]]:
        query = build_rag_query(move_event, episode)
        detail = detail_level_for_key_moment(move_event)
        rag_top_k = compute_rag_top_k(detail, query.phase)
        rationale = (
            rationale_override
            if rationale_override is not None
            else build_rationale(move_event, move_event.future_line)
        )
        motif_hint_keys = cap_motifs_for_prompt(
            move_event, rationale.primary_motif_label
        )

        rag_retrieval_debug: dict[str, Any] = {}
        if rag_results is None:
            rag_results = await self._retrieve_event_rag(
                query, move_event, detail, rag_top_k, rag_retrieval_debug
            )

        digest = getattr(game_context, "game_digest", None) or {}
        ctx = _EventCtx(
            move_event=move_event,
            game_context=game_context,
            analyzed_row=analyzed_row,
            detail=detail,
            rationale=rationale,
            motif_hint_keys=motif_hint_keys,
            played_move_block=self._format_played_move_block(move_event),
            digest=digest,
            digest_archetype=str(digest.get("strategic_archetype") or "other").strip()
            or "other",
        )

        prompt = (
            _ComposedPrompt(self._build_book_user_text(ctx))
            if detail == "book"
            else self._build_enriched_user_text(ctx, rag_results or [])
        )
        debug_dict = self._build_event_debug(
            ctx, query, rag_top_k, rag_retrieval_debug, prompt, composer_effort
        )
        logger.info(
            "build_event_llm_input: move_category=%s key_moment_type=%s detail=%s",
            debug_dict["move_category"],
            debug_dict["key_moment_type"],
            detail,
        )
        _debug_log_prompt("build_event_input", "", prompt.text)
        return prompt.text, rag_results or [], debug_dict

    async def _retrieve_event_rag(
        self,
        query: Any,
        move_event: MoveEvent,
        detail: str,
        rag_top_k: int,
        rag_retrieval_debug: dict[str, Any],
    ) -> list[RAGResult]:
        """Retrieve master-game annotations for this move (none for book moves)."""
        if detail == "book":
            return []
        rag_results = await self._rag.retrieve(
            query, top_k=max(1, rag_top_k), retrieval_debug=rag_retrieval_debug
        )
        self._log_rag_results(query, move_event, rag_results, rag_retrieval_debug)
        return rag_results

    @staticmethod
    def _log_rag_results(
        query: Any,
        move_event: MoveEvent,
        rag_results: list[RAGResult],
        rag_retrieval_debug: dict[str, Any],
    ) -> None:
        logger.info(
            "RAG query: phase=%s pawn_structure=%s motifs=%s theme=%s fen=%s pv_san=%s",
            query.phase,
            query.pawn_structure_type,
            query.tactical_motifs,
            query.theme_hint,
            (query.fen or "")[:80],
            query.pv_san,
        )
        for i, r in enumerate(rag_results):
            logger.info(
                "RAG hit #%d: source=%s fen=%.60s score=%.3f text=%.120s",
                i + 1,
                r.source,
                r.fen or "",
                r.similarity_score or 0.0,
                r.annotation_text,
            )
        if rag_results:
            return
        logger.info("RAG: no results returned")
        km_w = move_event.key_moment_type or ""
        if km_w in ("critical_decision", "mistake", "blunder") and rag_retrieval_debug:
            logger.info(
                "RAG rejection debug ply=%s key_moment=%s %s",
                move_event.ply,
                km_w,
                rag_retrieval_debug,
            )

    @staticmethod
    def _format_played_move_block(move_event: MoveEvent) -> str:
        try:
            board = chess.Board(move_event.fen_before)
            mover_side = "White" if board.turn == chess.WHITE else "Black"
            next_side = "Black" if board.turn == chess.WHITE else "White"
        except Exception:
            mover_side = "?"
            next_side = "?"
        dots = "." if mover_side == "White" else "..."
        move_label = f"{(move_event.ply + 1) // 2}{dots} {move_event.san}"
        return (
            "PLAYED_MOVE (this is the ONLY move to describe):\n"
            f"- SAN: {move_event.san}\n"
            f"- Numbered: {move_label}\n"
            f"- UCI: {move_event.uci}\n"
            f"- Ply: {move_event.ply}\n"
            f"- Move index: {move_event.move_index}\n"
            f"- FEN before: {move_event.fen_before}\n"
            f"- FEN after:  {move_event.fen_after}\n"
            f"- Played by: {mover_side}\n"
            f"- Side to move after: {next_side}\n\n"
            "TASK: Describe THIS move and the position it creates.\n"
            "Do NOT write the commentary as if the engine's best reply (or any PV continuation) "
            "had been played. If the played move is a simple recapture or quiet developing move, "
            "say so plainly — do not describe the opponent's upcoming threat as if it were this move."
        )

    def _build_book_user_text(self, ctx: _EventCtx) -> str:
        move_event = ctx.move_event
        opening_lines: list[str] = []
        if move_event.opening_name or move_event.opening_eco:
            opening_lines.append(
                f"Opening: {move_event.opening_name or ''} ({move_event.opening_eco or ''})".strip()
            )
        opening_hdr = ("\n\n".join(opening_lines) + "\n\n") if opening_lines else ""
        show_motifs = (
            bool(move_event.tactical_motifs) and (move_event.eval_swing_cp or 0) != 0
        )
        mot_block = ""
        if show_motifs and ctx.motif_hint_keys:
            mot_block = (
                "MOTIF HINTS (cap 3 tactical + 2 strategic; prose may name at most ONE motif key):\n"
                + format_motif_digest_lines(ctx.motif_hint_keys)
                + "\n\n"
            )
        dynamic_blocks = [
            opening_hdr + mot_block + ctx.played_move_block,
            "POSITION SUMMARY (anchor to this; do not invent lines):\n"
            + self._format_move_event_minimal(move_event, []),
        ]
        return "\n\n".join(dynamic_blocks)

    def _build_enriched_user_text(
        self, ctx: _EventCtx, rag_results: list[RAGResult]
    ) -> _ComposedPrompt:
        move_event = ctx.move_event
        rag_block, rag_snippets_used = self._format_rag_block(
            rag_results, max_chars=self._rag_max_chars(ctx.detail)
        )
        static_blocks = self._build_static_blocks(ctx)
        dynamic_blocks = self._build_dynamic_blocks(ctx)
        dynamic_blocks.append(rag_block)
        text = (
            "\n\n".join(static_blocks)
            + DYNAMIC_SECTION_SENTINEL
            + "\n\n".join(b for b in dynamic_blocks if b)
        )
        compact_injected = (
            compact_game_context_for_move(
                ctx.digest, move_event.ply, current_phase=move_event.phase or ""
            )
            if ctx.digest
            else None
        )
        return _ComposedPrompt(
            text=text,
            rag_snippets_used=rag_snippets_used,
            rag_hit_count=len(rag_results),
            compact_context=compact_injected,
        )

    def _build_static_blocks(self, ctx: _EventCtx) -> list[str]:
        move_event = ctx.move_event
        blocks: list[str] = []
        if ctx.digest:
            compact_ctx = compact_game_context_for_move(
                ctx.digest, move_event.ply, current_phase=move_event.phase or ""
            )
            blocks.append(
                "GAME CONTEXT (whole-game digest; use for tone and continuity, do not re-quote):\n"
                + json.dumps(compact_ctx, indent=2)
            )
        pri = getattr(ctx.game_context, "prior_context_snippets", None) or []
        if pri:
            blocks.append(
                "PRIOR_CONTEXT (continuity only; do not restate):\n"
                + "\n".join(pri[-2:])
            )
        blocks.append(
            "MOTIF HINTS (cap 3 tactical + 2 strategic; prose may name at most ONE motif key):\n"
            + (
                format_motif_digest_lines(ctx.motif_hint_keys)
                if ctx.motif_hint_keys
                else "(none)"
            )
        )
        return blocks

    def _build_dynamic_blocks(self, ctx: _EventCtx) -> list[str]:
        """Move-specific prompt blocks (played move, rationale, position/engine data)."""
        move_event = ctx.move_event
        proj_detail = (
            "full"
            if ctx.detail == "full"
            else ("compact" if ctx.detail == "compact" else "minimal")
        )
        rationale_blob = json.dumps(
            prompt_projection(ctx.rationale, detail=proj_detail), indent=2
        )
        blocks: list[str] = [
            ctx.played_move_block,
            "MOVE_RATIONALE_JSON (internal evidence; weave meaning into prose—never quote JSON keys verbatim):\n"
            + rationale_blob,
        ]
        if ctx.detail == "full":
            blocks.append(
                "POSITION AND ENGINE DATA (anchor commentary to this; do not invent lines):\n"
                + self._format_move_event_block(move_event, ctx.motif_hint_keys)
            )
            blocks.append(
                self._format_pv_position_comparison(move_event, ctx.analyzed_row)
            )
            blocks.append(self._format_future_line_block(move_event))
        else:
            blocks.append(
                "POSITION SUMMARY (anchor to this; do not invent lines):\n"
                + self._format_move_event_minimal(move_event, ctx.motif_hint_keys)
            )
        return blocks

    @staticmethod
    def _rag_max_chars(detail: str) -> int:
        return RAG_MAX_CHARS_BY_DETAIL.get(detail, DEFAULT_RAG_MAX_CHARS)

    def _build_event_debug(
        self,
        ctx: _EventCtx,
        query: Any,
        rag_top_k: int,
        rag_retrieval_debug: dict[str, Any],
        prompt: _ComposedPrompt,
        composer_effort: str | None,
    ) -> dict[str, Any]:
        move_event = ctx.move_event
        cat = move_event.move_category.value if move_event.move_category else None
        km = move_event.key_moment_type
        tier_base = dict(KEY_MOMENT_TIERS.get(km or "", TIER_SINGLE))
        if composer_effort:
            tier_base["effort"] = composer_effort
        tier_effort = str(tier_base.get("effort", TIER_SINGLE["effort"]))
        passes, system_prompts = build_planned_llm_passes_and_system_prompts(
            km,
            cat,
            tier_effort,
            detail_level=ctx.detail,
            strategic_archetype=ctx.digest_archetype if ctx.detail != "book" else None,
        )
        return {
            "move_category": cat,
            "key_moment_type": km,
            "tier": tier_base,
            "detail_level": ctx.detail,
            "motif_hint_keys": ctx.motif_hint_keys,
            "rag_top_k": rag_top_k,
            "rag_hit_count": prompt.rag_hit_count,
            "detected_motif_labels": ctx.motif_hint_keys,
            "rag_query": query.model_dump(),
            "rationale": ctx.rationale.model_dump(),
            "rag_retrieval_debug": rag_retrieval_debug or None,
            "system_prompts": system_prompts,
            "user_text": prompt.text,
            "passes": passes,
            "token_usage_total": None,
            "tokens_by_pass": {},
            "elapsed_ms": None,
            "composer_named_motifs": [],
            "commentary_audit": None,
            "game_digest": ctx.digest if ctx.digest else None,
            "game_context_injected": prompt.compact_context,
            "rag_snippets_used": prompt.rag_snippets_used,
            "composer_forcing_pv_bracket": _composer_forcing_pv_bracket(move_event),
        }

    async def analyze_and_compose_raw_text(
        self,
        structured_text: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        key_moment_type: str | None = None,
        move_category: str | None = None,
        llm_debug: dict[str, Any] | None = None,
        fen_before: str | None = None,
        fen_after: str | None = None,
    ) -> str:
        if not self._provider.is_configured():
            if llm_debug is not None:
                llm_debug["token_usage_total"] = None
            return _LLM_DISABLED_MESSAGE

        started_at = time.perf_counter()
        debug = llm_debug or {}
        tier = KEY_MOMENT_TIERS.get(key_moment_type or "", TIER_SINGLE)
        role = composer_role_key(move_category or MoveCategory.POSITIONAL.value)
        inputs = _ComposerInputs(
            composer_prompt=self._select_composer_prompt(role, debug),
            model=model or resolve_model(self.provider_name, "composer"),
            effort=effort or tier["effort"],
            max_output_tokens=int(tier.get("max_tokens", TIER_SINGLE["max_tokens"])),
            fen_before=fen_before,
            fen_after=fen_after,
            rag_snippets=list(debug.get("rag_snippets_used") or []),
            motif_hint_allowlist=[
                str(x) for x in (debug.get("motif_hint_keys") or []) if str(x).strip()
            ],
            forcing_pv_bracket=_as_str_or_none(
                debug.get("composer_forcing_pv_bracket")
            ),
        )
        self._log_composer_selection(move_category, key_moment_type, role)

        prose, composer_named, composer_extra = await self._compose_event_prose(
            inputs, structured_text
        )
        token_total = self._last_token_usage
        self._warn_if_token_budget_high(token_total)
        if llm_debug is not None:
            self._record_composer_debug(
                llm_debug,
                prose,
                composer_named,
                composer_extra,
                token_total,
                started_at,
            )
        return prose

    def _select_composer_prompt(self, role: str, debug: dict[str, Any]) -> str:
        if debug.get("detail_level") == "book":
            return COMPOSER_ROLE_PROMPTS.get(role, POSITIONAL_PLAN_COMPOSER_PROMPT)
        gd = debug.get("game_digest") or {}
        archetype = str(gd.get("strategic_archetype") or "other").strip() or "other"
        return composer_system_prompt(role, archetype=archetype)

    @staticmethod
    def _log_composer_selection(
        move_category: str | None, key_moment_type: str | None, role: str
    ) -> None:
        if not _log_llm_prompts_enabled():
            return
        logger.info(
            "analyze_and_compose_raw_text: branch=single move_category=%s "
            "key_moment_type=%s composer_role=%s",
            move_category,
            key_moment_type,
            role,
        )
        logger.info("composer system prompt selected: composer_%s", role)

    async def _compose_event_prose(
        self, inputs: _ComposerInputs, structured_text: str
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Run the composer pass, then one retry if banned phrases slip through."""
        prose, named, extra = await self._run_composer_with(
            inputs, structured_text, "composer_single"
        )
        if _is_forbidden_retry_enabled() and prose.strip():
            return await self._retry_without_forbidden_phrases(
                inputs, structured_text, prose, named, extra
            )
        return prose, named, extra

    async def _run_composer_with(
        self, inputs: _ComposerInputs, user_text: str, prompt_name: str
    ) -> tuple[str, list[str], dict[str, Any]]:
        return await self._run_composer_segments(
            inputs.composer_prompt,
            user_text,
            model=inputs.model,
            effort=inputs.effort,
            prompt_name=prompt_name,
            max_output_tokens=inputs.max_output_tokens,
            fen_before=inputs.fen_before,
            fen_after=inputs.fen_after,
            rag_snippets=inputs.rag_snippets,
            motif_hint_allowlist=inputs.motif_hint_allowlist,
            forcing_pv_bracket=inputs.forcing_pv_bracket,
        )

    async def _retry_without_forbidden_phrases(
        self,
        inputs: _ComposerInputs,
        structured_text: str,
        prose: str,
        named: list[str],
        extra: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, Any]]:
        hits = forbidden_hit_strings(prose)
        if not hits:
            return prose, named, extra
        revised_user = structured_text + _FORBIDDEN_RETRY_SUFFIX + "; ".join(hits)
        retry_prose, retry_named, retry_extra = await self._run_composer_with(
            inputs, revised_user, "composer_single_forbidden_retry"
        )
        if retry_prose.strip() and forbidden_hit_count(
            retry_prose
        ) < forbidden_hit_count(prose):
            return retry_prose, retry_named, retry_extra
        return prose, named, extra

    @staticmethod
    def _warn_if_token_budget_high(token_total: int) -> None:
        if token_total > LLM_TOKEN_WARN_THRESHOLD:
            logger.warning(
                "LLM token budget high for move: ~%s total (warn threshold %s)",
                token_total,
                LLM_TOKEN_WARN_THRESHOLD,
            )

    @staticmethod
    def _record_composer_debug(
        llm_debug: dict[str, Any],
        prose: str,
        composer_named: list[str],
        composer_extra: dict[str, Any],
        token_total: int,
        started_at: float,
    ) -> None:
        llm_debug["token_usage_total"] = token_total
        llm_debug["tokens_by_pass"] = {"composer": token_total}
        llm_debug["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        llm_debug["composer_named_motifs"] = composer_named
        llm_debug["composer_better_alternative"] = composer_extra.get(
            "better_alternative", ""
        )
        llm_debug["composer_rag_idea_used"] = composer_extra.get("rag_idea_used", "")
        llm_debug["composer_rag_applied"] = composer_extra.get("rag_applied")
        llm_debug["commentary_audit"] = compute_commentary_audit(
            prose,
            llm_debug.get("detected_motif_labels") or [],
            rag_had_hits=bool(llm_debug.get("rag_hit_count")),
            rag_applied=composer_extra.get("rag_applied"),
        )

    async def generate_episode_commentary(
        self,
        episode: Episode,
        *,
        model: str | None = None,
        effort: str = "low",
    ) -> str:
        if not self._provider.is_configured():
            return ""
        strat_union = sorted(
            {m.value for ev in episode.move_events for m in ev.strategic_motifs}
        )
        if len(strat_union) > 6:
            strat_line = ", ".join(strat_union[:6]) + " (additional themes omitted)"
        else:
            strat_line = ", ".join(strat_union) if strat_union else "(none)"
        moves_summary: list[str] = []
        for e in episode.move_events:
            eva = e.eval_after_cp / 100.0 if e.eval_after_cp is not None else None
            evs = f"{eva:+.2f}" if eva is not None else "?"
            mq = e.move_quality.value if e.move_quality else ""
            km = e.key_moment_type or ""
            tact = ",".join(m.value for m in e.tactical_motifs[:5])
            moves_summary.append(
                f"{e.san} | eval {evs} | mq={mq} | km={km} | tact={tact}"
            )
        text = (
            f"Episode: {episode.title}\n"
            f"Theme: {episode.dominant_theme}\n"
            f"Phase: {episode.phase}\n"
            f"Eval trend (White POV pawns): "
            f"{[x / 100.0 for x in episode.eval_trend]}\n"
            f"STRATEGIC THEMES (episode-wide): {strat_line}\n"
            f"Moves (SAN | eval | move_quality | key_moment | tactical):\n"
            + "\n".join(moves_summary)
            + "\n"
        )
        mdl = model or resolve_model(self.provider_name, "episode")
        raw = await self._llm_call_json_schema(
            EPISODE_NARRATIVE_PROMPT,
            text,
            model=mdl,
            effort=effort,
            schema=EPISODE_COMMENTARY_SCHEMA,
            schema_name="episode_commentary",
            max_output_tokens=512,
        )
        if raw:
            try:
                obj = json.loads(raw)
                return sanitize_text(obj.get("commentary", raw))
            except Exception:
                return sanitize_text(raw.strip())
        return ""

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    async def _run_composer_segments(
        self,
        structured_system: str,
        user_text: str,
        *,
        model: str | None,
        effort: str | None,
        prompt_name: str = "composer_segment",
        max_output_tokens: int | None = None,
        fen_before: str | None = None,
        fen_after: str | None = None,
        rag_snippets: list[str] | None = None,
        motif_hint_allowlist: list[str] | None = None,
        forcing_pv_bracket: str | None = None,
    ) -> tuple[str, list[str], dict[str, Any]]:
        """Structured composer JSON; tokenize prose; one retry on empty/failure."""
        start_fen = chess.Board().fen()
        fb = fen_before or start_fen
        fa = fen_after or start_fen
        rag_list = list(rag_snippets or [])
        last_raw: str | None = None
        last_plain_len: int | None = None
        for attempt in range(COMPOSER_MAX_ATTEMPTS):
            try:
                raw = await self._llm_call(
                    structured_system,
                    user_text if attempt == 0 else _UNDER_WORD_LIMIT_PREFIX + user_text,
                    model=model,
                    effort=effort,
                    use_structured_composer=True,
                    prompt_name=prompt_name if attempt == 0 else f"{prompt_name}_retry",
                    max_output_tokens=max_output_tokens,
                )
            except Exception as e:
                logger.warning(
                    "Structured composer failed (attempt %s): %s", attempt + 1, e
                )
                continue
            last_raw = raw
            if not raw:
                continue
            named, plain, extras = self._parse_composer_json(
                raw, motif_hint_allowlist, rag_list
            )
            last_plain_len = len(plain)
            prose = self._finalize_composer_prose(
                plain, extras["better_alternative"], forcing_pv_bracket, fb, fa
            )
            if prose:
                self._record_composer_postcheck(named, extras)
                return prose, named, extras
            logger.warning(
                "Structured composer empty prose after parse (attempt %s): "
                "raw_preview=%.500s plain_len=%s",
                attempt + 1,
                strip_json_fence(raw),
                last_plain_len,
            )
        self._log_composer_empty(last_raw, last_plain_len)
        return "", [], _empty_composer_extras()

    def _parse_composer_json(
        self, raw: str, motif_hint_allowlist: list[str] | None, rag_list: list[str]
    ) -> tuple[list[str], str, dict[str, Any]]:
        """Parse the composer JSON into (named motifs, prose, validated extras)."""
        obj = json.loads(strip_json_fence(raw))
        named = [
            str(x)
            for x in (obj.get("named_motifs") or [])
            if isinstance(x, str) and x.strip()
        ]
        allow_set = {str(x) for x in (motif_hint_allowlist or []) if str(x).strip()}
        if allow_set:
            named = [x for x in named if x in allow_set][:1]
        extras = self._validate_rag_extras(
            {
                "better_alternative": str(obj.get("better_alternative") or "").strip(),
                "rag_idea_used": str(obj.get("rag_idea_used") or "").strip(),
                "rag_applied": bool(obj.get("rag_applied", False)),
            },
            rag_list,
        )
        return named, str(obj.get("text") or "").strip(), extras

    @staticmethod
    def _validate_rag_extras(
        extras: dict[str, Any], rag_list: list[str]
    ) -> dict[str, Any]:
        """Clear the RAG claim unless an annotation idea overlaps the snippets."""
        applied = extras["rag_applied"]
        if applied and (
            not rag_list
            or not rag_idea_overlap_tokens(extras["rag_idea_used"], rag_list)
        ):
            extras["rag_applied"] = False
            extras["rag_idea_used"] = ""
        return extras

    @staticmethod
    def _finalize_composer_prose(
        plain: str,
        better_alternative: str,
        forcing_pv_bracket: str | None,
        fen_before: str,
        fen_after: str,
    ) -> str:
        """Tokenize the prose; append the alternative clause and forcing PV bracket."""
        if not plain:
            return ""
        out = sanitize_text(auto_tokenize(plain, fen_before, fen_after)).rstrip()
        if not out:
            return ""
        if better_alternative and better_alternative.lower() not in out.lower():
            out = f"{out} {better_alternative}".strip()
        if forcing_pv_bracket and not _PV_TOKEN_RE.search(out.lower()):
            out = f"{out} {forcing_pv_bracket}".strip()
        return out.strip()

    def _record_composer_postcheck(
        self, named: list[str], extras: dict[str, Any]
    ) -> None:
        ref = self._last_llm_log_seq
        if ref is None:
            return
        append_postcheck(
            ref_seq=ref,
            payload={
                "rag_applied": extras["rag_applied"],
                "rag_idea_used": extras["rag_idea_used"],
                "named_motifs": named,
            },
        )

    @staticmethod
    def _log_composer_empty(last_raw: str | None, last_plain_len: int | None) -> None:
        plain_len = last_plain_len
        if plain_len is None and last_raw:
            try:
                parsed = json.loads(strip_json_fence(last_raw))
                plain_len = len(str(parsed.get("text") or "").strip())
            except Exception:
                pass
        logger.warning(
            "Structured composer returned empty after retries; raw_preview=%s plain_len=%s",
            (last_raw or "")[:COMPOSER_RAW_PREVIEW_CHARS],
            plain_len,
        )

    async def _llm_call(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        use_structured_composer: bool = False,
        prompt_name: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str | None:
        """Make a single LLM call and return the raw text response."""
        if _log_llm_prompts_enabled() and prompt_name:
            _debug_log_prompt(prompt_name, system_prompt, user_text)
        resolved_model = model or resolve_model(self.provider_name, "composer")
        pass_name = prompt_name or (
            "composer_structured" if use_structured_composer else "text_call"
        )
        schema_for_log = (
            "chess_commentary_composer" if use_structured_composer else None
        )
        t0 = time.perf_counter()
        raw: str | None = None
        usage: int | None = None
        err: str | None = None
        ok = False
        try:
            if use_structured_composer:
                raw, usage = await self._provider.json_schema_call(
                    system_prompt,
                    user_text,
                    model=resolved_model,
                    effort=effort or "low",
                    schema=COMPOSER_OUTPUT_SCHEMA,
                    schema_name="chess_commentary_composer",
                    max_output_tokens=max_output_tokens,
                )
                self._last_token_usage = usage
                ok = True
                return raw or None
            raw, usage = await self._provider.text_call(
                system_prompt,
                user_text,
                model=resolved_model,
                effort=effort or "low",
                max_output_tokens=max_output_tokens,
            )
            self._last_token_usage = usage
            ok = True
            return raw or None
        except Exception as e:
            err = repr(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if ok and usage is None:
                logger.warning(
                    "LLM call completed without usage metadata (pass_name=%s model=%s)",
                    pass_name,
                    resolved_model,
                )
            seq = log_llm_call(
                pass_name=pass_name,
                system=system_prompt,
                user=user_text,
                response=raw if raw is not None else "",
                model=resolved_model,
                effort=effort or "low",
                schema_name=schema_for_log,
                token_usage=usage,
                elapsed_ms=elapsed_ms,
                ok=ok,
                error=err,
            )
            if use_structured_composer:
                self._last_llm_log_seq = seq

    async def _llm_call_json_schema(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: str | None,
        effort: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> str:
        """Structured JSON via provider json_schema call."""
        if _log_llm_prompts_enabled():
            _debug_log_prompt(f"json_schema:{schema_name}", system_prompt, user_text)
        resolved_model = model or resolve_model(self.provider_name, "digest")
        pass_name = f"json_schema:{schema_name}"
        t0 = time.perf_counter()
        raw: str | None = None
        usage: int | None = None
        err: str | None = None
        ok = False
        try:
            raw, usage = await self._provider.json_schema_call(
                system_prompt,
                user_text,
                model=resolved_model,
                effort=effort,
                schema=schema,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )
            self._last_token_usage = usage
            logger.info(
                "LLM JSON pass %s token_usage≈%s",
                schema_name,
                self._last_token_usage,
            )
            ok = True
            return (raw or "").strip()
        except Exception as e:
            err = repr(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if ok and usage is None:
                logger.warning(
                    "LLM json_schema call completed without usage metadata "
                    "(schema_name=%s model=%s)",
                    schema_name,
                    resolved_model,
                )
            log_llm_call(
                pass_name=pass_name,
                system=system_prompt,
                user=user_text,
                response=(raw or "") if raw is not None else "",
                model=resolved_model,
                effort=effort,
                schema_name=schema_name,
                token_usage=usage,
                elapsed_ms=elapsed_ms,
                ok=ok,
                error=err,
            )
