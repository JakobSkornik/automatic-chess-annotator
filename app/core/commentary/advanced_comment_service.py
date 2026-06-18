from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
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
from app.core.commentary.forbidden_phrases import (
    forbidden_hit_count,
    forbidden_hit_strings,
)
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
    MoveQuality,
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


# ---------------------------------------------------------------------------
# Tier definitions – map key moment types to pipeline depth & effort
# ---------------------------------------------------------------------------
TIER_FULL_HIGH = {"steps": 1, "effort": "medium", "max_tokens": 900}
TIER_FULL_LOW = {"steps": 1, "effort": "low", "max_tokens": 700}
TIER_SINGLE = {"steps": 1, "effort": "low", "max_tokens": 500}

KEY_MOMENT_TIERS: dict[str, dict[str, Any]] = {
    "brilliant": TIER_FULL_HIGH,
    "blunder": TIER_FULL_HIGH,
    "critical_decision": TIER_FULL_HIGH,
    "mistake": TIER_FULL_LOW,
    "structural_transformation": TIER_FULL_LOW,
    "king_safety_crisis": TIER_FULL_LOW,
    "hidden_inflection": TIER_SINGLE,
    "great_move": TIER_SINGLE,
    "inaccuracy": TIER_SINGLE,
    "good_defense": TIER_SINGLE,
    "initiative_shift": TIER_SINGLE,
    "piece_activation": TIER_SINGLE,
    "missed_opportunity": TIER_SINGLE,
    "opening_transition": TIER_SINGLE,
    "endgame_transition": TIER_SINGLE,
}


def _detail_level_for_key_moment(move_event: MoveEvent) -> str:
    km = move_event.key_moment_type or ""
    if km in ("brilliant", "blunder", "critical_decision"):
        return "full"
    if km in ("mistake", "structural_transformation", "king_safety_crisis"):
        return "compact"
    ph = (move_event.phase or "").strip().lower()
    if (
        ph == "opening"
        and move_event.move_index <= 14
        and move_event.move_quality in (MoveQuality.BEST, MoveQuality.EXCELLENT)
        and km not in ("blunder", "mistake", "critical_decision", "brilliant")
    ):
        return "book"
    if (
        not km
        and move_event.best_move_uci
        and move_event.uci == move_event.best_move_uci
        and abs(move_event.eval_swing_cp or 0) < 15
    ):
        return "book"
    return "minimal"


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


def compute_rag_top_k(detail: str, phase: str | None) -> int:
    """Per-phase Top-K defaults; overridden by RAG_TOP_K_PHASE or global RAG_TOP_K."""
    if detail == "book":
        return 0
    ph_raw = (phase or "middlegame").strip().lower()
    if ph_raw in ("end", "endgame"):
        env_key = "RAG_TOP_K_ENDGAME"
    elif ph_raw == "opening":
        env_key = "RAG_TOP_K_OPENING"
    else:
        env_key = "RAG_TOP_K_MIDDLEGAME"
    for key in (env_key, "RAG_TOP_K"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(1, min(8, int(raw)))
            except ValueError:
                break
    if ph_raw in ("opening", "end", "endgame"):
        return 3 if detail != "minimal" else 2
    if detail == "full":
        return 3
    if detail == "compact":
        return 2
    return 1


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


def _non_ascii_letter_ratio(s: str) -> float:
    letters = [c for c in s if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 127) / len(letters)


_SPANISH_HINT_RE = re.compile(
    r"\b(que|las|los|del|por|para|una|unos|muy|como|esta|está|fueron|decidieron|fin)\b",
    re.I,
)


def _looks_like_section_header_line(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if re.match(r"^[AB]\)\s*\d", t):
        return True
    low = t.lower()
    return bool(low.startswith("now we will look") or low.startswith("here are two"))


def _rag_annotation_passes_filters(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    words = raw.split()
    if len(words) < 20:
        return False
    lines = raw.splitlines()
    if lines and _looks_like_section_header_line(lines[0]):
        return False
    if _non_ascii_letter_ratio(raw) > 0.03:
        return False
    if len(_SPANISH_HINT_RE.findall(raw)) >= 3:
        return False
    san_like = sum(
        1
        for w in words
        if re.match(r"^[NBRQK]?[a-h]?x?[a-h][1-8](?:=[NBRQ])?[+#]?$", w)
        or re.match(r"^[1-9]\d*\.\.\.?$", w)
    )
    return not san_like >= len(words) * 0.6


def _truncate_annotation_at_sentence(raw: str, cap: int) -> str:
    if len(raw) <= cap:
        return raw
    chunk = raw[:cap]
    for sep in ("\n", ". ", "! ", "? "):
        idx = chunk.rfind(sep)
        if idx > cap // 4:
            if sep == "\n":
                return chunk[:idx].rstrip()
            return chunk[: idx + 1].rstrip()
    return chunk.rstrip()


def _rag_min_score() -> float:
    raw = os.environ.get("RAG_MIN_SCORE", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 0.62


def _rag_idea_overlap_tokens(idea: str, snippets: list[str]) -> bool:
    blob = " ".join(snippets).lower()
    idea_tokens = {
        t for t in re.findall(r"[a-zA-Z0-9]+", (idea or "").lower()) if len(t) >= 3
    }
    if len(idea_tokens) < 3:
        return False
    blob_tokens = {t for t in re.findall(r"[a-zA-Z0-9]+", blob) if len(t) >= 3}
    return len(idea_tokens & blob_tokens) >= 3


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


def _strip_json_fence(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


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
        min_sc = _rag_min_score()
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
            if not _rag_annotation_passes_filters(ann_raw):
                continue
            ann = _truncate_annotation_at_sentence(ann_raw, cap)
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
        detail = _detail_level_for_key_moment(move_event)
        rag_top_k = compute_rag_top_k(detail, query.phase)

        rationale_pre = (
            rationale_override
            if rationale_override is not None
            else build_rationale(move_event, move_event.future_line)
        )
        motif_hint_keys = cap_motifs_for_prompt(
            move_event, rationale_pre.primary_motif_label
        )

        rag_retrieval_debug: dict[str, Any] = {}
        if detail == "book":
            if rag_results is None:
                rag_results = []
        elif rag_results is None:
            rag_results = await self._rag.retrieve(
                query, top_k=max(1, rag_top_k), retrieval_debug=rag_retrieval_debug
            )
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
            if not rag_results:
                logger.info("RAG: no results returned")
                km_w = move_event.key_moment_type or ""
                if (
                    km_w in ("critical_decision", "mistake", "blunder")
                    and rag_retrieval_debug
                ):
                    logger.info(
                        "RAG rejection debug ply=%s key_moment=%s %s",
                        move_event.ply,
                        km_w,
                        rag_retrieval_debug,
                    )

        if detail == "full":
            rag_max_chars = 1600
        elif detail in ("minimal", "book"):
            rag_max_chars = 500
        else:
            rag_max_chars = 800
        gd = getattr(game_context, "game_digest", None) or {}
        digest_archetype = (
            str(gd.get("strategic_archetype") or "other").strip() or "other"
        )

        try:
            _b_pre = chess.Board(move_event.fen_before)
            mover_side = "White" if _b_pre.turn == chess.WHITE else "Black"
            next_side = "Black" if _b_pre.turn == chess.WHITE else "White"
        except Exception:
            mover_side = "?"
            next_side = "?"
        move_label = f"{(move_event.ply + 1) // 2}{'.' if mover_side == 'White' else '...'} {move_event.san}"

        rationale = rationale_pre
        proj_detail = (
            "full"
            if detail == "full"
            else ("compact" if detail == "compact" else "minimal")
        )
        rationale_blob = json.dumps(
            prompt_projection(rationale, detail=proj_detail), indent=2
        )

        played_move_block = (
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

        rag_snippets_used: list[str] = []

        if detail == "book":
            opening_lines: list[str] = []
            if move_event.opening_name or move_event.opening_eco:
                opening_lines.append(
                    f"Opening: {move_event.opening_name or ''} ({move_event.opening_eco or ''})".strip()
                )
            opening_hdr = ("\n\n".join(opening_lines) + "\n\n") if opening_lines else ""
            show_motifs = (
                bool(move_event.tactical_motifs)
                and (move_event.eval_swing_cp or 0) != 0
            )
            mot_block = ""
            if show_motifs and motif_hint_keys:
                mot_block = (
                    "MOTIF HINTS (cap 3 tactical + 2 strategic; prose may name at most ONE motif key):\n"
                    + format_motif_digest_lines(motif_hint_keys)
                    + "\n\n"
                )
            dynamic_blocks = [
                opening_hdr + mot_block + played_move_block,
                "POSITION SUMMARY (anchor to this; do not invent lines):\n"
                + self._format_move_event_minimal(move_event, []),
            ]
            structured_text = "\n\n".join(dynamic_blocks)
            rag_hit_ct = 0
            compact_injected = None
        else:
            static_blocks: list[str] = []
            if gd:
                compact_ctx = compact_game_context_for_move(
                    gd, move_event.ply, current_phase=move_event.phase or ""
                )
                static_blocks.append(
                    "GAME CONTEXT (whole-game digest; use for tone and continuity, do not re-quote):\n"
                    + json.dumps(compact_ctx, indent=2)
                )

            pri = getattr(game_context, "prior_context_snippets", None) or []
            if pri:
                static_blocks.append(
                    "PRIOR_CONTEXT (continuity only; do not restate):\n"
                    + "\n".join(pri[-2:])
                )

            static_blocks.append(
                "MOTIF HINTS (cap 3 tactical + 2 strategic; prose may name at most ONE motif key):\n"
                + (
                    format_motif_digest_lines(motif_hint_keys)
                    if motif_hint_keys
                    else "(none)"
                )
            )

            dynamic_blocks: list[str] = [played_move_block]
            dynamic_blocks.append(
                "MOVE_RATIONALE_JSON (internal evidence; weave meaning into prose—never quote JSON keys verbatim):\n"
                + rationale_blob
            )

            if detail == "full":
                dynamic_blocks.append(
                    "POSITION AND ENGINE DATA (anchor commentary to this; do not invent lines):\n"
                    + self._format_move_event_block(move_event, motif_hint_keys)
                )
                pv_block = self._format_pv_position_comparison(move_event, analyzed_row)
                if pv_block:
                    dynamic_blocks.append(pv_block)
                fl_block = self._format_future_line_block(move_event)
                if fl_block:
                    dynamic_blocks.append(fl_block)
            else:
                dynamic_blocks.append(
                    "POSITION SUMMARY (anchor to this; do not invent lines):\n"
                    + self._format_move_event_minimal(move_event, motif_hint_keys)
                )

            rb, rag_snippets_used = self._format_rag_block(
                rag_results or [], max_chars=rag_max_chars
            )
            if rb:
                dynamic_blocks.append(rb)

            structured_text = (
                "\n\n".join(static_blocks)
                + DYNAMIC_SECTION_SENTINEL
                + "\n\n".join(dynamic_blocks)
            )
            rag_hit_ct = len(rag_results or [])
            compact_injected = (
                compact_game_context_for_move(
                    gd, move_event.ply, current_phase=move_event.phase or ""
                )
                if gd
                else None
            )

        km = move_event.key_moment_type
        cat = move_event.move_category.value if move_event.move_category else None
        tier_base = dict(
            KEY_MOMENT_TIERS.get(move_event.key_moment_type or "", TIER_SINGLE)
        )
        if composer_effort:
            tier_base["effort"] = composer_effort
        tier_effort = str(tier_base.get("effort", TIER_SINGLE["effort"]))
        passes, system_prompts = build_planned_llm_passes_and_system_prompts(
            km,
            cat,
            tier_effort,
            detail_level=detail,
            strategic_archetype=digest_archetype if detail != "book" else None,
        )
        debug_dict: dict[str, Any] = {
            "move_category": cat,
            "key_moment_type": km,
            "tier": tier_base,
            "detail_level": detail,
            "motif_hint_keys": motif_hint_keys,
            "rag_top_k": rag_top_k,
            "rag_hit_count": rag_hit_ct,
            "detected_motif_labels": motif_hint_keys,
            "rag_query": query.model_dump(),
            "rationale": rationale.model_dump(),
            "rag_retrieval_debug": rag_retrieval_debug or None,
            "system_prompts": system_prompts,
            "user_text": structured_text,
            "passes": passes,
            "token_usage_total": None,
            "tokens_by_pass": {},
            "elapsed_ms": None,
            "composer_named_motifs": [],
            "commentary_audit": None,
            "game_digest": gd if gd else None,
            "game_context_injected": compact_injected,
            "rag_snippets_used": rag_snippets_used,
            "composer_forcing_pv_bracket": _composer_forcing_pv_bracket(move_event),
        }
        logger.info(
            "build_event_llm_input: move_category=%s key_moment_type=%s detail=%s",
            cat,
            km,
            detail,
        )
        _debug_log_prompt("build_event_input", "", structured_text)
        return structured_text, rag_results or [], debug_dict

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
            return (
                "AI comments are disabled. Set OPENAI_API_KEY (OpenAI) or "
                "ANTHROPIC_API_KEY (Anthropic)."
            )
        t0 = time.perf_counter()
        tokens_by_pass: dict[str, int] = {}
        tier = KEY_MOMENT_TIERS.get(key_moment_type or "", TIER_SINGLE)
        tier_effort = effort or tier["effort"]
        tier_max_tokens = int(tier.get("max_tokens", TIER_SINGLE["max_tokens"]))
        cat_key = move_category or MoveCategory.POSITIONAL.value
        role = composer_role_key(cat_key)
        detail_lvl = (llm_debug or {}).get("detail_level") or ""
        gd_ctx = (llm_debug or {}).get("game_digest") or {}
        digest_arch = (
            str(gd_ctx.get("strategic_archetype") or "other").strip() or "other"
        )
        if detail_lvl == "book":
            composer_prompt = COMPOSER_ROLE_PROMPTS.get(
                role, POSITIONAL_PLAN_COMPOSER_PROMPT
            )
        else:
            composer_prompt = composer_system_prompt(role, archetype=digest_arch)
        if _log_llm_prompts_enabled():
            logger.info(
                "analyze_and_compose_raw_text: branch=single move_category=%s key_moment_type=%s composer_role=%s",
                move_category,
                key_moment_type,
                role,
            )
            logger.info("composer system prompt selected: composer_%s", role)

        prose: str = ""
        composer_named: list[str] = []

        mdl = model or resolve_model(self.provider_name, "composer")
        rag_snip = list((llm_debug or {}).get("rag_snippets_used") or [])
        allow = [
            str(x)
            for x in (llm_debug or {}).get("motif_hint_keys") or []
            if str(x).strip()
        ]
        forcing_bracket = (llm_debug or {}).get("composer_forcing_pv_bracket")
        prose, composer_named, composer_extra = await self._run_composer_segments(
            composer_prompt,
            structured_text,
            model=mdl,
            effort=tier_effort,
            prompt_name="composer_single",
            max_output_tokens=tier_max_tokens,
            fen_before=fen_before,
            fen_after=fen_after,
            rag_snippets=rag_snip,
            motif_hint_allowlist=allow,
            forcing_pv_bracket=forcing_bracket
            if isinstance(forcing_bracket, str)
            else None,
        )
        retry_forbidden = os.environ.get(
            "FORBIDDEN_PHRASE_RETRY", ""
        ).strip().lower() in ("1", "true")
        if retry_forbidden and prose.strip():
            hits = forbidden_hit_strings(prose)
            if hits:
                prose2, named2, extra2 = await self._run_composer_segments(
                    composer_prompt,
                    structured_text
                    + "\n\nRevise your commentary JSON: remove these banned phrases entirely: "
                    + "; ".join(hits),
                    model=mdl,
                    effort=tier_effort,
                    prompt_name="composer_single_forbidden_retry",
                    max_output_tokens=tier_max_tokens,
                    fen_before=fen_before,
                    fen_after=fen_after,
                    rag_snippets=rag_snip,
                    motif_hint_allowlist=allow,
                    forcing_pv_bracket=forcing_bracket
                    if isinstance(forcing_bracket, str)
                    else None,
                )
                if prose2.strip() and forbidden_hit_count(prose2) < forbidden_hit_count(
                    prose
                ):
                    prose, composer_named, composer_extra = prose2, named2, extra2
        tokens_by_pass["composer"] = self._last_token_usage

        total_tokens = sum(tokens_by_pass.values())
        if total_tokens > 4000:
            logger.warning(
                "LLM token budget high for move: ~%s total (warn threshold 4000)",
                total_tokens,
            )

        if llm_debug is not None:
            llm_debug["token_usage_total"] = total_tokens
            llm_debug["tokens_by_pass"] = dict(tokens_by_pass)
            llm_debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            llm_debug["composer_named_motifs"] = composer_named
            llm_debug["composer_better_alternative"] = composer_extra.get(
                "better_alternative", ""
            )
            llm_debug["composer_rag_idea_used"] = composer_extra.get(
                "rag_idea_used", ""
            )
            llm_debug["composer_rag_applied"] = composer_extra.get("rag_applied")
            det = llm_debug.get("detected_motif_labels") or []
            rag_hit = bool(llm_debug.get("rag_hit_count"))
            llm_debug["commentary_audit"] = compute_commentary_audit(
                prose,
                det,
                rag_had_hits=rag_hit,
                rag_applied=composer_extra.get("rag_applied"),
            )
        return prose

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
        fb = fen_before or chess.Board().fen()
        fa = fen_after or chess.Board().fen()
        rag_list = list(rag_snippets or [])
        last_raw: str | None = None
        last_plain_len: int | None = None
        empty_extras: dict[str, Any] = {
            "better_alternative": "",
            "rag_idea_used": "",
            "rag_applied": False,
        }
        for attempt in range(2):
            user_for_attempt = user_text
            if attempt == 1:
                user_for_attempt = "Keep the answer under 70 words.\n\n" + user_text
            try:
                raw = await self._llm_call(
                    structured_system,
                    user_for_attempt,
                    model=model,
                    effort=effort,
                    use_structured_composer=True,
                    prompt_name=prompt_name if attempt == 0 else f"{prompt_name}_retry",
                    max_output_tokens=max_output_tokens,
                )
                last_raw = raw
                if raw:
                    text = _strip_json_fence(raw)
                    obj = json.loads(text)
                    named_raw = obj.get("named_motifs") or []
                    named = [
                        str(x) for x in named_raw if isinstance(x, str) and x.strip()
                    ]
                    allow_set = {
                        str(x) for x in (motif_hint_allowlist or []) if str(x).strip()
                    }
                    if allow_set:
                        named = [x for x in named if x in allow_set][:1]
                    plain = str(obj.get("text") or "").strip()
                    ba = str(obj.get("better_alternative") or "").strip()
                    riu = str(obj.get("rag_idea_used") or "").strip()
                    rap = bool(obj.get("rag_applied", False))
                    extras = {
                        "better_alternative": ba,
                        "rag_idea_used": riu,
                        "rag_applied": rap,
                    }
                    if not rag_list and extras.get("rag_applied"):
                        extras["rag_applied"] = False
                        extras["rag_idea_used"] = ""
                    if rag_list and extras.get("rag_applied"):
                        if not _rag_idea_overlap_tokens(
                            extras.get("rag_idea_used", ""), rag_list
                        ):
                            extras["rag_applied"] = False
                            extras["rag_idea_used"] = ""
                    last_plain_len = len(plain)
                    base = sanitize_text(auto_tokenize(plain, fb, fa)) if plain else ""
                    if base.strip():
                        out = base.rstrip()
                        if ba and ba.lower() not in out.lower():
                            out = f"{out} {ba}".strip()
                        if forcing_pv_bracket and not re.search(
                            r"\[pv\s*:", out.lower()
                        ):
                            out = f"{out} {forcing_pv_bracket}".strip()
                        if out.strip():
                            ref = self._last_llm_log_seq
                            if ref is not None:
                                append_postcheck(
                                    ref_seq=ref,
                                    payload={
                                        "rag_applied": extras.get("rag_applied"),
                                        "rag_idea_used": extras.get("rag_idea_used"),
                                        "named_motifs": named,
                                    },
                                )
                            return out, named, extras
                    logger.warning(
                        "Structured composer empty prose after parse (attempt %s): "
                        "raw_preview=%.500s plain_len=%s",
                        attempt + 1,
                        text,
                        last_plain_len,
                    )
            except Exception as e:
                logger.warning(
                    "Structured composer failed (attempt %s): %s", attempt + 1, e
                )
        preview = (last_raw or "")[:500]
        plen = last_plain_len
        if plen is None and last_raw:
            try:
                t2 = _strip_json_fence(last_raw)
                o2 = json.loads(t2)
                plen = len(str(o2.get("text") or "").strip())
            except Exception:
                pass
        logger.warning(
            "Structured composer returned empty after retries; raw_preview=%.500s plain_len=%s",
            preview,
            plen,
        )
        return "", [], empty_extras

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
