from __future__ import annotations

import chess
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.core.commentary.annotation_tokens import auto_tokenize
from app.core.commentary.rag_retriever import (
    RAGRetriever,
    RAGResult,
    build_rag_query,
)
from app.core.commentary.tantivy_positional_retriever import get_default_retriever
from app.core.commentary.move_rationale import build_rationale
from app.models.chess_events import (
    AnalyzedMoveData,
    Episode,
    GameAnalysisContext,
    MoveCategory,
    MoveEvent,
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
_UNICODE_REPLACEMENTS: List[Tuple[str, str]] = [
    ("\u2018", "'"),   # left single quote
    ("\u2019", "'"),   # right single quote
    ("\u201C", '"'),   # left double quote
    ("\u201D", '"'),   # right double quote
    ("\u2014", " -- "),  # em dash
    ("\u2013", " - "),   # en dash
    ("\u2011", "-"),     # non-breaking hyphen
    ("\u2010", "-"),     # hyphen
    ("\u2026", "..."),   # horizontal ellipsis
    ("\u00A0", " "),     # non-breaking space
    ("\u200B", ""),      # zero-width space
]


def sanitize_text(text: str) -> str:
    """Replace common Unicode typographic characters with ASCII equivalents."""
    for src, dst in _UNICODE_REPLACEMENTS:
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Tier definitions – map key moment types to pipeline depth & effort
# ---------------------------------------------------------------------------
TIER_FULL_HIGH = {"steps": 2, "effort": "medium", "max_tokens": 180}
TIER_FULL_LOW = {"steps": 2, "effort": "low", "max_tokens": 120}
TIER_SINGLE = {"steps": 1, "effort": "low", "max_tokens": 80}

KEY_MOMENT_TIERS: Dict[str, Dict[str, Any]] = {
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

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

POSITION_NARRATOR_PROMPT = (
    "You are a chess grandmaster. Given engine data and position features (not a board image),\n"
    "produce a compact strategic snapshot. Be terse: one idea per field.\n\n"
    "Rules:\n"
    "- Base conclusions on the FEATURES and FEN provided, not inventing tactics.\n"
    "- Each strengths/weaknesses list: at most 1 short phrase.\n"
    "- key_squares: at most 2 entries, square + half-line why.\n"
    "- plans_white / plans_black: one short clause each.\n"
    "- Output strictly valid JSON, ASCII only.\n\n"
    "Output format:\n"
    "{\n"
    "  \"character\": \"open | closed | tactical | strategic | transitional (pick one)\",\n"
    "  \"white_strengths\": [\"one phrase\"],\n"
    "  \"white_weaknesses\": [\"one phrase\"],\n"
    "  \"black_strengths\": [\"one phrase\"],\n"
    "  \"black_weaknesses\": [\"one phrase\"],\n"
    "  \"key_squares\": [\"e5 — ...\", \"d4 — ...\"],\n"
    "  \"plans_white\": \"one clause\",\n"
    "  \"plans_black\": \"one clause\"\n"
    "}"
)

EPISODE_NARRATIVE_PROMPT = (
    "You are a grandmaster chess commentator narrating the story of a game phase.\n"
    "Given: move sequence with evaluations, tactical motifs, dominant theme, and eval trend,\n"
    "write 2-4 sentences that explain the strategic narrative of this phase.\n"
    "Cover: each side's plan, how the position evolved, and where the critical idea or turning point was.\n"
    "Reference specific moves and squares. Output strictly valid JSON:\n"
    "{\"commentary\": \"2-4 sentences\"}"
)

GAME_NARRATIVE_PROMPT = (
    "You are a grandmaster chess commentator. Given episode summaries and key results,\n"
    "write a 3-5 sentence overview of the whole game: opening character, critical phase,\n"
    "and how the result arose. No move-by-move listing. Output strictly valid JSON:\n"
    "{\"commentary\": \"3-5 sentences\"}"
)

GAME_DIGEST_PROMPT = (
    "You are a chess coach. Given a compact move log with evaluations and key-moment tags, produce a "
    "structured digest of the whole game. Output strict JSON; no markdown. Be specific but compact.\n"
)

GAME_DIGEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_story": {"type": "string"},
        "opening_character": {"type": "string"},
        "phase_story": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "phase": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["phase", "summary"],
                "additionalProperties": False,
            },
        },
        "turning_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ply": {"type": "integer"},
                    "san": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["ply", "san", "why"],
                "additionalProperties": False,
            },
        },
        "winning_side_plan": {"type": "string"},
        "losing_side_mistakes": {"type": "string"},
    },
    "required": [
        "overall_story",
        "opening_character",
        "phase_story",
        "turning_points",
        "winning_side_plan",
        "losing_side_mistakes",
    ],
    "additionalProperties": False,
}


def build_game_digest_input(context: GameAnalysisContext) -> str:
    """Compact text for whole-game digest LLM pass."""
    hdr = context.metadata or {}
    w = hdr.get("white") or hdr.get("White", "?")
    b = hdr.get("black") or hdr.get("Black", "?")
    res = hdr.get("result") or hdr.get("Result", "*")
    we = hdr.get("whiteElo")
    be = hdr.get("blackElo")
    elo_w = f" ({we})" if we is not None else ""
    elo_b = f" ({be})" if be is not None else ""
    lines: List[str] = [
        f"WHITE: {w}{elo_w} vs BLACK: {b}{elo_b}",
        f"RESULT: {res}  OPENING: {context.opening_name or ''} ({context.opening_eco or ''})",
        "MOVES (ply san eval_before->eval_after swing key_moment/category motifs):",
    ]
    for me in context.move_events:
        eb = f"{me.eval_before_cp / 100.0:+.2f}" if me.eval_before_cp is not None else "?"
        ea = f"{me.eval_after_cp / 100.0:+.2f}" if me.eval_after_cp is not None else "?"
        sw = f"{me.eval_swing_cp / 100.0:+.2f}" if me.eval_swing_cp is not None else ""
        km = me.key_moment_type or ""
        cat = me.move_category.value if me.move_category else ""
        mot = ",".join(m.value for m in me.tactical_motifs[:3])
        mq = me.move_quality.value if me.move_quality else ""
        line = f"{me.ply:>3} {me.san:<7} {eb}->{ea} {sw} {km}/{cat} mq={mq} {mot}".rstrip()
        lines.append(line)
    return "\n".join(lines)


def compact_game_context_for_move(digest: Dict[str, Any], current_ply: int) -> Dict[str, Any]:
    """Subset of digest for per-move prompt: nearby turning points only."""
    tps = digest.get("turning_points") or []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for tp in tps:
        if not isinstance(tp, dict):
            continue
        try:
            p = int(tp.get("ply", 0))
        except (TypeError, ValueError):
            continue
        scored.append((abs(float(p - current_ply)), tp))
    scored.sort(key=lambda x: x[0])
    near = [x[1] for x in scored[:2]]
    return {
        "overall_story": digest.get("overall_story", ""),
        "opening_character": digest.get("opening_character", ""),
        "turning_points_nearby": near,
        "phase_story": digest.get("phase_story", []),
        "winning_side_plan": digest.get("winning_side_plan", ""),
        "losing_side_mistakes": digest.get("losing_side_mistakes", ""),
    }


SEGMENT_OUTPUT_INSTRUCTIONS = (
    "\n\nSTYLE: Write as a human annotator describing the move in one connected paragraph of 2-4 sentences. "
    "Do NOT output labeled lists or fragments. Never start a clause with labels like "
    "\"Immediate:\", \"Future:\", \"Counterfactual:\", \"Eval swing:\", \"Tactic:\", \"Consequence:\". "
    "Do not repeat raw engine numbers beyond at most one signed eval in pawns when it carries meaning. "
    "Weave cause-and-effect naturally: what the move does now, why it matters, and the alternative only "
    "if it changes the story. Use tokens for SAN moves, squares/files, and evals per the format below.\n\n"
    "OUTPUT FORMAT: Respond with JSON only (no markdown fences). "
    "The object must have a \"segments\" array in reading order. "
    "Each segment has \"segment_kind\" (\"text\" or \"token\"), \"prose\", \"token_type\", \"token_content\" (all strings). "
    "For prose: segment_kind=\"text\", put words in \"prose\", use empty strings for token_type and token_content. "
    "For an interactive UI token: segment_kind=\"token\", token_type one of move|square|file|eval|piece|pv, "
    "token_content the value (e.g. Nf3, e5, d, +0.25, Nd7, or space-separated SAN for pv), prose empty. "
    "Whenever the commentary refers to a file such as \"the e-file\", \"the d-file\", \"a-file\" etc., emit a "
    "token segment with token_type=\"file\" and token_content the single letter a-h (never \"e-file\" in prose). "
    "Never put SAN moves or signed eval numbers in text segments — use token segments for those. "
    "The server converts tokens to [type:content] for the UI."
)

# Per-category composer prompts: flowing prose; never quote rationale JSON field names.
TACTICAL_COMPOSER_PROMPT = (
    "You annotate a TACTICAL moment as flowing prose. Narrate what the move does, the short forcing idea, "
    "and its practical consequence in one paragraph of 2-3 sentences. Reference the rationale and engine "
    "data but never quote JSON field names or emit labeled lists. Do NOT list feature deltas or mobility numbers.\n"
    "Max 50 words of prose in segments.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

POSITIONAL_COMPOSER_PROMPT = (
    "You annotate a POSITIONAL / quiet plan move as flowing prose. Describe long-term structure and how "
    "this move fits the plan versus alternatives, in 2-4 connected sentences. Weave alternatives naturally "
    "when they matter; do not enumerate engine feature tables or quote rationale field names.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

DEFENSIVE_COMPOSER_PROMPT = (
    "You annotate a DEFENSIVE resource as flowing prose. Describe the danger that existed and how this move "
    "answers it, in 2-3 sentences. Never use labeled sub-headings or quote JSON keys.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

PROPHYLACTIC_COMPOSER_PROMPT = (
    "You annotate a PROPHYLACTIC move as flowing prose. Explain which opponent idea was restrained and why "
    "this move stops it, in 2-3 sentences. Integrate best-line hints naturally; no labeled lists.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

FORCING_COMPOSER_PROMPT = (
    "You annotate a FORCING sequence (check, capture, or sharp threat) as flowing prose. Keep the concrete "
    "sequence clear in 2-3 sentences without bullet labels or \"Immediate:/Future:\" fragments.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

BOOK_COMPOSER_PROMPT = (
    "You annotate an OPENING / book move as brief flowing prose (1-2 sentences): development or theory note only.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

INACCURACY_COMPOSER_PROMPT = (
    "You annotate a small slip (inaccuracy) as flowing prose: what the engine prefers instead and why, in plain "
    "chess terms, in 2 short sentences. No labeled checklist.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

CRITICAL_COMPOSER_PROMPT = (
    "You annotate a CRITICAL error or turning point as flowing prose. Give one clear strategic reason the "
    "engine line matters, in 2-3 sentences; avoid dumping numbers or using fragment labels.\n"
) + SEGMENT_OUTPUT_INSTRUCTIONS

CATEGORY_COMPOSER_PROMPTS: Dict[str, str] = {
    MoveCategory.TACTICAL.value: TACTICAL_COMPOSER_PROMPT,
    MoveCategory.POSITIONAL.value: POSITIONAL_COMPOSER_PROMPT,
    MoveCategory.DEFENSIVE.value: DEFENSIVE_COMPOSER_PROMPT,
    MoveCategory.PROPHYLACTIC.value: PROPHYLACTIC_COMPOSER_PROMPT,
    MoveCategory.FORCING.value: FORCING_COMPOSER_PROMPT,
    MoveCategory.BOOK.value: BOOK_COMPOSER_PROMPT,
    MoveCategory.INACCURACY.value: INACCURACY_COMPOSER_PROMPT,
    MoveCategory.CRITICAL.value: CRITICAL_COMPOSER_PROMPT,
}

MOTIF_SYNTH_PROMPT = (
    "From MOVE_RATIONALE_JSON and engine context, extract motif labels only. Output strict JSON.\n"
    "Fields: primary_motif (string), supporting_motifs (array of strings), named_idea (short chess name or empty).\n"
)

REASONING_JSON_PROMPT = (
    "Given motif JSON + rationale + engine context, output JSON with keys: "
    "why (string), risk (string), plan (string), counterplay (string). "
    "Each value one or two clauses, grounded in data; no feature enumeration.\n"
)

MOTIF_SYNTH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_motif": {"type": "string"},
        "supporting_motifs": {"type": "array", "items": {"type": "string"}},
        "named_idea": {"type": "string"},
    },
    "required": ["primary_motif", "supporting_motifs", "named_idea"],
    "additionalProperties": False,
}

REASONING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "why": {"type": "string"},
        "risk": {"type": "string"},
        "plan": {"type": "string"},
        "counterplay": {"type": "string"},
    },
    "required": ["why", "risk", "plan", "counterplay"],
    "additionalProperties": False,
}

# JSON Schema for OpenAI Structured Outputs (strict). All segment fields required per item.
COMPOSER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_kind": {"type": "string", "enum": ["text", "token"]},
                    "prose": {"type": "string"},
                    "token_type": {
                        "type": "string",
                        "enum": ["", "move", "square", "file", "eval", "piece", "pv"],
                    },
                    "token_content": {"type": "string"},
                },
                "required": ["segment_kind", "prose", "token_type", "token_content"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["segments"],
    "additionalProperties": False,
}


def build_planned_llm_passes_and_system_prompts(
    key_moment_type: Optional[str],
    move_category: Optional[str],
    tier_effort: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Planned passes and system prompts for WS debug (mirrors analyze_and_compose_raw_text)."""
    km = key_moment_type or ""
    tier = KEY_MOMENT_TIERS.get(km, TIER_SINGLE)
    cat_key = move_category or MoveCategory.POSITIONAL.value
    composer_prompt = CATEGORY_COMPOSER_PROMPTS.get(
        cat_key, CATEGORY_COMPOSER_PROMPTS[MoveCategory.POSITIONAL.value]
    )
    composer_name = f"composer_{cat_key}"
    use_two_steps = tier["steps"] == 2
    is_full_high = km in (
        "brilliant",
        "blunder",
        "critical_decision",
    ) and tier.get("effort") == "medium" and tier.get("steps") == 2

    passes: List[Dict[str, str]] = []
    system_prompts: List[Dict[str, str]] = []

    if use_two_steps and is_full_high:
        passes.append({"name": "motif_synth", "effort": "low"})
        system_prompts.append({"name": "MOTIF_SYNTH_PROMPT", "text": MOTIF_SYNTH_PROMPT})
        passes.append({"name": "reasoning_pass", "effort": "medium"})
        system_prompts.append({"name": "REASONING_JSON_PROMPT", "text": REASONING_JSON_PROMPT})
        passes.append({"name": "composer_full_high", "effort": tier_effort})
        system_prompts.append({"name": composer_name, "text": composer_prompt})
    elif use_two_steps:
        passes.append({"name": "position_narrator", "effort": tier_effort})
        system_prompts.append({"name": "POSITION_NARRATOR_PROMPT", "text": POSITION_NARRATOR_PROMPT})
        passes.append({"name": "composer_two_step", "effort": tier_effort})
        system_prompts.append({"name": composer_name, "text": composer_prompt})
    else:
        passes.append({"name": "composer_single", "effort": tier_effort})
        system_prompts.append({"name": composer_name, "text": composer_prompt})
    return passes, system_prompts


def assemble_commentary_from_segments(obj: Dict[str, Any]) -> str:
    """Turn structured segment JSON into inline [type:content] commentary text."""
    parts: List[str] = []
    for seg in obj.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        sk = seg.get("segment_kind")
        if sk == "text":
            parts.append(str(seg.get("prose", "")))
        elif sk == "token":
            tt = (seg.get("token_type") or "").strip()
            tc = str(seg.get("token_content", "")).strip()
            if tt and tc:
                parts.append(f"[{tt}:{tc}]")
    return sanitize_text("".join(parts))


def commentary_from_composer_parsed(obj: Dict[str, Any]) -> str:
    """Assemble inline [type:content] text from structured segment JSON only."""
    if isinstance(obj.get("segments"), list):
        return assemble_commentary_from_segments(obj)
    return ""


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
    def __init__(self, rag_retriever: Optional[RAGRetriever] = None) -> None:
        self.client = AsyncOpenAI()
        self._rag: RAGRetriever = rag_retriever or get_default_retriever()
        self._last_token_usage: int = 0

    @staticmethod
    def _format_move_event_block(move_event: MoveEvent) -> str:
        fe = move_event
        parts: List[str] = []
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
            bev = fe.best_move_eval_cp / 100.0 if fe.best_move_eval_cp is not None else None
            parts.append(
                f"Best engine move: {fe.best_move_san}"
                + (f" (eval {bev:+.2f})" if bev is not None else "")
            )
        for pv in fe.pv_lines[:3]:
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
        if fe.tactical_motifs:
            parts.append(
                "Tactical motifs: "
                + ", ".join(m.value for m in fe.tactical_motifs)
            )
        if fe.strategic_motifs:
            parts.append(
                "Strategic motifs: "
                + ", ".join(m.value for m in fe.strategic_motifs)
            )
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
        analyzed_row: Optional[AnalyzedMoveData],
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
            parts.append(f"Played-line SAN (first plies): {' '.join(fl.played_line_san)}")
        if fl.best_line_san:
            parts.append(f"Best-line SAN (first plies): {' '.join(fl.best_line_san)}")
        if fl.played_leaf_eval_cp is not None:
            parts.append(f"Eval at played-line leaf: {fl.played_leaf_eval_cp / 100.0:+.2f} pawns")
        if fl.best_leaf_eval_cp is not None:
            parts.append(f"Eval at best-line leaf: {fl.best_leaf_eval_cp / 100.0:+.2f} pawns")
        if fl.eval_gap_cp is not None:
            parts.append(f"Leaf eval gap (best - played): {fl.eval_gap_cp / 100.0:+.2f} pawns")
        if fl.feature_deltas:
            fd = ", ".join(f"{k}={v:+.3f}" for k, v in list(fl.feature_deltas.items())[:8])
            parts.append(f"Feature deltas (best vs played leaf): {fd}")
        if fl.played_targets:
            parts.append(f"Recurring destination squares (played line): {', '.join(fl.played_targets)}")
        if fl.best_targets:
            parts.append(f"Recurring destination squares (best line): {', '.join(fl.best_targets)}")
        return "\n".join(parts)

    @staticmethod
    def _format_rag_block(results: List[RAGResult]) -> str:
        if not results:
            return ""
        lines = [
            "REFERENCE EXAMPLES FROM MASTER GAMES:",
            "(Examples may quote master annotations from a few plies later in similar games; "
            "use them as plan inspiration, not as the current position's evaluation.)",
        ]
        for i, r in enumerate(results, 1):
            tags = ", ".join(f"{k}={v}" for k, v in list(r.relevance_tags.items())[:6])
            lines.append(f"[{i}] Source: {r.source}")
            if tags:
                lines.append(f"    Matched: {tags}")
            lines.append(f"    {r.annotation_text[:1200]}")
        return "\n".join(lines)

    async def build_event_llm_input(
        self,
        move_event: MoveEvent,
        episode: Optional[Episode],
        game_context: GameAnalysisContext,
        analyzed_row: Optional[AnalyzedMoveData] = None,
        composer_effort: Optional[str] = None,
    ) -> Tuple[str, List[RAGResult], Dict[str, Any]]:
        query = build_rag_query(move_event, episode)
        rag_results = await self._rag.retrieve(query, top_k=2)
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
        blocks: List[str] = []
        gd = getattr(game_context, "game_digest", None) or {}
        if gd:
            compact_ctx = compact_game_context_for_move(gd, move_event.ply)
            blocks.append(
                "GAME CONTEXT (whole-game digest; use for tone and continuity, do not re-quote):\n"
                + json.dumps(compact_ctx, indent=2)
            )
        rb = self._format_rag_block(rag_results)
        if rb:
            blocks.append(rb)
        rationale = build_rationale(move_event, move_event.future_line)
        blocks.append(
            "MOVE_RATIONALE_JSON (internal evidence; weave into prose, never quote field names or bullet them):\n"
            + rationale.model_dump_json(indent=2)
        )
        blocks.append(
            "POSITION AND ENGINE DATA (anchor commentary to this; do not invent lines):\n"
            + self._format_move_event_block(move_event)
        )
        pv_block = self._format_pv_position_comparison(move_event, analyzed_row)
        if pv_block:
            blocks.append(pv_block)
        fl_block = self._format_future_line_block(move_event)
        if fl_block:
            blocks.append(fl_block)
        structured_text = "\n\n".join(blocks)
        km = move_event.key_moment_type
        cat = move_event.move_category.value if move_event.move_category else None
        tier_base = dict(KEY_MOMENT_TIERS.get(move_event.key_moment_type or "", TIER_SINGLE))
        if composer_effort:
            tier_base["effort"] = composer_effort
        tier_effort = str(tier_base.get("effort", TIER_SINGLE["effort"]))
        passes, system_prompts = build_planned_llm_passes_and_system_prompts(
            km, cat, tier_effort
        )
        debug_dict: Dict[str, Any] = {
            "move_category": cat,
            "key_moment_type": km,
            "tier": tier_base,
            "rag_query": query.model_dump(),
            "rationale": rationale.model_dump(),
            "system_prompts": system_prompts,
            "user_text": structured_text,
            "passes": passes,
            "token_usage_total": None,
            "game_digest": gd if gd else None,
            "game_context_injected": compact_game_context_for_move(gd, move_event.ply) if gd else None,
        }
        logger.info(
            "build_event_llm_input: move_category=%s key_moment_type=%s",
            cat,
            km,
        )
        _debug_log_prompt("build_event_input", "", structured_text)
        return structured_text, rag_results, debug_dict

    async def analyze_and_compose_event(
        self,
        move_event: MoveEvent,
        episode: Optional[Episode],
        game_context: GameAnalysisContext,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        key_moment_type: Optional[str] = None,
        analyzed_row: Optional[AnalyzedMoveData] = None,
    ) -> Tuple[str, List[RAGResult], Dict[str, Any]]:
        structured_text, rag_results, debug_dict = await self.build_event_llm_input(
            move_event,
            episode,
            game_context,
            analyzed_row=analyzed_row,
            composer_effort=effort,
        )
        text = await self.analyze_and_compose_raw_text(
            structured_text,
            model=model,
            effort=effort,
            key_moment_type=key_moment_type or move_event.key_moment_type,
            move_category=move_event.move_category.value if move_event.move_category else None,
            llm_debug=debug_dict,
        )
        text = auto_tokenize(text, move_event.fen_before, move_event.fen_after)
        return text, rag_results, debug_dict

    async def analyze_and_compose_raw_text(
        self,
        structured_text: str,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        key_moment_type: Optional[str] = None,
        move_category: Optional[str] = None,
        llm_debug: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.client.api_key:
            if llm_debug is not None:
                llm_debug["token_usage_total"] = None
            return "AI comments are disabled. Please set the OPENAI_API_KEY environment variable."
        tier = KEY_MOMENT_TIERS.get(key_moment_type or "", TIER_SINGLE)
        tier_effort = effort or tier["effort"]
        cat_key = move_category or MoveCategory.POSITIONAL.value
        composer_prompt = CATEGORY_COMPOSER_PROMPTS.get(
            cat_key, CATEGORY_COMPOSER_PROMPTS[MoveCategory.POSITIONAL.value]
        )
        use_two_steps = tier["steps"] == 2
        is_full_high = key_moment_type in (
            "brilliant",
            "blunder",
            "critical_decision",
        ) and tier.get("effort") == "medium" and tier.get("steps") == 2

        if _log_llm_prompts_enabled():
            logger.info(
                "analyze_and_compose_raw_text: branch=%s move_category=%s key_moment_type=%s",
                "full_high_three_pass"
                if (use_two_steps and is_full_high)
                else ("two_step" if use_two_steps else "single"),
                move_category,
                key_moment_type,
            )
            logger.info(
                "composer system prompt selected: %s",
                {
                    MoveCategory.TACTICAL.value: "TACTICAL_COMPOSER_PROMPT",
                    MoveCategory.POSITIONAL.value: "POSITIONAL_COMPOSER_PROMPT",
                    MoveCategory.DEFENSIVE.value: "DEFENSIVE_COMPOSER_PROMPT",
                    MoveCategory.PROPHYLACTIC.value: "PROPHYLACTIC_COMPOSER_PROMPT",
                    MoveCategory.FORCING.value: "FORCING_COMPOSER_PROMPT",
                    MoveCategory.BOOK.value: "BOOK_COMPOSER_PROMPT",
                    MoveCategory.INACCURACY.value: "INACCURACY_COMPOSER_PROMPT",
                    MoveCategory.CRITICAL.value: "CRITICAL_COMPOSER_PROMPT",
                }.get(cat_key, "POSITIONAL_COMPOSER_PROMPT"),
            )

        if use_two_steps and is_full_high:
            total_tokens = 0
            motif_raw = await self._llm_call_json_schema(
                MOTIF_SYNTH_PROMPT,
                structured_text,
                model=model,
                effort="low",
                schema=MOTIF_SYNTH_SCHEMA,
                schema_name="motif_synth",
            )
            total_tokens += self._last_token_usage
            reasoning_raw = await self._llm_call_json_schema(
                REASONING_JSON_PROMPT,
                f"MOTIF_SYNTH_JSON:\n{motif_raw}\n\n{structured_text}",
                model=model,
                effort="medium",
                schema=REASONING_SCHEMA,
                schema_name="reasoning_pass",
            )
            total_tokens += self._last_token_usage
            prose = await self._run_composer_segments(
                composer_prompt,
                f"MOTIF_SYNTH_JSON:\n{motif_raw}\nREASONING_JSON:\n{reasoning_raw}\n\n{structured_text}",
                model=model,
                effort=tier_effort,
                prompt_name="composer_full_high",
            )
            total_tokens += self._last_token_usage
            if llm_debug is not None:
                llm_debug["token_usage_total"] = total_tokens
            if total_tokens > 4000:
                logger.warning(
                    "LLM token budget high for multi-pass move: ~%s total (warn threshold 4000)",
                    total_tokens,
                )
            return prose

        if use_two_steps:
            narrator_raw = await self._llm_call(
                POSITION_NARRATOR_PROMPT,
                structured_text,
                model=model,
                effort=tier_effort,
                prompt_name="POSITION_NARRATOR_PROMPT",
            )
            total_tokens = self._last_token_usage
            narrator_json: Dict[str, Any] = {}
            if narrator_raw:
                try:
                    narrator_json = json.loads(_strip_json_fence(narrator_raw))
                except Exception:
                    narrator_json = {"raw": narrator_raw}
            step2_input = (
                f"POSITION ASSESSMENT:\n{json.dumps(narrator_json, indent=2)}\n\n"
                f"{structured_text}"
            )
            prose = await self._run_composer_segments(
                composer_prompt,
                step2_input,
                model=model,
                effort=tier_effort,
                prompt_name="composer_two_step",
            )
            total_tokens += self._last_token_usage
            if llm_debug is not None:
                llm_debug["token_usage_total"] = total_tokens
            return prose
        prose = await self._run_composer_segments(
            composer_prompt,
            structured_text,
            model=model,
            effort=tier_effort,
            prompt_name="composer_single",
        )
        if llm_debug is not None:
            llm_debug["token_usage_total"] = self._last_token_usage
        return prose

    async def generate_game_digest(
        self,
        context: GameAnalysisContext,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Whole-game structured summary for per-move context (runs before move commentary)."""
        if not self.client.api_key:
            return {}
        user = build_game_digest_input(context)
        raw = await self._llm_call_json_schema(
            GAME_DIGEST_PROMPT,
            user,
            model=model,
            effort=effort or "low",
            schema=GAME_DIGEST_SCHEMA,
            schema_name="game_digest",
        )
        try:
            text = _strip_json_fence(raw)
            if not text:
                return {}
            return json.loads(text)
        except Exception as e:
            logger.warning("generate_game_digest parse failed: %s", e)
            return {}

    async def generate_episode_commentary(
        self,
        episode: Episode,
        *,
        model: Optional[str] = None,
        effort: str = "low",
    ) -> str:
        if not self.client.api_key:
            return ""
        moves_summary = []
        for e in episode.move_events:
            eva = e.eval_after_cp / 100.0 if e.eval_after_cp is not None else None
            moves_summary.append(
                f"{e.san} (eval {eva:+.2f})" if eva is not None else e.san
            )
        text = (
            f"Episode: {episode.title}\n"
            f"Theme: {episode.dominant_theme}\n"
            f"Phase: {episode.phase}\n"
            f"Eval trend (White POV pawns): "
            f"{[x/100.0 for x in episode.eval_trend]}\n"
            f"Moves: {' '.join(moves_summary)}\n"
        )
        raw = await self._llm_call(
            EPISODE_NARRATIVE_PROMPT, text, model=model, effort=effort
        )
        if raw:
            try:
                obj = json.loads(raw)
                return sanitize_text(obj.get("commentary", raw))
            except Exception:
                return sanitize_text(raw.strip())
        return ""

    async def generate_game_narrative(
        self,
        context: GameAnalysisContext,
        *,
        model: Optional[str] = None,
        effort: str = "low",
    ) -> str:
        if not self.client.api_key:
            return ""
        parts = []
        for ep in context.episodes:
            parts.append(
                f"Ep{ep.episode_index}: {ep.title} | {ep.dominant_theme} | "
                f"narrative={ep.narrative_summary or ''}"
            )
        raw = await self._llm_call(
            GAME_NARRATIVE_PROMPT,
            "\n".join(parts),
            model=model,
            effort=effort,
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
        model: Optional[str],
        effort: Optional[str],
        prompt_name: str = "composer_segment",
    ) -> str:
        """Structured segment JSON only (OpenAI json_schema); one retry on empty/failure."""
        last_raw: Optional[str] = None
        for attempt in range(2):
            try:
                raw = await self._llm_call(
                    structured_system,
                    user_text,
                    model=model,
                    effort=effort,
                    use_structured_composer=True,
                    prompt_name=prompt_name if attempt == 0 else f"{prompt_name}_retry",
                )
                last_raw = raw
                if raw:
                    text = _strip_json_fence(raw)
                    obj = json.loads(text)
                    out = commentary_from_composer_parsed(obj)
                    if out.strip():
                        return out
            except Exception as e:
                logger.warning("Structured composer failed (attempt %s): %s", attempt + 1, e)
        logger.warning(
            "Structured composer returned empty after retries; last raw=%.200s",
            (last_raw or ""),
        )
        return ""

    async def _llm_call(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        use_structured_composer: bool = False,
        prompt_name: Optional[str] = None,
    ) -> Optional[str]:
        """Make a single LLM call and return the raw text response."""
        if _log_llm_prompts_enabled() and prompt_name:
            _debug_log_prompt(prompt_name, system_prompt, user_text)
        kwargs: Dict[str, Any] = {
            "model": model or "gpt-5.4",
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            "reasoning": {"effort": effort or "low"},
        }
        if use_structured_composer:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "chess_commentary_composer",
                    "strict": True,
                    "schema": COMPOSER_OUTPUT_SCHEMA,
                }
            }
        response = await self.client.responses.create(**kwargs)
        self._last_token_usage = self._extract_token_usage(response)
        # Extract text from response
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output") and response.output:
            try:
                return response.output[0].content[0].text
            except Exception:
                pass
        return None

    @staticmethod
    def _extract_token_usage(response: Any) -> int:
        u = getattr(response, "usage", None)
        if u is None:
            return 0
        total = getattr(u, "total_tokens", None)
        if isinstance(total, int):
            return total
        return 0

    async def _llm_call_json_schema(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: Optional[str],
        effort: str,
        schema: Dict[str, Any],
        schema_name: str,
    ) -> str:
        """Structured JSON via json_schema (strict)."""
        if _log_llm_prompts_enabled():
            _debug_log_prompt(f"json_schema:{schema_name}", system_prompt, user_text)
        kwargs: Dict[str, Any] = {
            "model": model or "gpt-5.4",
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
            ],
            "reasoning": {"effort": effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        response = await self.client.responses.create(**kwargs)
        self._last_token_usage = self._extract_token_usage(response)
        logger.info(
            "LLM JSON pass %s token_usage≈%s",
            schema_name,
            self._last_token_usage,
        )
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        if hasattr(response, "output") and response.output:
            try:
                return response.output[0].content[0].text.strip()
            except Exception:
                pass
        return ""

