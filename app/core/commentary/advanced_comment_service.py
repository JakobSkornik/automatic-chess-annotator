from __future__ import annotations

import chess
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.commentary.annotation_tokens import auto_tokenize
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.llm_providers import DYNAMIC_SECTION_SENTINEL, LlmProvider, make_llm_provider
from app.core.commentary.rag_retriever import (
    RAGRetriever,
    RAGResult,
    build_rag_query,
)
from app.core.commentary.tantivy_positional_retriever import get_default_retriever
from app.core.commentary.move_rationale import build_rationale
from app.core.commentary.motif_phrases import motif_glossary_prompt_block_for
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
TIER_FULL_HIGH = {"steps": 1, "effort": "medium", "max_tokens": 900}
TIER_FULL_LOW = {"steps": 1, "effort": "low", "max_tokens": 700}
TIER_SINGLE = {"steps": 1, "effort": "low", "max_tokens": 500}

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

ARCHETYPE_IDEAS: Dict[str, List[str]] = {
    "opposite_side_race": [
        "Tempo on the attacker's wing beats material.",
        "Open lines toward the enemy king decide races.",
        "Central breaks can defuse a wing attack.",
    ],
    "iqp": [
        "The IQP side seeks piece activity and open files.",
        "Blockading the IQP is a key defensive plan.",
    ],
    "minority_attack": [
        "Create a weakness on the minority wing with pawn levers.",
        "Rooks belong on the open file toward the target.",
    ],
    "hedgehog": [
        "Elastic pawn chain; breaks with b5 or f5 come later.",
        "Pieces re-route behind the pawns before the rupture.",
    ],
    "closed_maneuvering": [
        "Regroup knights to better squares; avoid pawn tension.",
        "Probe for weaknesses before committing a break.",
    ],
    "endgame_technique": [
        "Activate the king; create passed pawns with tempo.",
        "Opposition and zugzwang motifs decide many endings.",
    ],
    "king_hunt": [
        "Forcing checks drive the king into the open.",
        "Sacrifices clear escape squares.",
    ],
    "simplification_endgame": [
        "Trade into a won or holdable ending.",
        "Remove the opponent's active pieces first.",
    ],
    "maroczy_bind": [
        "Space clamp on d5; knights hop to ideal squares.",
        "Black seeks c5 or f5 breaks under restraint.",
    ],
    "carlsbad": [
        "Minority attack on the queenside is a main plan.",
        "The exchange variation changes pawn structure goals.",
    ],
    "other": [
        "Connect rooks and improve the worst piece.",
        "Match plans to the pawn structure center type.",
    ],
}


def _detail_level_for_key_moment(km: Optional[str]) -> str:
    if km in ("brilliant", "blunder", "critical_decision"):
        return "full"
    if km in ("mistake", "structural_transformation", "king_safety_crisis"):
        return "compact"
    return "minimal"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

EPISODE_NARRATIVE_PROMPT = (
    "You are a grandmaster chess commentator narrating the story of a game phase.\n"
    "Input lists each move as: SAN | eval | move_quality | key_moment | tactical_motifs | strategic_motifs.\n"
    "Write 2-4 sentences on plans, how the phase evolved, and the critical idea. Name motifs when relevant.\n"
    "Output strictly valid JSON: {\"commentary\": \"2-4 sentences\"}"
)

GAME_NARRATIVE_PROMPT = (
    "You are a grandmaster chess commentator. Input includes RESULT, Elos, episode summaries, "
    "digest turning points (ply, san, why, motif), and strategic_archetype.\n"
    "Write 3-5 sentences: opening character, critical phase, how the result arose. No move-by-move dump.\n"
    "Output strictly valid JSON: {\"commentary\": \"3-5 sentences\"}"
)

GAME_DIGEST_PROMPT = (
    "You are a chess coach. Given a compact move log with evaluations and key-moment tags, produce a "
    "structured digest of the whole game. Output strict JSON; no markdown. Be specific but compact.\n"
    "Pick strategic_archetype from the enum that best fits the game.\n"
    "For each turning point, motif must echo the strongest tactical/strategic tag from the move line "
    "(or \"none\" if absent).\n"
)

GAME_DIGEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_story": {"type": "string"},
        "opening_character": {"type": "string"},
        "strategic_archetype": {
            "type": "string",
            "enum": [
                "opposite_side_race",
                "iqp",
                "minority_attack",
                "hedgehog",
                "closed_maneuvering",
                "endgame_technique",
                "king_hunt",
                "simplification_endgame",
                "maroczy_bind",
                "carlsbad",
                "other",
            ],
        },
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
                    "motif": {"type": "string"},
                },
                "required": ["ply", "san", "why", "motif"],
                "additionalProperties": False,
            },
        },
        "winning_side_plan": {"type": "string"},
        "losing_side_mistakes": {"type": "string"},
    },
    "required": [
        "overall_story",
        "opening_character",
        "strategic_archetype",
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
        "strategic_archetype": digest.get("strategic_archetype", ""),
        "turning_points_nearby": near,
        "phase_story": digest.get("phase_story", []),
        "winning_side_plan": digest.get("winning_side_plan", ""),
        "losing_side_mistakes": digest.get("losing_side_mistakes", ""),
    }


STYLE_GUIDE_BLOCK = (
    "VOICE — coach, not engine dump.\n"
    "- Prefer at most one explicit eval number in the prose; omit it if the story is clear without it.\n"
    "- Forbidden phrases (never write): \"engine confirms\", \"engine preference\", \"engine's top choice\", "
    "\"settles near\", \"settling near\", \"holds the balance\", \"preserves the rhythm\", \"drives the rhythm\", "
    "\"stalls the initiative\", \"flows through\", \"keeps matters level\", \"king tuck\".\n"
    "- On captures or checks, include a short forcing line inline as [pv:san1 san2 ...] (2-3 plies).\n"
    "- Keep prose tight: ~50 words for low-effort tiers; up to ~90 for medium.\n"
    "- One short paragraph only; no labeled lists. Never start with \"Immediate:\", \"Future:\", \"Counterfactual:\".\n"
)

MOTIF_FIRST_BLOCK = (
    "WORKFLOW\n"
    "1) Read PLAYED_MOVE. The subject of the commentary is exactly that SAN and no other.\n"
    "   NEVER describe the engine's next move / PV continuation as if it were the played move.\n"
    "   If the played move is a recapture or quiet development with near-zero eval swing,\n"
    "   a one-sentence factual note is fine — do not invent a tactic.\n"
    "2) Silently pick 1-2 motifs from DETECTED_MOTIFS that are actually expressed by the PLAYED_MOVE.\n"
    "   A motif that only appears in the opponent's planned reply does NOT qualify.\n"
    "3) Write 2-3 flowing sentences that make those motifs concrete (pieces, squares, lines).\n"
    "4) Fill named_motifs with the motif keys you actually reflected in the prose.\n"
    "Do NOT name motifs that are absent from DETECTED_MOTIFS.\n"
)

PLAIN_OUTPUT_INSTRUCTIONS = (
    "\nOUTPUT: JSON only (no markdown). Fields: \"named_motifs\" (array of strings), \"text\" (string).\n"
    "The \"text\" field is plain prose (2-3 sentences):\n"
    "- SAN moves: write plainly (e.g. Nf3, cxd4). Do NOT wrap them in [move:...]; the backend tokenizes legal SAN.\n"
    "- Forcing continuations: use literals exactly like [pv:san1 san2 ...] (space-separated SAN inside brackets).\n"
    "- Squares: plain (e3, b3). Signed evals: plain (+0.42). Files: phrase like \"the e-file\" (not bracket tokens).\n"
)


def _composer_system_with_role(role: str) -> str:
    return STYLE_GUIDE_BLOCK + "\n\n" + MOTIF_FIRST_BLOCK + "\n\n" + role.strip() + "\n" + PLAIN_OUTPUT_INSTRUCTIONS


TACTICAL_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: TACTICAL — forcing tactics, captures, concrete threats.\n"
    "GOOD: names the motif, cites squares. BAD: generic engine praise."
)

POSITIONAL_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: POSITIONAL — structure, plans, long-term trumps."
)

DEFENSIVE_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: DEFENSIVE — danger that existed and how this move answers it."
)

PROPHYLACTIC_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: PROPHYLACTIC — which opponent idea was restrained."
)

FORCING_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: FORCING — check, capture, or sharp threat; keep the sequence vivid."
)

BOOK_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: OPENING / book — 1-2 sentences on development or theory."
)

INACCURACY_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: INACCURACY — what was better in plain chess terms."
)

CRITICAL_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: CRITICAL — one sharp strategic reason the best line matters."
)

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

# JSON Schema for OpenAI Structured Outputs (strict) — plain prose + motif tags.
COMPOSER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "named_motifs": {"type": "array", "items": {"type": "string"}},
        "text": {"type": "string"},
    },
    "required": ["named_motifs", "text"],
    "additionalProperties": False,
}


FORBIDDEN_PHRASES_RE = re.compile(
    r"engine confirms|engine preference|engine's top choice|settles near|settling near|"
    r"holds the balance|preserves the rhythm|drives the rhythm|stalls the initiative|"
    r"flows through|keeps matters level|king tuck",
    re.I,
)


def compute_commentary_audit(text: str, detected_motifs: List[str]) -> Dict[str, Any]:
    eval_tokens = len(re.findall(r"\[eval:[^\]]+\]", text))
    forbidden = len(FORBIDDEN_PHRASES_RE.findall(text))
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
    return {
        "motif_coverage_ratio": round(coverage, 3),
        "eval_token_count": eval_tokens,
        "forbidden_phrase_hits": forbidden,
    }


def build_planned_llm_passes_and_system_prompts(
    key_moment_type: Optional[str],
    move_category: Optional[str],
    tier_effort: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Planned passes and system prompts for WS debug (mirrors analyze_and_compose_raw_text)."""
    km = key_moment_type or ""
    tier = KEY_MOMENT_TIERS.get(km, TIER_SINGLE)
    cat_key = move_category or MoveCategory.POSITIONAL.value
    composer_prompt = CATEGORY_COMPOSER_PROMPTS.get(
        cat_key, CATEGORY_COMPOSER_PROMPTS[MoveCategory.POSITIONAL.value]
    )
    composer_name = f"composer_{cat_key}"
    passes: List[Dict[str, Any]] = [
        {
            "name": "composer_single",
            "effort": tier_effort,
            "tier_max_output_tokens": tier.get("max_tokens"),
        }
    ]
    system_prompts: List[Dict[str, str]] = [{"name": composer_name, "text": composer_prompt}]
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
        rag_retriever: Optional[RAGRetriever] = None,
        *,
        provider: Optional[LlmProvider] = None,
        provider_key: Optional[str] = None,
    ) -> None:
        self._provider: LlmProvider = provider or make_llm_provider(provider_key)
        self._rag: RAGRetriever = rag_retriever or get_default_retriever()
        self._last_token_usage: int = 0

    @property
    def provider_name(self) -> str:
        return self._provider.name

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
    def _format_rag_block(results: List[RAGResult], *, max_chars: int = 1200) -> str:
        if not results:
            return ""
        cap = max(80, int(max_chars))
        lines = [
            "REFERENCE EXAMPLES FROM MASTER GAMES:",
            "(Examples may quote master annotations from a few plies later in similar games; "
            "use them as plan inspiration, not as the current position's evaluation.)",
        ]
        for i, r in enumerate(results, 1):
            tags = ", ".join(f"{k}={v}" for k, v in list(r.relevance_tags.items())[:10])
            ph = (r.relevance_tags.get("phase") or "").strip()
            sc = r.similarity_score
            score_txt = f" score={sc:.3f}" if sc is not None else ""
            ph_txt = f" phase={ph}" if ph else ""
            lines.append(f"[{i}]{ph_txt}{score_txt} source={r.source}")
            if tags:
                lines.append(f"    Matched: {tags}")
            ann = r.annotation_text or ""
            lines.append(f"    {ann[:cap]}")
        return "\n".join(lines)

    @staticmethod
    def _format_move_event_minimal(move_event: MoveEvent) -> str:
        """Reduced engine/position block for low-tier moves (no PV dump, no feature laundry list)."""
        fe = move_event
        parts: List[str] = []
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
            bev = fe.best_move_eval_cp / 100.0 if fe.best_move_eval_cp is not None else None
            parts.append(
                f"Best engine move: {fe.best_move_san}"
                + (f" (eval {bev:+.2f})" if bev is not None else "")
            )
        parts.append(f"Move quality: {fe.move_quality.value}")
        if fe.move_category:
            parts.append(f"Move category: {fe.move_category.value}")
        if fe.tactical_motifs:
            parts.append("Tactical motifs: " + ", ".join(m.value for m in fe.tactical_motifs))
        if fe.strategic_motifs:
            parts.append("Strategic motifs: " + ", ".join(m.value for m in fe.strategic_motifs))
        if fe.pawn_structure_type:
            parts.append(f"Pawn structure (center): {fe.pawn_structure_type}")
        if fe.opening_name or fe.opening_eco:
            parts.append(
                f"Opening: {fe.opening_name or ''} ({fe.opening_eco or ''})".strip()
            )
        parts.append(f"FEN after: {fe.fen_after}")
        return "\n".join(parts)

    async def build_event_llm_input(
        self,
        move_event: MoveEvent,
        episode: Optional[Episode],
        game_context: GameAnalysisContext,
        analyzed_row: Optional[AnalyzedMoveData] = None,
        composer_effort: Optional[str] = None,
        *,
        rag_results: Optional[List[RAGResult]] = None,
        rationale_override: Optional[MoveRationale] = None,
    ) -> Tuple[str, List[RAGResult], Dict[str, Any]]:
        query = build_rag_query(move_event, episode)
        detail_pre = _detail_level_for_key_moment(move_event.key_moment_type)
        rag_top_k = 1 if detail_pre == "minimal" else 2
        if rag_results is None:
            rag_results = await self._rag.retrieve(query, top_k=rag_top_k)
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

        detail = detail_pre
        if detail == "full":
            rag_max_chars = 1200
        elif detail == "minimal":
            rag_max_chars = 240
        else:
            rag_max_chars = 300
        gd = getattr(game_context, "game_digest", None) or {}
        detected = [m.value for m in move_event.tactical_motifs] + [
            m.value for m in move_event.strategic_motifs
        ]

        try:
            _b_pre = chess.Board(move_event.fen_before)
            mover_side = "White" if _b_pre.turn == chess.WHITE else "Black"
            next_side = "Black" if _b_pre.turn == chess.WHITE else "White"
        except Exception:
            mover_side = "?"
            next_side = "?"
        move_label = f"{(move_event.ply + 1) // 2}{'.' if mover_side == 'White' else '...'} {move_event.san}"

        static_blocks: List[str] = []
        if gd:
            compact_ctx = compact_game_context_for_move(gd, move_event.ply)
            static_blocks.append(
                "GAME CONTEXT (whole-game digest; use for tone and continuity, do not re-quote):\n"
                + json.dumps(compact_ctx, indent=2)
            )

        arch = str(gd.get("strategic_archetype") or "other").strip() or "other"
        ideas = ARCHETYPE_IDEAS.get(arch, ARCHETYPE_IDEAS["other"])
        static_blocks.append(
            "STRATEGIC ARCHETYPE TEXTURE (do not quote verbatim; ground plans in these ideas):\n"
            f"Archetype: {arch}\n" + "\n".join(f"- {line}" for line in ideas)
        )

        pri = getattr(game_context, "prior_context_snippets", None) or []
        if pri:
            static_blocks.append(
                "PRIOR_CONTEXT (continuity only; do not restate):\n" + "\n".join(pri[-2:])
            )

        static_blocks.append(
            "DETECTED_MOTIFS (only name motifs from this list):\n"
            + (", ".join(detected) if detected else "(none)")
        )
        static_blocks.append(motif_glossary_prompt_block_for(detected))

        dynamic_blocks: List[str] = []
        dynamic_blocks.append(
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

        rb = self._format_rag_block(rag_results, max_chars=rag_max_chars)
        if rb:
            dynamic_blocks.append(rb)

        rationale = (
            rationale_override
            if rationale_override is not None
            else build_rationale(move_event, move_event.future_line)
        )
        if detail == "minimal":
            rdict = rationale.model_dump()
            rdict.pop("counterfactual", None)
            rdict.pop("future_effect", None)
            rationale_blob = json.dumps(rdict, indent=2)
        else:
            rationale_blob = rationale.model_dump_json(indent=2)
        dynamic_blocks.append(
            "MOVE_RATIONALE_JSON (internal evidence; weave into prose, never quote field names or bullet them):\n"
            + rationale_blob
        )

        if detail == "full":
            dynamic_blocks.append(
                "POSITION AND ENGINE DATA (anchor commentary to this; do not invent lines):\n"
                + self._format_move_event_block(move_event)
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
                + self._format_move_event_minimal(move_event)
            )

        if static_blocks:
            structured_text = (
                "\n\n".join(static_blocks) + DYNAMIC_SECTION_SENTINEL + "\n\n".join(dynamic_blocks)
            )
        else:
            structured_text = "\n\n".join(dynamic_blocks)
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
            "detail_level": detail,
            "detected_motif_labels": detected,
            "rag_query": query.model_dump(),
            "rationale": rationale.model_dump(),
            "system_prompts": system_prompts,
            "user_text": structured_text,
            "passes": passes,
            "token_usage_total": None,
            "tokens_by_pass": {},
            "elapsed_ms": None,
            "composer_named_motifs": [],
            "commentary_audit": None,
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
            fen_before=move_event.fen_before,
            fen_after=move_event.fen_after,
        )
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
        fen_before: Optional[str] = None,
        fen_after: Optional[str] = None,
    ) -> str:
        if not self._provider.is_configured():
            if llm_debug is not None:
                llm_debug["token_usage_total"] = None
            return (
                "AI comments are disabled. Set OPENAI_API_KEY (OpenAI) or "
                "ANTHROPIC_API_KEY (Anthropic)."
            )
        t0 = time.perf_counter()
        tokens_by_pass: Dict[str, int] = {}
        tier = KEY_MOMENT_TIERS.get(key_moment_type or "", TIER_SINGLE)
        tier_effort = effort or tier["effort"]
        tier_max_tokens = int(tier.get("max_tokens", TIER_SINGLE["max_tokens"]))
        cat_key = move_category or MoveCategory.POSITIONAL.value
        composer_prompt = CATEGORY_COMPOSER_PROMPTS.get(
            cat_key, CATEGORY_COMPOSER_PROMPTS[MoveCategory.POSITIONAL.value]
        )
        if _log_llm_prompts_enabled():
            logger.info(
                "analyze_and_compose_raw_text: branch=single move_category=%s key_moment_type=%s",
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

        prose: str = ""
        composer_named: List[str] = []

        mdl = model or resolve_model(self.provider_name, "composer")
        prose, composer_named = await self._run_composer_segments(
            composer_prompt,
            structured_text,
            model=mdl,
            effort=tier_effort,
            prompt_name="composer_single",
            max_output_tokens=tier_max_tokens,
            fen_before=fen_before,
            fen_after=fen_after,
        )
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
            det = llm_debug.get("detected_motif_labels") or []
            llm_debug["commentary_audit"] = compute_commentary_audit(prose, det)
        return prose

    async def generate_game_digest(
        self,
        context: GameAnalysisContext,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Whole-game structured summary for per-move context (runs before move commentary)."""
        if not self._provider.is_configured():
            return {}
        user = build_game_digest_input(context)
        digest_cap = int(os.environ.get("LLM_DIGEST_MAX_OUTPUT_TOKENS", "8192"))
        mdl = model or resolve_model(self.provider_name, "digest")
        raw = await self._llm_call_json_schema(
            GAME_DIGEST_PROMPT,
            user,
            model=mdl,
            effort=effort or "low",
            schema=GAME_DIGEST_SCHEMA,
            schema_name="game_digest",
            max_output_tokens=digest_cap,
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
        if not self._provider.is_configured():
            return ""
        moves_summary: List[str] = []
        for e in episode.move_events:
            eva = e.eval_after_cp / 100.0 if e.eval_after_cp is not None else None
            evs = f"{eva:+.2f}" if eva is not None else "?"
            mq = e.move_quality.value if e.move_quality else ""
            km = e.key_moment_type or ""
            tact = ",".join(m.value for m in e.tactical_motifs[:5])
            strat = ",".join(m.value for m in e.strategic_motifs[:5])
            moves_summary.append(
                f"{e.san} | eval {evs} | mq={mq} | km={km} | tact={tact} | strat={strat}"
            )
        text = (
            f"Episode: {episode.title}\n"
            f"Theme: {episode.dominant_theme}\n"
            f"Phase: {episode.phase}\n"
            f"Eval trend (White POV pawns): "
            f"{[x/100.0 for x in episode.eval_trend]}\n"
            f"Moves (SAN | eval | move_quality | key_moment | tactical | strategic):\n"
            + "\n".join(moves_summary)
            + "\n"
        )
        mdl = model or resolve_model(self.provider_name, "episode")
        raw = await self._llm_call(
            EPISODE_NARRATIVE_PROMPT,
            text,
            model=mdl,
            effort=effort,
            max_output_tokens=512,
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
        if not self._provider.is_configured():
            return ""
        meta = context.metadata or {}
        digest = context.game_digest or {}
        header_lines = [
            f"RESULT: {meta.get('result', '*')}",
            f"WHITE_ELO: {meta.get('whiteElo')} BLACK_ELO: {meta.get('blackElo')}",
            f"STRATEGIC_ARCHETYPE: {digest.get('strategic_archetype', '')}",
            f"TURNING_POINTS_JSON:\n{json.dumps(digest.get('turning_points', []), indent=2)}",
            "",
            "EPISODES:",
        ]
        parts: List[str] = list(header_lines)
        for ep in context.episodes:
            parts.append(
                f"Ep{ep.episode_index}: {ep.title} | {ep.dominant_theme} | "
                f"narrative={ep.narrative_summary or ''}"
            )
        mdl = model or resolve_model(self.provider_name, "narrative")
        raw = await self._llm_call(
            GAME_NARRATIVE_PROMPT,
            "\n".join(parts),
            model=mdl,
            effort=effort,
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
        model: Optional[str],
        effort: Optional[str],
        prompt_name: str = "composer_segment",
        max_output_tokens: Optional[int] = None,
        fen_before: Optional[str] = None,
        fen_after: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Plain-text composer JSON (named_motifs + text); tokenize prose; one retry on empty/failure."""
        fb = fen_before or chess.Board().fen()
        fa = fen_after or chess.Board().fen()
        last_raw: Optional[str] = None
        last_plain_len: Optional[int] = None
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
                    named = [str(x) for x in named_raw if isinstance(x, str) and x.strip()]
                    plain = str(obj.get("text") or "").strip()
                    last_plain_len = len(plain)
                    out = sanitize_text(auto_tokenize(plain, fb, fa)) if plain else ""
                    if out.strip():
                        return out, named
                    logger.warning(
                        "Structured composer empty prose after parse (attempt %s): "
                        "raw_preview=%.500s plain_len=%s",
                        attempt + 1,
                        text,
                        last_plain_len,
                    )
            except Exception as e:
                logger.warning("Structured composer failed (attempt %s): %s", attempt + 1, e)
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
        return "", []

    async def _llm_call(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        use_structured_composer: bool = False,
        prompt_name: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Make a single LLM call and return the raw text response."""
        if _log_llm_prompts_enabled() and prompt_name:
            _debug_log_prompt(prompt_name, system_prompt, user_text)
        resolved_model = model or resolve_model(self.provider_name, "composer")
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
            return raw or None
        raw, usage = await self._provider.text_call(
            system_prompt,
            user_text,
            model=resolved_model,
            effort=effort or "low",
            max_output_tokens=max_output_tokens,
        )
        self._last_token_usage = usage
        return raw or None

    async def _llm_call_json_schema(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: Optional[str],
        effort: str,
        schema: Dict[str, Any],
        schema_name: str,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Structured JSON via provider json_schema call."""
        if _log_llm_prompts_enabled():
            _debug_log_prompt(f"json_schema:{schema_name}", system_prompt, user_text)
        resolved_model = model or resolve_model(self.provider_name, "digest")
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
        return (raw or "").strip()

