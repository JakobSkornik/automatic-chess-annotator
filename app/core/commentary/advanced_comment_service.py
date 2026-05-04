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
from app.core.commentary.motif_phrases import glossary_phrase_for, motif_glossary_prompt_block_for
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


def compute_rag_top_k(detail: str, phase: Optional[str]) -> int:
    """Per-phase Top-K defaults; overridden by RAG_TOP_K_PHASE or global RAG_TOP_K."""
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


STRATEGIC_MOTIF_CONTRADICTIONS: Tuple[frozenset, ...] = (
    frozenset({"bad_bishop", "bishop_pair_advantage"}),
)


def _collapse_contradictory_strategic(labels: List[str], primary: str) -> List[str]:
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


def cap_motifs_for_prompt(move_event: MoveEvent, primary_motif: str = "") -> List[str]:
    """Expose at most 3 tactical + 2 strategic motifs, deduped, primary first."""
    tact = [m.value for m in move_event.tactical_motifs]
    strat = _collapse_contradictory_strategic(
        [m.value for m in move_event.strategic_motifs],
        primary_motif,
    )
    tact_u = list(dict.fromkeys(tact))
    strat_u = list(dict.fromkeys(strat))
    pri = primary_motif.strip()

    tact_ord: List[str] = []
    if pri and pri in tact_u:
        tact_ord.append(pri)
    for t in tact_u:
        if t not in tact_ord:
            tact_ord.append(t)
    tact_ord = tact_ord[:3]

    strat_ord: List[str] = []
    if pri and pri in strat_u:
        strat_ord.append(pri)
    for s in strat_u:
        if s not in strat_ord:
            strat_ord.append(s)
    strat_ord = strat_ord[:2]

    return list(dict.fromkeys(tact_ord + strat_ord))


def format_motif_digest_lines(keys: List[str]) -> str:
    if not keys:
        return "(none)"
    return "\n".join(f"- {k} — {glossary_phrase_for(k)}" for k in keys)


def composer_role_key(category_value: Optional[str]) -> str:
    """Collapse MoveCategory routing to four composer roles."""
    c = (category_value or MoveCategory.POSITIONAL.value).strip().lower()
    if c == MoveCategory.BOOK.value:
        return "book"
    if c in (MoveCategory.TACTICAL.value, MoveCategory.FORCING.value, MoveCategory.DEFENSIVE.value):
        return "tactical_forcing"
    if c in (MoveCategory.INACCURACY.value, MoveCategory.CRITICAL.value):
        return "mistake_explainer"
    return "positional_plan"


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
    "AUDIENCE ~1500–2000: explain WHY in plain chess language, not motif laundry lists.\n"
    "VOICE — coach, not engine dump.\n"
    "- Prefer at most one explicit eval number in the whole comment; otherwise use words (~\"about a pawn\").\n"
    "- Name at most ONE motif key from MOTIF HINTS, and only if it is concretely visible; otherwise describe in English.\n"
    "- Forbidden phrases (never write): \"engine confirms\", \"engine preference\", \"engine's top choice\", "
    "\"settles near\", \"settling near\", \"holds the balance\", \"preserves the rhythm\", \"drives the rhythm\", "
    "\"stalls the initiative\", \"flows through\", \"keeps matters level\", \"king tuck\".\n"
    "- On captures or checks, include a short forcing line as [pv:san1 san2 ...] (2-3 plies).\n"
    "- Opponent-aware phrasing (\"Black can now ...\").\n"
    "- In openings still in theory, cite opening name/code and typical plans when known.\n"
    "- Endgames: name the decisive element (king, passed pawn, bishop colour, etc.).\n"
    "- Keep prose tight: ~50 words for low-effort tiers; up to ~90 for medium.\n"
    "- One short paragraph only; no labeled lists. Never start with \"Immediate:\", \"Future:\", \"Counterfactual:\".\n"
)

EXPLAIN_WHY_BLOCK = (
    "WORKFLOW\n"
    "1) Identify the concrete idea this move embodies (attacks, frees a line, fixes a weakness, prepares a break).\n"
    "   NEVER describe the opponent's refutation/PV continuation as if it were the played move.\n"
    "2) Say why this is desirable or costly for side to move AFTER the ply, referencing alternatives when useful.\n"
    "   For mistakes: name the refutation idea IN WORDS before the [pv:...] snippet.\n"
    "3) Compare briefly to engine best-move when PLAYED differs from BEST (POSITION block), unless BOOK tier.\n"
    "4) Only after (1)-(3): optionally attach ONE motif hint from MOTIF HINTS if genuinely expressed.\n\n"
    "RAG MASTER ANNOTATIONS block (when present):\n"
    "- If one idea obviously applies here, summarize it once in prose (do NOT quote verbatim).\n"
    "- Populate rag_applied=true and rag_idea_used with ONE sentence naming that idea.\n"
    "- If none apply, rag_applied=false and rag_idea_used=\"\".\n"
)

PLAIN_OUTPUT_INSTRUCTIONS = (
    "\nOUTPUT: JSON only (no markdown).\n"
    "- \"named_motifs\": array of 0–1 motif keys you anchored in prose (subset of MOTIF HINTS).\n"
    "- \"text\": plain prose paragraph (SAN tokens plain; forcing lines as [pv:san san ...]).\n"
    "- \"better_alternative\": one sentence-ready clause (Better was … / Instead … with reason), or \"\" if irrelevant.\n"
    "  REQUIRED for inaccuracies/mistakes/blunders/critical classifications; optional otherwise.\n"
    "- \"rag_idea_used\": one sentence naming the reused master-note idea (or \"\").\n"
    "- \"rag_applied\": boolean — true iff MASTER ANNOTATIONS influenced your explanation.\n"
)


def _composer_system_with_role(role: str) -> str:
    return (
        STYLE_GUIDE_BLOCK
        + "\n\n"
        + EXPLAIN_WHY_BLOCK
        + "\n\n"
        + role.strip()
        + "\n"
        + PLAIN_OUTPUT_INSTRUCTIONS
    )


BOOK_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: BOOK / theory — openings where development and tabiya still matter.\n"
    "- 1 crisp sentence tying the SAN to typical plans.\n"
    "- Reference opening name/code when MOTIF hints or POSITION mention them.\n"
    "- better_alternative \"\" unless theory clearly rejects the move.\n"
)

TACTICAL_FORCING_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: TACTICAL / FORCING / DEFENCE — captures, checks, tactical shots, tactical defence.\n"
    "- Lead with forcing consequences and concrete squares.\n"
    "- Mention defence only when STOPPING a direct threat matters.\n"
)

POSITIONAL_PLAN_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: POSITIONAL PLAN / PROPHYLAXIS — structure, slow manoeuvres, quiet improvements.\n"
    "- Name files, pawn breaks, weaknesses, timing — not jargon stacks.\n"
)

MISTAKE_EXPLAINER_COMPOSER_PROMPT = _composer_system_with_role(
    "ROLE: MISTAKE & CRITICAL MOMENTS — eval swings, dubious choices, decisive branches.\n"
    "- Mandatory better_alternative whenever BEST_MOVE differs materially from PLAYED and quality is dubious.\n"
    "- Tie alternatives to measurable plans (outposts, breaks, king safety).\n"
)

COMPOSER_ROLE_PROMPTS: Dict[str, str] = {
    "book": BOOK_COMPOSER_PROMPT,
    "tactical_forcing": TACTICAL_FORCING_COMPOSER_PROMPT,
    "positional_plan": POSITIONAL_PLAN_COMPOSER_PROMPT,
    "mistake_explainer": MISTAKE_EXPLAINER_COMPOSER_PROMPT,
}

CATEGORY_COMPOSER_PROMPTS: Dict[str, str] = {
    MoveCategory.BOOK.value: BOOK_COMPOSER_PROMPT,
    MoveCategory.TACTICAL.value: TACTICAL_FORCING_COMPOSER_PROMPT,
    MoveCategory.FORCING.value: TACTICAL_FORCING_COMPOSER_PROMPT,
    MoveCategory.DEFENSIVE.value: TACTICAL_FORCING_COMPOSER_PROMPT,
    MoveCategory.POSITIONAL.value: POSITIONAL_PLAN_COMPOSER_PROMPT,
    MoveCategory.PROPHYLACTIC.value: POSITIONAL_PLAN_COMPOSER_PROMPT,
    MoveCategory.INACCURACY.value: MISTAKE_EXPLAINER_COMPOSER_PROMPT,
    MoveCategory.CRITICAL.value: MISTAKE_EXPLAINER_COMPOSER_PROMPT,
}

# JSON Schema for OpenAI Structured Outputs (strict).
COMPOSER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "named_motifs": {"type": "array", "items": {"type": "string"}},
        "text": {"type": "string"},
        "better_alternative": {"type": "string"},
        "rag_idea_used": {"type": "string"},
        "rag_applied": {"type": "boolean"},
    },
    "required": ["named_motifs", "text", "better_alternative", "rag_idea_used", "rag_applied"],
    "additionalProperties": False,
}


FORBIDDEN_PHRASES_RE = re.compile(
    r"engine confirms|engine preference|engine's top choice|settles near|settling near|"
    r"holds the balance|preserves the rhythm|drives the rhythm|stalls the initiative|"
    r"flows through|keeps matters level|king tuck",
    re.I,
)


def compute_commentary_audit(
    text: str,
    detected_motifs: List[str],
    *,
    rag_had_hits: bool = False,
    rag_applied: Optional[bool] = None,
) -> Dict[str, Any]:
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
    audit: Dict[str, Any] = {
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
    key_moment_type: Optional[str],
    move_category: Optional[str],
    tier_effort: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Planned passes and system prompts for WS debug (mirrors analyze_and_compose_raw_text)."""
    km = key_moment_type or ""
    tier = KEY_MOMENT_TIERS.get(km, TIER_SINGLE)
    cat_key = move_category or MoveCategory.POSITIONAL.value
    role = composer_role_key(cat_key)
    composer_prompt = COMPOSER_ROLE_PROMPTS.get(role, POSITIONAL_PLAN_COMPOSER_PROMPT)
    composer_name = f"composer_{role}"
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
    def _format_move_event_block(move_event: MoveEvent, motif_hint_keys: List[str]) -> str:
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
        if motif_hint_keys:
            parts.append(
                "MOTIF HINTS (describe in English; max one label in prose):\n"
                + format_motif_digest_lines(motif_hint_keys)
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
    def _format_rag_block(results: List[RAGResult], *, max_chars: int = 1600) -> str:
        if not results:
            return ""
        cap = max(80, int(max_chars))
        lines = [
            "MASTER ANNOTATIONS — these excerpts come from strongly annotated GM/IM games "
            "in tactically or structurally similar positions.",
            "If an idea plainly applies here, weave it into your prose (ONE short paraphrase). "
            "Do NOT quote verbatim.",
        ]
        for i, r in enumerate(results, 1):
            tags = r.relevance_tags or {}
            ph = (tags.get("phase") or "").strip()
            on = (tags.get("opening_name") or tags.get("opening") or "").strip()
            eco = (tags.get("opening_eco") or tags.get("eco") or "").strip()
            mats = (tags.get("material_signature") or "").strip()
            sc = r.similarity_score
            score_txt = f"score={sc:.3f}" if sc is not None else "score=?"
            head_bits = [
                f"[{i}]",
                score_txt,
                f"phase={ph}" if ph else "",
                f"opening={on}" if on else "",
                f"eco={eco}" if eco else "",
                f"material={mats}" if mats else "",
                f"source={r.source}",
            ]
            lines.append(" ".join(b for b in head_bits if b))
            ann = r.annotation_text or ""
            lines.append(f"    {ann[:cap]}")
        return "\n".join(lines)

    @staticmethod
    def _format_move_event_minimal(move_event: MoveEvent, motif_hint_keys: List[str]) -> str:
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
        if motif_hint_keys:
            parts.append(
                "MOTIF HINTS (describe in English; max one label in prose):\n"
                + format_motif_digest_lines(motif_hint_keys)
            )
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
        rag_top_k = compute_rag_top_k(detail_pre, query.phase)

        rationale_pre = (
            rationale_override
            if rationale_override is not None
            else build_rationale(move_event, move_event.future_line)
        )
        motif_hint_keys = cap_motifs_for_prompt(move_event, rationale_pre.primary_motif_label)

        rag_retrieval_debug: Dict[str, Any] = {}
        if rag_results is None:
            rag_results = await self._rag.retrieve(
                query, top_k=rag_top_k, retrieval_debug=rag_retrieval_debug
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
                if km_w in ("critical_decision", "mistake", "blunder") and rag_retrieval_debug:
                    logger.info(
                        "RAG rejection debug ply=%s key_moment=%s %s",
                        move_event.ply,
                        km_w,
                        rag_retrieval_debug,
                    )

        detail = detail_pre
        if detail == "full":
            rag_max_chars = 1600
        elif detail == "minimal":
            rag_max_chars = 500
        else:
            rag_max_chars = 800
        gd = getattr(game_context, "game_digest", None) or {}

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
            "MOTIF HINTS (cap 3 tactical + 2 strategic; prose may name at most ONE motif key):\n"
            + (format_motif_digest_lines(motif_hint_keys) if motif_hint_keys else "(none)")
        )
        static_blocks.append(motif_glossary_prompt_block_for(motif_hint_keys))

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

        rationale = rationale_pre
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

        rb = self._format_rag_block(rag_results, max_chars=rag_max_chars)
        if rb:
            dynamic_blocks.append(rb)

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
        rag_hit_ct = len(rag_results or [])
        debug_dict: Dict[str, Any] = {
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
        role = composer_role_key(cat_key)
        composer_prompt = COMPOSER_ROLE_PROMPTS.get(role, POSITIONAL_PLAN_COMPOSER_PROMPT)
        if _log_llm_prompts_enabled():
            logger.info(
                "analyze_and_compose_raw_text: branch=single move_category=%s key_moment_type=%s composer_role=%s",
                move_category,
                key_moment_type,
                role,
            )
            logger.info("composer system prompt selected: composer_%s", role)

        prose: str = ""
        composer_named: List[str] = []

        mdl = model or resolve_model(self.provider_name, "composer")
        prose, composer_named, composer_extra = await self._run_composer_segments(
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
            llm_debug["composer_better_alternative"] = composer_extra.get("better_alternative", "")
            llm_debug["composer_rag_idea_used"] = composer_extra.get("rag_idea_used", "")
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
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """Structured composer JSON; tokenize prose; one retry on empty/failure."""
        fb = fen_before or chess.Board().fen()
        fa = fen_after or chess.Board().fen()
        last_raw: Optional[str] = None
        last_plain_len: Optional[int] = None
        empty_extras: Dict[str, Any] = {
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
                    named = [str(x) for x in named_raw if isinstance(x, str) and x.strip()]
                    plain = str(obj.get("text") or "").strip()
                    ba = str(obj.get("better_alternative") or "").strip()
                    riu = str(obj.get("rag_idea_used") or "").strip()
                    rap = bool(obj.get("rag_applied", False))
                    extras = {
                        "better_alternative": ba,
                        "rag_idea_used": riu,
                        "rag_applied": rap,
                    }
                    last_plain_len = len(plain)
                    base = sanitize_text(auto_tokenize(plain, fb, fa)) if plain else ""
                    if base.strip():
                        out = base.rstrip()
                        if ba and ba.lower() not in out.lower():
                            out = f"{out} {ba}".strip()
                        if out.strip():
                            return out, named, extras
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
        return "", [], empty_extras

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

