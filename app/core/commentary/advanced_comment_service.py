from __future__ import annotations

import chess
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.core.commentary.annotation_tokens import auto_tokenize
from app.core.commentary.rag_retriever import (
    NullRetriever,
    RAGRetriever,
    RAGResult,
    build_rag_query,
)
from app.models.Move import Move
from app.models.chess_events import AnalyzedMoveData, Episode, GameAnalysisContext, MoveEvent

logger = logging.getLogger(__name__)

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


def board_to_svg(
    fen: str,
    last_move_uci: Optional[str] = None,
    perspective: chess.Color = chess.WHITE,
    size: int = 360,
) -> Optional[str]:
    """Generate an SVG board image from a FEN string.

    Requires ``chess.svg`` (included in python-chess).  Returns the SVG XML
    string or *None* on failure.  The SVG can be converted to PNG via
    ``cairosvg.svg2png()`` for multimodal LLM APIs.

    Parameters:
        fen: Board position in FEN notation.
        last_move_uci: Optional UCI string of the last move to highlight.
        perspective: Which side is at the bottom of the board.
        size: SVG image width/height in pixels.
    """
    try:
        import chess.svg as chess_svg

        board = chess.Board(fen)
        kwargs: dict = {
            "board": board,
            "orientation": perspective,
            "size": size,
        }
        if last_move_uci:
            try:
                kwargs["lastmove"] = chess.Move.from_uci(last_move_uci)
            except Exception:
                pass
        return chess_svg.board(**kwargs)
    except ImportError:
        return None
    except Exception:
        return None


def board_to_ascii(fen: str, perspective: chess.Color = chess.WHITE) -> str:
    """Generate an ASCII board diagram from a FEN string.

    Returns an 8x8 text board with file/rank coordinates suitable for
    inclusion in LLM prompts (~100 tokens).
    """
    try:
        board = chess.Board(fen)
    except Exception:
        return f"[invalid FEN: {fen}]"

    piece_map = board.piece_map()
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(8)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)

    lines: list[str] = ["  a b c d e f g h"]
    for rank in ranks:
        row = [str(rank + 1)]
        for file_idx in files:
            sq = chess.square(file_idx, rank)
            piece = piece_map.get(sq)
            if piece is None:
                row.append(".")
            else:
                row.append(piece.symbol())
        lines.append(" ".join(row))
    return "\n".join(lines)


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

STRATEGIC_ANALYST_PROMPT = (
    "You are a grandmaster annotating for a strong club player (master book style).\n"
    "You receive: a short JSON position assessment, engine data, optional RAG references, and a one-line PV note.\n\n"
    "Write 2-3 sentences, maximum 80 words total.\n"
    "Focus on ONE strategic reason the engine prefers its line when the played move differs:\n"
    "activity, king safety, tempo, pawn structure, or central tension — pick the single clearest.\n\n"
    "Rules:\n"
    "- Do NOT enumerate feature deltas, mobility numbers, or space scores.\n"
    "- Do NOT explain both sides' plans in detail; at most one clause on how the choice changes the game.\n"
    "- Do NOT repeat centipawn values from the data block; use move/eval tokens only.\n"
    "- If the played move is best, say why in one strategic phrase.\n"
    "- Anchor to tactical motifs and PV lines given; do not invent variations.\n"
    "- The user message states \"Move played by: White|Black\". Treat that as ground truth; do not swap colors.\n"
    "- File references like \"e-file\" / \"the d-file\" must be emitted as file tokens, never as bare prose.\n\n"
    "OUTPUT: use the segment JSON format in the next instruction block only.\n"
)

SINGLE_STEP_PROMPT = (
    "You are a grandmaster annotating for a strong club player (master book style).\n"
    "Write 1-2 sentences, maximum 40 words total, about the played move.\n\n"
    "Rules:\n"
    "- Focus on ONE idea: the main strategic or tactical point.\n"
    "- If not best, name the single concrete reason the engine move is better (activity, king safety, tempo, structure).\n"
    "- Do not enumerate feature deltas. Do not list plans for both sides.\n"
    "- Do not repeat centipawn numbers as prose; put evaluations in eval segments only.\n"
    "- Every SAN move reference must be a move segment. Every evaluation an eval segment. Squares as square segments.\n"
    "- No bare SAN like Qxd5 or Nf6 in text segments.\n"
    "- The user message states \"Move played by: White|Black\". Treat that as ground truth. Never attribute the played move to the other color.\n"
    "- File references like \"e-file\" / \"the d-file\" must be emitted as file tokens, never as bare prose.\n\n"
    "Example (meaning, not literal output): prose + [move:Qxd5] + prose + [move:Nf6] + prose + [square:d4].\n\n"
    "OUTPUT: use the segment JSON format in the next instruction block only.\n"
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

SEGMENT_OUTPUT_INSTRUCTIONS = (
    "\n\nOUTPUT FORMAT: Respond with JSON only (no markdown fences). "
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

STRATEGIC_ANALYST_COMPOSER_SEGMENT_PROMPT = STRATEGIC_ANALYST_PROMPT + SEGMENT_OUTPUT_INSTRUCTIONS
SINGLE_STEP_COMPOSER_SEGMENT_PROMPT = SINGLE_STEP_PROMPT + SEGMENT_OUTPUT_INSTRUCTIONS

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
        self._rag: RAGRetriever = rag_retriever or NullRetriever()

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
        if fe.tactical_motifs:
            parts.append(
                "Tactical motifs: "
                + ", ".join(m.value for m in fe.tactical_motifs)
            )
        if fe.pawn_structure_type:
            parts.append(f"Pawn structure (center): {fe.pawn_structure_type}")
        if fe.opening_name or fe.opening_eco:
            parts.append(
                f"Opening: {fe.opening_name or ''} ({fe.opening_eco or ''})".strip()
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
    def _format_rag_block(results: List[RAGResult]) -> str:
        if not results:
            return ""
        lines = ["REFERENCE EXAMPLES FROM MASTER GAMES:"]
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
    ) -> Tuple[str, List[RAGResult]]:
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
        blocks = []
        rb = self._format_rag_block(rag_results)
        if rb:
            blocks.append(rb)
        blocks.append(
            "POSITION AND ENGINE DATA (anchor commentary to this; do not invent lines):\n"
            + self._format_move_event_block(move_event)
        )
        pv_block = self._format_pv_position_comparison(move_event, analyzed_row)
        if pv_block:
            blocks.append(pv_block)
        return "\n\n".join(blocks), rag_results

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
    ) -> Tuple[str, List[RAGResult]]:
        structured_text, rag_results = await self.build_event_llm_input(
            move_event, episode, game_context, analyzed_row=analyzed_row
        )
        text = await self.analyze_and_compose_raw_text(
            structured_text,
            model=model,
            effort=effort,
            key_moment_type=key_moment_type or move_event.key_moment_type,
        )
        text = auto_tokenize(text, move_event.fen_before, move_event.fen_after)
        return text, rag_results

    async def analyze_and_compose_raw_text(
        self,
        structured_text: str,
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        key_moment_type: Optional[str] = None,
    ) -> str:
        if not self.client.api_key:
            return "AI comments are disabled. Please set the OPENAI_API_KEY environment variable."
        tier = KEY_MOMENT_TIERS.get(key_moment_type or "", TIER_SINGLE)
        tier_effort = effort or tier["effort"]
        # Note: tier["max_tokens"] is kept as a soft documentation budget only.
        # Responses API `max_output_tokens` is a HARD cap that also counts reasoning tokens,
        # so passing small values here (e.g. 80) starves the JSON output. Prompt word caps
        # already enforce brevity.
        use_two_steps = tier["steps"] == 2
        if use_two_steps:
            narrator_raw = await self._llm_call(
                POSITION_NARRATOR_PROMPT,
                structured_text,
                model=model,
                effort=tier_effort,
            )
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
            return await self._run_composer_segments(
                STRATEGIC_ANALYST_COMPOSER_SEGMENT_PROMPT,
                step2_input,
                model=model,
                effort=tier_effort,
            )
        return await self._run_composer_segments(
            SINGLE_STEP_COMPOSER_SEGMENT_PROMPT,
            structured_text,
            model=model,
            effort=tier_effort,
        )

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

    @staticmethod
    def _uci_to_san(fen: str, uci_moves: List[str], max_len: int = 6) -> List[str]:
        """Convert a sequence of UCI moves to SAN given a starting FEN.

        Falls back to raw UCI strings on any parsing failure.
        """
        try:
            board = chess.Board(fen)
        except Exception:
            return uci_moves[:max_len]

        result: List[str] = []
        for uci_str in uci_moves[:max_len]:
            try:
                uci_move = chess.Move.from_uci(uci_str)
                if uci_move in board.legal_moves:
                    result.append(board.san(uci_move))
                    board.push(uci_move)
                else:
                    result.append(uci_str)
                    break  # cannot continue after an illegal move
            except Exception:
                result.append(uci_str)
                break
        return result

    @staticmethod
    def _format_pv_as_san(
        pv: Optional[List[Move]], starting_fen: str, max_len: int = 6
    ) -> List[str]:
        """Convert PV Move objects to SAN notation using *starting_fen* as the
        board state before the first PV move."""
        if not pv:
            return []
        uci_moves: List[str] = []
        for m in pv[:max_len]:
            try:
                if getattr(m, "move", None):
                    uci_moves.append(str(m.move))
            except Exception:
                pass
        return AdvancedCommentService._uci_to_san(starting_fen, uci_moves, max_len)

    @staticmethod
    def _move_uci_to_san(fen: str, uci: str) -> str:
        """Convert a single UCI move string to SAN. Returns the original UCI on failure."""
        try:
            board = chess.Board(fen)
            uci_move = chess.Move.from_uci(uci)
            if uci_move in board.legal_moves:
                return board.san(uci_move)
        except Exception:
            pass
        return uci

    @staticmethod
    def _format_moves_list(pv: Optional[List[Move]], max_len: int = 6) -> List[str]:
        """Legacy helper – returns UCI strings (kept for backward compat)."""
        if not pv:
            return []
        result: List[str] = []
        for idx, m in enumerate(pv[:max_len]):
            try:
                if getattr(m, "move", None):
                    result.append(str(m.move))
            except Exception:
                pass
        return result

    @staticmethod
    def _filter_feature_deltas(delta: Dict[str, Any], limit: int = 6, min_abs: float = 0.2) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for k, v in (delta or {}).items():
            try:
                val = float(v)
                if abs(val) >= min_abs:
                    items.append((k, val))
            except Exception:
                continue
        # sort by absolute magnitude desc and cut
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return items[:limit]

    def build_compact_input(
        self,
        move: Move,
        previous_move: Optional[Move],
        pvs_for_move: Optional[List[List[Move]]],
        *,
        opening: Optional[Dict[str, str]] = None,
        narrative_context: str = "",
        key_moment_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Extract AI-prepared meta from hiddenFeatures._ai if present
        ai_meta = {}
        features_after: Dict[str, Any] = {}
        features_delta: Dict[str, Any] = {}
        try:
            if isinstance(move.hiddenFeatures, dict):
                ai_meta = (move.hiddenFeatures or {}).get("_ai", {}) or {}
                features_after = (move.hiddenFeatures or {})
                features_delta = ai_meta.get("delta", {}) or {}
        except Exception:
            pass

        # Determine the board FEN *before* the played move (needed for SAN conversion)
        # previous_move.position is the FEN after the previous move = before the current move
        starting_fen = chess.STARTING_FEN
        if previous_move and getattr(previous_move, "position", None):
            starting_fen = previous_move.position
        fen_after = move.position  # FEN after the played move

        # Convert played move to SAN
        played_san = self._move_uci_to_san(starting_fen, move.move)

        # Scores in pawns for readability
        prev_cp = ai_meta.get("prevScore", getattr(previous_move, "score", None))
        now_cp = ai_meta.get("scoreNow", move.score)
        swing_cp = None
        if isinstance(prev_cp, (int, float)) and isinstance(now_cp, (int, float)):
            swing_cp = now_cp - prev_cp

        # Build PV summaries (in SAN)
        best_pv = pvs_for_move[0] if pvs_for_move and len(pvs_for_move) > 0 else None
        second_pv = pvs_for_move[1] if pvs_for_move and len(pvs_for_move) > 1 else None

        best_line_san = self._format_pv_as_san(best_pv, starting_fen)
        second_line_san = self._format_pv_as_san(second_pv, starting_fen)

        # PV end features/scores if present
        def last_features_and_score(seq: Optional[List[Move]]) -> Tuple[Dict[str, Any], Optional[int]]:
            if not seq:
                return {}, None
            last = seq[-1]
            try:
                lf = last.hiddenFeatures or {}
            except Exception:
                lf = {}
            return (lf, getattr(last, "score", None))

        best_end_features, best_end_score = last_features_and_score(best_pv)
        second_end_features, second_end_score = last_features_and_score(second_pv)

        # Is played move equal to best PV first move?
        played_is_best = False
        try:
            played_is_best = bool(best_pv and len(best_pv) > 0 and getattr(best_pv[0], "move", None) == move.move)
        except Exception:
            played_is_best = False

        material = features_after.get("material", {}) if isinstance(features_after, dict) else {}
        open_files = features_after.get("openFiles", {}) if isinstance(features_after, dict) else {}
        white_features = features_after.get("white", {}) if isinstance(features_after, dict) else {}
        black_features = features_after.get("black", {}) if isinstance(features_after, dict) else {}

        # Build ASCII board from the position AFTER the played move
        ascii_board = board_to_ascii(fen_after)

        # Focused static context for the LLM – includes new strategic features
        static_context = {
            "material": material,
            "openFiles": open_files,
            "white": {k: white_features.get(k) for k in [
                "kingShieldPawns", "openFilesAdjacent", "semiOpenFilesAdjacent",
                "kingZoneAttacks", "mobility", "centerControl", "hasBishopPair",
                "connectedRooks", "passedPawns", "weakSquares", "outposts",
                "goodBishops", "badBishops", "space", "centralization",
                "kingExposure", "batteries", "doubledPawns", "isolatedPawns",
            ]},
            "black": {k: black_features.get(k) for k in [
                "kingShieldPawns", "openFilesAdjacent", "semiOpenFilesAdjacent",
                "kingZoneAttacks", "mobility", "centerControl", "hasBishopPair",
                "connectedRooks", "passedPawns", "weakSquares", "outposts",
                "goodBishops", "badBishops", "space", "centralization",
                "kingExposure", "batteries", "doubledPawns", "isolatedPawns",
            ]},
        }
        # Pawn structure classification (top-level)
        pawn_structure = features_after.get("pawnStructure") if isinstance(features_after, dict) else None

        score_gap_cp = None
        if isinstance(now_cp, (int, float)) and isinstance(best_end_score, (int, float)):
            score_gap_cp = best_end_score - now_cp

        compact: Dict[str, Any] = {
            "meta": {
                "moveId": move.id,
                "ply": move.depth,
                "movedBy": "white" if move.depth % 2 == 1 else "black",
                "opening": opening or {},
                "phase": move.phase or "unknown",
                "keyMomentType": key_moment_type,
            },
            "board": ascii_board,
            "fen": fen_after,
            "scores": {
                "prevCp": prev_cp,
                "nowCp": now_cp,
                "swingCp": swing_cp,
                "prevPawns": (prev_cp / 100.0) if isinstance(prev_cp, (int, float)) else None,
                "nowPawns": (now_cp / 100.0) if isinstance(now_cp, (int, float)) else None,
                "swingPawns": (swing_cp / 100.0) if isinstance(swing_cp, (int, float)) else None,
            },
            "current": {
                "san": played_san,
                "uci": move.move,
                "featuresDeltaTop": self._filter_feature_deltas(features_delta),
                "static": static_context,
                "pawnStructure": pawn_structure,
            },
            "pv": {
                "playedIsBest": played_is_best,
                "best": {
                    "lineSan": best_line_san,
                    "endScoreCp": best_end_score,
                },
                "second": {
                    "lineSan": second_line_san,
                    "endScoreCp": second_end_score,
                },
                "scoreGapCp": score_gap_cp,
            },
            "narrative": narrative_context,
        }
        return compact

    # ------------------------------------------------------------------
    # Structured-text builder (replaces raw JSON dump for LLM input)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_structured_text(compact: Dict[str, Any]) -> str:
        """Convert the compact input dict into structured text that LLMs
        comprehend better than raw JSON."""
        meta = compact.get("meta", {})
        scores = compact.get("scores", {})
        current = compact.get("current", {})
        pv = compact.get("pv", {})

        parts: List[str] = []

        # Game context
        opening = meta.get("opening") or {}
        opening_str = ""
        if opening.get("name"):
            eco = f" ({opening['eco']})" if opening.get("eco") else ""
            var = f", {opening['variation']}" if opening.get("variation") else ""
            opening_str = f"{opening['name']}{var}{eco}"
        move_num = (meta.get("ply", 0) + 1) // 2
        side = meta.get("movedBy", "?")
        phase = meta.get("phase", "unknown")
        km = meta.get("keyMomentType") or "none"

        parts.append(f"GAME CONTEXT:")
        if opening_str:
            parts.append(f"Opening: {opening_str}")
        mover_label = side.capitalize() if side in ("white", "black") else "?"
        to_move_after_label = (
            "Black" if side == "white" else "White" if side == "black" else "?"
        )
        parts.append(
            f"Move {move_num}, played by {mover_label} (FEN below has {to_move_after_label} to move), Phase: {phase}"
        )
        parts.append(f"Key moment: {km}")

        # Narrative
        narrative = compact.get("narrative", "")
        if narrative:
            parts.append(f"\nNARRATIVE:\n{narrative}")

        # Board
        board_str = compact.get("board", "")
        if board_str:
            parts.append(f"\nBOARD:\n{board_str}")

        fen = compact.get("fen", "")
        if fen:
            parts.append(f"\nFEN: {fen}")

        # Scores
        prev_p = scores.get("prevPawns")
        now_p = scores.get("nowPawns")
        swing_p = scores.get("swingPawns")
        parts.append(f"\nEVALUATION:")
        if prev_p is not None:
            parts.append(f"Score before: {prev_p:+.2f} pawns")
        if now_p is not None:
            parts.append(f"Score after: {now_p:+.2f} pawns")
        if swing_p is not None:
            parts.append(f"Score swing: {swing_p:+.2f} pawns")

        # Played move
        san = current.get("san", current.get("uci", "?"))
        parts.append(f"\nPLAYED MOVE: {san}")

        # Feature deltas
        deltas = current.get("featuresDeltaTop", [])
        if deltas:
            parts.append(f"\nFEATURE CHANGES:")
            for name, val in deltas:
                direction = "improved" if val > 0 else "worsened"
                parts.append(f"  {name}: {val:+.2f} ({direction})")

        # Pawn structure
        ps = current.get("pawnStructure")
        if ps and isinstance(ps, dict):
            parts.append(f"\nPAWN STRUCTURE: center={ps.get('centerType', '?')}")
            tension = ps.get("tension", [])
            if tension:
                parts.append(f"  Tension: {', '.join(tension[:4])}")
            breaks = ps.get("breaks", [])
            if breaks:
                parts.append(f"  Breaks: {', '.join(breaks[:4])}")

        # Static features (abbreviated)
        static = current.get("static", {})
        for side_label in ("white", "black"):
            s = static.get(side_label, {})
            if not s:
                continue
            items = []
            for k in ["mobility", "centerControl", "space", "centralization",
                       "kingExposure", "passedPawns", "weakSquares", "outposts",
                       "goodBishops", "badBishops", "batteries",
                       "kingShieldPawns", "kingZoneAttacks"]:
                v = s.get(k)
                if v is not None and v != 0 and v != [] and v != False:
                    if isinstance(v, list):
                        v = ", ".join(str(x) for x in v[:4]) if v else "none"
                    elif isinstance(v, dict):
                        occ = v.get("occupied", [])
                        avail = v.get("available", [])
                        v = f"occupied={occ}, available={avail}" if occ or avail else "none"
                    items.append(f"{k}={v}")
            if items:
                parts.append(f"\n{side_label.upper()} FEATURES: {', '.join(items)}")

        # PVs
        parts.append(f"\nPRINCIPAL VARIATIONS:")
        is_best = pv.get("playedIsBest", False)
        parts.append(f"Played move is best: {'yes' if is_best else 'no'}")
        best = pv.get("best", {})
        best_line = best.get("lineSan", best.get("lineUci", []))
        if best_line:
            parts.append(f"Best line: {' '.join(best_line)} (end score: {best.get('endScoreCp')} cp)")
        second = pv.get("second", {})
        second_line = second.get("lineSan", second.get("lineUci", []))
        if second_line:
            parts.append(f"Second line: {' '.join(second_line)} (end score: {second.get('endScoreCp')} cp)")
        gap = pv.get("scoreGapCp")
        if gap is not None:
            parts.append(f"Score gap vs best: {gap} cp")

        return "\n".join(parts)

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
    ) -> Optional[str]:
        """Make a single LLM call and return the raw text response."""
        kwargs: Dict[str, Any] = {
            "model": model or "gpt-5",
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
        # Extract text from response
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output") and response.output:
            try:
                return response.output[0].content[0].text
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def analyze_and_compose(
        self,
        compact_input: Dict[str, Any],
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        key_moment_type: Optional[str] = None,
    ) -> str:
        structured_text = self._build_structured_text(compact_input)
        return await self.analyze_and_compose_raw_text(
            structured_text,
            model=model,
            effort=effort,
            key_moment_type=key_moment_type,
        )


