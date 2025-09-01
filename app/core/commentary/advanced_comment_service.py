from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.models.Move import Move


ANALYZER_SYSTEM_PROMPT = (
    "You are a chess analysis assistant. You receive compact JSON describing a move, evals, feature deltas, and principal variations (PVs).\n"
    "Your job is to analyze WHY the played move is good or not, which features matter, and what the alternative best line achieves.\n"
    "Rules:\n"
    "- Treat evals as authoritative and in WHITE POV centipawns; use provided pawns values.\n"
    "- Prefer large swings (>= 1.0 pawns), blunders/mistakes, initiative shifts, and concrete tactical/structural changes.\n"
    "- If opening context contains ECO/name/variation, incorporate it into your reasoning (e.g., typical plans, motifs).\n"
    "- Weigh features with larger absolute deltas higher.\n"
    "- Use alt/best PV summaries to explain 'what is lost' when the current move is not best.\n"
    "- Output strictly valid JSON.\n"
    "Output format: {\n"
    "  \"playedIsBest\": boolean,\n"
    "  \"salient\": [ { \"feature\": string, \"delta\": number, \"reason\": string } ... up to 4 ],\n"
    "  \"whyBetterAlt\": string,    // if not best: concise note of what the best line achieves\n"
    "  \"whyBest\": string,         // if best: concise note of why it's best\n"
    "  \"pvMoves\": string          // short PV move list text (e.g., \"e4, Nf3, Bb5\")\n"
    "}"
)


COMPOSER_SYSTEM_PROMPT = (
    "You are a professional chess annotator. You receive JSON analysis (salient features and rationale).\n"
    "Write ONE concise sentence suitable for PGN annotation. Use polished language.\n"
    "Rules:\n"
    "- Lead with who erred (if not best) or praise if best.\n"
    "- Mention eval swing in pawns if impactful.\n"
    "- Highlight 1-2 most important features.\n"
    "- If not best, mention what was lost versus best line; if best, say why it excels.\n"
    "- If opening context contains ECO/name/variation, begin by naming it (e.g., 'Najdorf, English Attack (B90)') when relevant.\n"
    "- Optionally end with \"PV begins with ...\" using pvMoves.\n"
    "- Output strictly valid JSON: {\"summary\": \"...\"}"
)


class AdvancedCommentService:
    def __init__(self) -> None:
        self.client = AsyncOpenAI()

    @staticmethod
    def _format_moves_list(pv: Optional[List[Move]], max_len: int = 6) -> List[str]:
        if not pv:
            return []
        result: List[str] = []
        for idx, m in enumerate(pv[:max_len]):
            try:
                if getattr(m, "move", None):
                    # UCI to a friendly-ish token; we keep UCI to be compact
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

        # Scores in pawns for readability
        prev_cp = ai_meta.get("prevScore", getattr(previous_move, "score", None))
        now_cp = ai_meta.get("scoreNow", move.score)
        swing_cp = None
        if isinstance(prev_cp, (int, float)) and isinstance(now_cp, (int, float)):
            swing_cp = now_cp - prev_cp

        # Build PV summaries
        best_pv = pvs_for_move[0] if pvs_for_move and len(pvs_for_move) > 0 else None
        second_pv = pvs_for_move[1] if pvs_for_move and len(pvs_for_move) > 1 else None

        best_line = self._format_moves_list(best_pv or [])
        second_line = self._format_moves_list(second_pv or [])

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

        compact = {
            "meta": {
                "moveId": move.id,
                "ply": move.depth,
                "movedBy": "white" if move.depth % 2 == 1 else "black",
                "opening": opening or {},
            },
            "scores": {
                "prevCp": prev_cp,
                "nowCp": now_cp,
                "swingCp": swing_cp,
                "prevPawns": (prev_cp / 100.0) if isinstance(prev_cp, (int, float)) else None,
                "nowPawns": (now_cp / 100.0) if isinstance(now_cp, (int, float)) else None,
                "swingPawns": (swing_cp / 100.0) if isinstance(swing_cp, (int, float)) else None,
            },
            "current": {
                "uci": move.move,
                "featuresDeltaTop": self._filter_feature_deltas(features_delta),
            },
            "pv": {
                "playedIsBest": played_is_best,
                "best": {
                    "lineUci": best_line,
                    "endScoreCp": best_end_score,
                },
                "second": {
                    "lineUci": second_line,
                    "endScoreCp": second_end_score,
                },
            },
        }
        return compact

    async def analyze_and_compose(
        self,
        compact_input: Dict[str, Any],
        *,
        model: Optional[str] = None,
        effort: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        if not self.client.api_key:
            return "AI comments are disabled. Please set the OPENAI_API_KEY environment variable."

        # Step 1: Analyzer
        analysis_json_str = json.dumps(compact_input)
        analysis = await self.client.responses.create(
            model=model or "gpt-5-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": ANALYZER_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": analysis_json_str}]},
            ],
            reasoning={"effort": (effort or "low")},
            # temperature=temperature,  # keep compact deterministic output
        )
        analyzer_content = None
        if hasattr(analysis, "output_text") and analysis.output_text:
            analyzer_content = analysis.output_text
        elif hasattr(analysis, "output") and analysis.output:
            try:
                analyzer_content = analysis.output[0].content[0].text
            except Exception:
                analyzer_content = None
        analysis_obj: Dict[str, Any] = {}
        try:
            if analyzer_content:
                analysis_obj = json.loads(analyzer_content)
        except Exception:
            analysis_obj = {}

        # Step 2: Composer
        composer = await self.client.responses.create(
            model=model or "gpt-5-mini",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": COMPOSER_SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(analysis_obj)}]},
            ],
            reasoning={"effort": (effort or "low")},
            # temperature=temperature,
        )
        content_final = None
        if hasattr(composer, "output_text") and composer.output_text:
            content_final = composer.output_text
        elif hasattr(composer, "output") and composer.output:
            try:
                content_final = composer.output[0].content[0].text
            except Exception:
                content_final = None
        if content_final:
            try:
                obj = json.loads(content_final)
                return obj.get("summary", "")
            except Exception:
                pass
        return ""


