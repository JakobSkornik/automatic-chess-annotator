from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI

from app.models.Move import Move

AI_SYSTEM_PROMPT = (
    "You are an expert chess commentator. You receive JSON for each move with evals in centipawns from WHITE's point of view.\n"
    "STRICT RULES:\n"
    "- Treat score fields as authoritative: use prevScore -> scoreNow (both white POV) and scoreDelta.\n"
    "- If scoreDelta > 0, position improved for White; if < 0, improved for Black.\n"
    "- Use movedBy and sideBenefited/moverMistake if present to label blunders correctly.\n"
    "- Prefer evalBeforePawns/evalAfterPawns if present; otherwise divide centipawns by 100 and format to 2 decimals.\n"
    "- Mention a concrete feature from featuresDelta and, if helpful, the first PV move.\n"
    "- Keep it to ONE short sentence.\n\n"
    "Output ONLY valid JSON: {\"summary\": \"...\" }"
)


def build_ai_prompt(move: Move, pvs_for_move: Optional[List[List[Move]]] = None) -> str:
    # We need to provide the PVs to the AI, so we'll add them to the prompt data.
    # For now, we'll just send the first PV.
    pvs = pvs_for_move or []
    pv_data = []
    if pvs and pvs[0]:
        pv_data = [m.model_dump() for m in pvs[0]]

    # Try to extract before/after features and deltas prepared for AI
    features_after = move.hiddenFeatures or {}
    ai_meta = {}
    features_before = {}
    features_delta = {}
    try:
        if isinstance(features_after, dict):
            ai_meta = features_after.get("_ai", {}) or {}
            features_before = ai_meta.get("before", {}) or {}
            features_delta = ai_meta.get("delta", {}) or {}
    except Exception:
        pass

    # Pull score context from meta if available
    prev_score = None
    score_delta = None
    score_trend = None
    try:
        if isinstance(ai_meta, dict):
            prev_score = ai_meta.get("prevScore")
            score_delta = ai_meta.get("scoreDelta")
            score_trend = ai_meta.get("scoreTrend")
            # Additional explicit meta to avoid POV confusion in the LLM
            score_now = ai_meta.get("scoreNow", move.score)
            score_pov = ai_meta.get("scorePov", "white")
            moved_by = ai_meta.get("movedBy")
            side_benefited = ai_meta.get("sideBenefited")
            mover_mistake = ai_meta.get("moverMistake")
            eval_before_pawns = ai_meta.get("evalBeforePawns")
            eval_after_pawns = ai_meta.get("evalAfterPawns")
            score_swing_pawns = ai_meta.get("scoreSwingPawns")
            swing_pawns_abs = ai_meta.get("swingPawnsAbs")
        else:
            score_now = move.score
            score_pov = "white"
            moved_by = None
            side_benefited = None
            mover_mistake = None
            eval_before_pawns = None
            eval_after_pawns = None
            score_swing_pawns = None
            swing_pawns_abs = None
    except Exception:
        score_now = move.score
        score_pov = "white"
        moved_by = None
        side_benefited = None
        mover_mistake = None
        eval_before_pawns = None
        eval_after_pawns = None
        score_swing_pawns = None
        swing_pawns_abs = None

    # PV score context
    pvs_top1_score = None
    pvs_top1_score_diff = None
    try:
        if pvs and len(pvs) > 0 and len(pvs[0]) > 0 and hasattr(pvs[0][0], "score"):
            pvs_top1_score = pvs[0][0].score
            if move.score is not None and pvs_top1_score is not None:
                pvs_top1_score_diff = pvs_top1_score - move.score
    except Exception:
        pass

    data: Dict = {
        "moveId": move.id,
        "fen": move.position,
        "uci": move.move,
        "depth": move.depth,
        "score": move.score,
        "prevScore": prev_score,
        "scoreDelta": score_delta,
        "scoreTrend": score_trend,
        "scoreNow": score_now,
        "scorePov": score_pov,
        "movedBy": moved_by,
        "sideBenefited": side_benefited,
        "moverMistake": mover_mistake,
        "evalBeforePawns": eval_before_pawns,
        "evalAfterPawns": eval_after_pawns,
        "scoreSwingPawns": score_swing_pawns,
        "swingPawnsAbs": swing_pawns_abs,
        "phase": move.phase,
        "hiddenFeatures": features_after,
        "featuresBefore": features_before,
        "featuresDelta": features_delta,
        "trace": move.trace or {},
        "pvs": pv_data,
        "pvsTop1Score": pvs_top1_score,
        "pvsTop1ScoreDiff": pvs_top1_score_diff,
    }
    return json.dumps(data)


class AICommentService:
    """
    A service for generating humanized comments using an AI model.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI()

    async def generate_comment(
        self,
        move: Move,
        features: Dict[str, Any],
        pvs_for_move: Optional[List[List[Move]]] = None,
    ) -> str:
        """
        Generates a humanized comment for a given move and its features.
        """
        if not self.client.api_key:
            return "AI comments are disabled. Please set the OPENAI_API_KEY environment variable."

        # Prefer the Responses API for GPT-5; fall back to chat.completions if not available
        try:
            try:
                # New Responses API
                response = await self.client.responses.create(
                    model="gpt-5-mini",
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": AI_SYSTEM_PROMPT}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": build_ai_prompt(move, pvs_for_move)}
                            ],
                        },
                    ],
                    reasoning={"effort": "low"},
                )

                # Try common fields for Responses API
                content = None
                if hasattr(response, "output_text") and response.output_text:
                    content = response.output_text
                elif hasattr(response, "output") and response.output:
                    # Some SDKs return a list of output items
                    try:
                        content = response.output[0].content[0].text
                    except Exception:
                        content = None
                if content:
                    comment_json = json.loads(content)
                    return comment_json.get("summary", "Could not parse AI comment.")
            except Exception as e:
                print(f"Error in Responses API: {e}")
                # Fallback to Chat Completions API
                response = await self.client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": AI_SYSTEM_PROMPT},
                        {"role": "user", "content": build_ai_prompt(move, pvs_for_move)},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=60,
                )
                content = response.choices[0].message.content
                if content:
                    comment_json = json.loads(content)
                    return comment_json.get("summary", "Could not parse AI comment.")
        except Exception as e:
            return f"Error generating AI comment: {e}"

        return "No comment generated."


