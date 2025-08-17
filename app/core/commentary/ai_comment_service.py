from __future__ import annotations
import os
import json
from typing import Dict, Any

from openai import AsyncOpenAI

from app.models.Move import Move

AI_SYSTEM_PROMPT = (
    "You are an expert chess commentator. You will be given a JSON object with structured data for a single chess move. "
    "Your task is to provide a concise, insightful comment based on the data. The 'hiddenFeatures' object may contain a 'keyMomentType' which can be 'blunder', 'mistake', or 'missed_opportunity'.\n\n"
    "- If the `keyMomentType` is 'blunder' or 'mistake', explain *why* it's a bad move. Use the engine's principal variation (PV) to show the opponent's threat.\n"
    "- If the `keyMomentType` is 'missed_opportunity', explain the better move and what it would have achieved. Use the PV to illustrate the advantage.\n"
    "- Do not mention pieces that have already been captured. The board state is final.\n"
    "- Your response MUST be a single, valid JSON object and nothing else. Do not include markdown formatting or any text outside the JSON structure.\n\n"
    "Adhere strictly to the following JSON schema:\n"
    "{\n"
    "  \"version\": 1.0,\n"
    "  \"summary\": \"<string: A one-sentence summary of the move's strategic importance.>\",\n"
    "  \"bullets\": [\n"
    "    \"<string: A bullet point explaining the most critical tactical or positional consequence.>\",\n"
    "    \"<string: (Optional) A second bullet point on another key aspect.>`\n"
    "  ]\n"
    "}"
)


def build_ai_prompt(move: Move) -> str:
    # We need to provide the PVs to the AI, so we'll add them to the prompt data.
    # For now, we'll just send the first PV.
    pvs = move.pvs or []
    pv_data = []
    if pvs and pvs[0]:
        pv_data = [m.model_dump() for m in pvs[0]]

    data: Dict = {
        "moveId": move.id,
        "fen": move.position,
        "uci": move.move,
        "depth": move.depth,
        "score": move.score,
        "phase": move.phase,
        "hiddenFeatures": move.hiddenFeatures or {},
        "trace": move.trace or {},
        "pvs": pv_data,
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
    ) -> str:
        """
        Generates a humanized comment for a given move and its features.
        """
        if not self.client.api_key:
            return "AI comments are disabled. Please set the OPENAI_API_KEY environment variable."

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": AI_SYSTEM_PROMPT},
                    {"role": "user", "content": build_ai_prompt(move)},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if content:
                comment_json = json.loads(content)
                return comment_json.get("summary", "Could not parse AI comment.")
        except Exception as e:
            return f"Error generating AI comment: {e}"

        return "No comment generated."


