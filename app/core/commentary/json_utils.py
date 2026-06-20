"""Small JSON helpers shared by the LLM commentary modules."""

from __future__ import annotations


def strip_json_fence(text: str) -> str:
    """Remove a leading/trailing ``` markdown code fence around a JSON payload."""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()
