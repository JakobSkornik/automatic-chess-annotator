"""Resolve concrete model ids per LLM provider and pipeline stage (server-side only)."""

from __future__ import annotations

from typing import Dict

_POLICY: Dict[str, Dict[str, str]] = {
    "openai": {
        "digest": "gpt-4.1-mini",
        "composer": "gpt-4.1-mini",
        "episode": "gpt-4.1-mini",
        "narrative": "gpt-4.1-mini",
    },
    "anthropic": {
        "digest": "claude-sonnet-4-5",
        "composer": "claude-haiku-4-5",
        "episode": "claude-haiku-4-5",
        "narrative": "claude-sonnet-4-5",
    },
}


def resolve_model(provider: str, stage: str) -> str:
    """Return model id for ``stage`` in ``digest|composer|episode|narrative``."""
    p = (provider or "openai").strip().lower()
    if p not in _POLICY:
        p = "openai"
    m = _POLICY[p].get(stage)
    if m:
        return m
    return _POLICY["openai"][stage]
