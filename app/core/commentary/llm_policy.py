"""Resolve concrete model ids per LLM provider and pipeline stage (server-side only)."""

from __future__ import annotations

from typing import Dict, Optional

_POLICY: Dict[str, Dict[str, str]] = {
    "openai": {
        "digest": "gpt-4.1",
        "composer": "gpt-4.1-mini",
        "episode": "gpt-4.1-mini",
        "narrative": "gpt-4.1",
    },
    "anthropic": {
        "digest": "claude-sonnet-4-5",
        "composer": "claude-haiku-4-5",
        "episode": "claude-haiku-4-5",
        "narrative": "claude-sonnet-4-5",
    },
}

# When ``pass_label`` is passed for composer, override base composer model (optional per provider).
_POLICY_COMPOSER_BY_PASS: Dict[str, Dict[str, str]] = {
    "openai": {
        "key_moment": "gpt-4.1",
        "teaching": "gpt-4.1-mini",
    },
}


def resolve_model(provider: str, stage: str, pass_label: Optional[str] = None) -> str:
    """Return model id for ``stage`` in ``digest|composer|episode|narrative``.

    For ``composer`` only, ``pass_label`` may be ``key_moment`` or ``teaching`` to pick a tiered model.
    """
    p = (provider or "openai").strip().lower()
    if p not in _POLICY:
        p = "openai"
    if stage == "composer" and pass_label:
        sub = _POLICY_COMPOSER_BY_PASS.get(p, {}).get(pass_label.strip().lower())
        if sub:
            return sub
    m = _POLICY[p].get(stage)
    if m:
        return m
    return _POLICY["openai"][stage]
