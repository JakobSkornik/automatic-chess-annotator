"""Resolve concrete model ids per LLM provider and pipeline stage (server-side only)."""

from __future__ import annotations
import os

_POLICY: dict[str, dict[str, str]] = {
    "openai": {"composer": "gpt-6-luna"},
    "anthropic": {"composer": "claude-haiku-5-5"},
    "cursor": {"composer": "composer-2.5"},
}

def resolve_model(provider: str, stage: str, pass_label: str | None = None) -> str:
    """Return model id for ``stage`` (currently only ``composer``).

    ``pass_label`` remains accepted for compatibility. All passes share one model
    so a game's conversation cannot silently change models between turns.
    """
    p = (provider or "openai").strip().lower()
    if p not in _POLICY:
        p = "openai"
    if stage == "composer":
        override = os.environ.get(f"COMMENTARY_{p.upper()}_MODEL", "").strip()
        if override:
            prefixes = {"openai": ("gpt-", "o1", "o3", "o4"),
                        "anthropic": ("claude-",), "cursor": ("composer-", "gpt-", "claude-")}
            if not override.startswith(prefixes[p]):
                raise ValueError(f"COMMENTARY_{p.upper()}_MODEL does not match provider {p}")
            return override
    m = _POLICY[p].get(stage)
    if m:
        return m
    return _POLICY["openai"][stage]
