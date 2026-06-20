"""Resolve concrete model ids per LLM provider and pipeline stage (server-side only)."""

from __future__ import annotations

_POLICY: dict[str, dict[str, str]] = {
    "openai": {"composer": "gpt-4.1-mini"},
    "anthropic": {"composer": "claude-haiku-4-5"},
    "cursor": {"composer": "composer-2.5"},
}

# When ``pass_label`` is passed for composer, override base composer model (optional per provider).
_POLICY_COMPOSER_BY_PASS: dict[str, dict[str, str]] = {
    "openai": {
        "key_moment": "gpt-4.1",
        "teaching": "gpt-4.1-mini",
    },
}


def resolve_model(provider: str, stage: str, pass_label: str | None = None) -> str:
    """Return model id for ``stage`` (currently only ``composer``).

    ``pass_label`` may be ``key_moment`` or ``teaching`` to pick a tiered composer model.
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
