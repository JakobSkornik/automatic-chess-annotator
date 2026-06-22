"""Post-process enforcement for phrases removed from the composer system prompt."""

from __future__ import annotations

import re

# NOTE: "holds the balance" is deliberately absent — it is a sanctioned
# transition verdict produced by the rule engine (rules/engine.py).
FORBIDDEN_REGEX = re.compile(
    r"engine confirms|engine preference|engine's top choice|settles near|settling near|"
    r"preserves the rhythm|drives the rhythm|stalls the initiative|"
    r"flows through|keeps matters level|king tuck",
    re.I,
)


def forbidden_hit_strings(text: str) -> list[str]:
    """Unique matched substrings (lowered for display), preserve first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for m in FORBIDDEN_REGEX.finditer(text or ""):
        s = m.group(0).strip()
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# Grammatical debris a mid-sentence removal can leave, e.g. removing "engine's
# top choice" from "is the engine's top choice," yields "is the ,".
_DANGLING_DET = re.compile(r"\b(is|was|are|were)\s+(?:the|a|an)\s*(?=[,.;:])", re.I)


def scrub_forbidden(text: str) -> tuple[str, list[str]]:
    """Remove forbidden phrases; returns (cleaned_text, unique hits before removal)."""
    hits = forbidden_hit_strings(text)
    cleaned = FORBIDDEN_REGEX.sub("", text or "")
    # Tidy any "is the ," / "was a ." left where a phrase was removed.
    cleaned = _DANGLING_DET.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), hits
