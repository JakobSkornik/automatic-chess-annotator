"""Post-process enforcement for phrases removed from the composer system prompt."""

from __future__ import annotations

import re
from typing import List, Tuple

# NOTE: "holds the balance" is deliberately absent — it is a sanctioned
# transition verdict produced by the rule engine (rules/engine.py).
FORBIDDEN_REGEX = re.compile(
    r"engine confirms|engine preference|engine's top choice|settles near|settling near|"
    r"preserves the rhythm|drives the rhythm|stalls the initiative|"
    r"flows through|keeps matters level|king tuck",
    re.I,
)


def forbidden_hit_count(text: str) -> int:
    return len(FORBIDDEN_REGEX.findall(text or ""))


def forbidden_hit_strings(text: str) -> List[str]:
    """Unique matched substrings (lowered for display), preserve first-seen order."""
    seen: set[str] = set()
    out: List[str] = []
    for m in FORBIDDEN_REGEX.finditer(text or ""):
        s = m.group(0).strip()
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def scrub_forbidden(text: str) -> Tuple[str, List[str]]:
    """Remove forbidden phrases; returns (cleaned_text, unique hits before removal)."""
    hits = forbidden_hit_strings(text)
    cleaned = FORBIDDEN_REGEX.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), hits
