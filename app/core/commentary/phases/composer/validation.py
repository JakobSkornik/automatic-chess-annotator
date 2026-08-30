"""Post-validation of an LLM rendering against the fact contract, plus robust
extraction of the comment text from a (possibly messy) provider response."""

from __future__ import annotations

import json
import re

from app.models.comment_facts import CommentFacts

from .tokens import eval_token, pv_token

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


_SCHEMA_MARKERS = (
    "additionalProperties",
    '"type":"object"',
    '"type": "object"',
    '"properties"',
)


def validate_facts_comment(text: str, facts: CommentFacts) -> bool:
    """The fact contract: eval and PV tokens verbatim; alternative named; no
    ungrounded tactical assertion.

    Also rejects raw JSON / schema leakage so a malformed provider response can
    never reach the UI (it falls back to the deterministic template instead)."""
    t = _normalize(text)
    if not t:
        return False
    # Reject schema echoes and raw JSON objects outright.
    if t.lstrip().startswith("{") or any(m in t for m in _SCHEMA_MARKERS):
        return False
    ev = eval_token(facts)
    if ev and ev not in t:
        return False
    pv = pv_token(facts)
    if pv and _normalize(pv) not in t:
        return False
    alt = facts.better_alternative
    if alt is not None and alt.san and alt.san not in t:
        return False
    if facts.refutation_san and facts.refutation_san not in t:
        return False
    return not ungrounded_motif_terms(t, facts)


# Named tactics are checkable board facts, not shades of phrasing: either the
# expert module found one or nobody may assert it. Each entry is a concept and
# the surface forms a renderer might reach for.
_MOTIF_TERMS: dict[str, re.Pattern[str]] = {
    "pin": re.compile(r"\bpin(?:s|ned|ning)?\b", re.I),
    "fork": re.compile(r"\bfork(?:s|ed|ing)?\b", re.I),
    "skewer": re.compile(r"\bskewer(?:s|ed|ing)?\b", re.I),
    "discovered": re.compile(r"\bdiscovered\s+(?:attack|check)\b", re.I),
    "double check": re.compile(r"\bdouble\s+check\b", re.I),
    "back rank": re.compile(r"\bback[-\s]rank\b", re.I),
    "zugzwang": re.compile(r"\bzugzwang\b", re.I),
    "en passant": re.compile(r"\ben\s+passant\b", re.I),
    "stalemate": re.compile(r"\bstalemate\b", re.I),
    "deflection": re.compile(r"\bdeflect(?:s|ed|ion|ing)?\b", re.I),
    "zwischenzug": re.compile(r"\bzwischenzug|intermezzo\b", re.I),
    "windmill": re.compile(r"\bwindmill\b", re.I),
    "trapped": re.compile(r"\btrapp(?:ed|ing)\b", re.I),
    "battery": re.compile(r"\bbatter(?:y|ies)\b", re.I),
    "overload": re.compile(r"\boverload(?:s|ed|ing)?\b", re.I),
}


def _grounding_corpus(facts: CommentFacts) -> str:
    """Everything the renderer was actually told, as one searchable blob."""
    parts: list[str] = [facts.verdict or ""]
    for c in facts.claims:
        parts += [c.rule_id, c.text, c.text_state or ""]
    alt = facts.better_alternative
    if alt is not None:
        parts.append(alt.verdict or "")
        for c in alt.claims:
            parts += [c.rule_id, c.text, c.text_state or ""]
    return " ".join(parts)


def ungrounded_motif_terms(text: str, facts: CommentFacts) -> list[str]:
    """Named tactics the text asserts that no fact given to it supports.

    The structural checks above catch a renderer that drops a token, but not one
    that invents a bind: the phantom-pin class of error, where fluent prose
    asserts a tactic the expert module never found.
    """
    corpus = _grounding_corpus(facts)
    return [
        concept
        for concept, pattern in _MOTIF_TERMS.items()
        if pattern.search(text) and not pattern.search(corpus)
    ]


_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")


def _parse_text(raw: str) -> str:
    """Extract the comment text from the provider response.

    Robust to providers that echo the JSON schema and/or concatenate several
    objects: scans every balanced top-level ``{...}`` and returns the ``text``
    of the LAST object that carries a non-empty, non-schema ``text`` field.
    """
    if not raw:
        return ""
    # Happy path: the whole response is the object.
    try:
        obj = json.loads(raw)
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("text"), str)
            and obj["text"].strip()
        ):
            return obj["text"].strip()
    except Exception:
        pass
    # Scan for embedded objects; prefer the last valid {text: ...}.
    best = ""
    for m in _OBJ_RE.finditer(raw):
        chunk = m.group(0)
        if "additionalProperties" in chunk or '"properties"' in chunk:
            continue  # this is the echoed schema, not a response
        try:
            o = json.loads(chunk)
        except Exception:
            continue
        if isinstance(o, dict) and isinstance(o.get("text"), str) and o["text"].strip():
            best = o["text"].strip()
    if best:
        return best
    # Last resort: strip schema-looking lines and surrounding braces.
    cleaned = "\n".join(
        ln
        for ln in raw.splitlines()
        if "additionalProperties" not in ln
        and '"type"' not in ln
        and '"properties"' not in ln
    ).strip()
    return cleaned
