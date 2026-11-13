"""v19 statement_links: which words of a comment make which statement.

A link ties a phrase of the final comment text to a position of one of the two displayed
lines and to the squares the statement is about, so the interface can play the line to that
position and highlight them. All three writers produce the same form:

    {"phrase": str, "start": int, "part": "main" | "alt", "ply": int, "evidence": {...}}

``ply`` counts along the line (1 = after its first move). The full writer and the template
take position and squares from the checked statement; the base model names them itself and
nothing verifies them. A link that cannot be placed is dropped; the text is never changed.
"""
from __future__ import annotations

import re

from app.core.commentary import pipeline_flags

from . import narrative

MAX_LINKS = 12
MAX_SQUARES = 6
_SQUARE = re.compile(r"[a-h][1-8]")
_PART = {"main": "main", "alternative": "alt", "played": "main", "other": "alt"}
_BRANCH = {"main": "main", "alt": "alternative"}


def enabled() -> bool:
    return pipeline_flags.enabled("statement_links")


def _pattern(phrase: str) -> re.Pattern | None:
    """The phrase as it may stand in the final text: spacing is free, and a move may have
    gained its number since the writer copied the phrase."""
    phrase = " ".join(str(phrase).replace("’", "'").split())
    if len(phrase) < 3:
        return None
    number = r"(?:\d+\s?(?:\.{3}|…|\.)\s?)?"
    out, at = [], 0
    for match in narrative._NAMED_MOVE.finditer(phrase):
        out.append(re.escape(phrase[at:match.start()]))
        out.append(number + re.escape(match["san"]) + r"[+#]?")
        at = match.end()
    out.append(re.escape(phrase[at:]))
    return re.compile("".join(out).replace(r"\ ", r"\s+"))


def _place(text: str, phrase: str, taken: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Where the phrase stands in the text: the first place that no earlier link covers."""
    pattern = _pattern(phrase)
    if pattern is None:
        return None
    for match in pattern.finditer(text):
        span = match.span()
        if all(span[1] <= a or span[0] >= b for a, b in taken):
            return span
    return None


def _link(text, span, part, ply, evidence):
    return {"phrase": text[span[0]:span[1]], "start": span[0], "part": part, "ply": ply, "evidence": evidence}


def _finish(links: list[dict]) -> list[dict]:
    return sorted(links, key=lambda link: link["start"])[:MAX_LINKS]


def _claims(facts, part):
    branch = facts if part == "main" else facts.better_alternative
    return list(branch.claims) if branch is not None else []


def _claim_evidence(claim) -> dict:
    evidence = getattr(claim, "evidence", None)
    return evidence.model_dump() if evidence is not None else {"squares": []}


def for_claims(proposed, facts, text: str) -> list[dict]:
    """The full writer's links: each phrase points to one statement it cites, and the
    statement supplies the position and the squares."""
    links, taken, used = [], [], set()
    for item in proposed or []:
        if not isinstance(item, dict):
            continue
        part = _PART.get(item.get("branch"))
        claims = _claims(facts, part) if part else []
        index = item.get("id")
        if type(index) is not int or not 0 <= index < len(claims) or (part, index) in used:
            continue
        span = _place(text, item.get("phrase") or "", taken)
        claim = claims[index]
        if span is None or not claim.support_ply:
            continue
        used.add((part, index))
        taken.append(span)
        links.append(_link(text, span, part, int(claim.support_ply), _claim_evidence(claim)))
    return _finish(links)


def _wordings(claim) -> list[str]:
    """How the template may have written a statement: whole, or without the explanation
    after its colon (the template explains a notion once), with either first letter."""
    out = []
    for text in (claim.text, getattr(claim, "text_state", None)):
        if not text:
            continue
        whole = text.strip().rstrip(".")
        for wording in (whole, whole.split(":")[0]):
            for variant in (wording, wording[:1].lower() + wording[1:]):
                if variant and variant not in out:
                    out.append(variant)
    return out


def for_template(facts, text: str) -> list[dict]:
    """The template's links, found again in its text: the first paragraph holds the
    statements of the played line, the second those of the other move."""
    links, taken = [], []
    paragraphs, offset = [], 0
    for paragraph in text.split("\n"):
        paragraphs.append((offset, offset + len(paragraph)))
        offset += len(paragraph) + 1
    for part, bounds in zip(("main", "alt"), paragraphs):
        for claim in _claims(facts, part):
            if not claim.support_ply:
                continue
            span = None
            for wording in _wordings(claim):
                pattern = _pattern(wording)
                found = pattern and next((m.span() for m in pattern.finditer(text, bounds[0], bounds[1])
                                          if all(m.end() <= a or m.start() >= b for a, b in taken)), None)
                if found:
                    span = found
                    break
            if span:
                taken.append(span)
                links.append(_link(text, span, part, int(claim.support_ply), _claim_evidence(claim)))
    return _finish(links)


def _line_ply(evidence, part: str, move) -> int | None:
    """The place of a move in a displayed line, written with or without its number."""
    anchors = [event["anchor"] for event in (evidence.get("traces") or {}).get(_BRANCH[part]) or []]
    match = narrative._NAMED_MOVE.search(str(move or "")) or re.search(r"(?P<san>[a-h][1-8](?:=[QRBN])?)", str(move or ""))
    if not match:
        return None
    key = narrative._san_key(match["san"])
    hits = [i for i, anchor in enumerate(anchors) if narrative._san_key(anchor.split(".")[-1]) == key]
    written = re.match(r"\s*(\d+)\s?(\.{3}|…|\.)", str(move))
    if written:
        label = f"{int(written[1])}{'.' if written[2] == '.' else '...'}"
        hits = [i for i in hits if anchors[i].startswith(label) and anchors[i][len(label):len(label) + 1] != "."] or hits
    return hits[0] + 1 if hits else None


def for_base(statements, evidence, text: str) -> list[dict]:
    """The base model's links, as it reported them: its own line, move and squares. Only
    the form is checked (the phrase is in the text, the move is in that line, the squares
    are squares)."""
    links, taken = [], []
    for item in statements or []:
        if not isinstance(item, dict):
            continue
        part = _PART.get(item.get("line"))
        ply = _line_ply(evidence, part, item.get("after_move")) if part else None
        if ply is None:
            continue
        span = _place(text, item.get("phrase") or "", taken)
        if span is None:
            continue
        squares = sorted({s for s in item.get("squares") or [] if isinstance(s, str) and _SQUARE.fullmatch(s)})
        taken.append(span)
        links.append(_link(text, span, part, ply, {"squares": squares[:MAX_SQUARES]}))
    return _finish(links)
