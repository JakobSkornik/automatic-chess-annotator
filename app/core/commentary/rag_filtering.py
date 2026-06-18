"""Quality filters for retrieved master-game annotations (RAG).

Decide whether a retrieved annotation is clean English prose worth feeding to
the composer, and trim it to a sentence boundary. Pure functions, no I/O beyond
reading the score threshold from the environment.
"""

from __future__ import annotations

import os
import re
import unicodedata

DEFAULT_MIN_SCORE = 0.62
MIN_ANNOTATION_WORDS = 20
MAX_NON_ASCII_LETTER_RATIO = 0.03
MAX_SPANISH_HINTS = 3
SAN_HEAVY_WORD_RATIO = 0.6
MIN_IDEA_OVERLAP_TOKENS = 3
MIN_OVERLAP_TOKEN_LEN = 3

_SPANISH_HINT_RE = re.compile(
    r"\b(que|las|los|del|por|para|una|unos|muy|como|esta|está|fueron|decidieron|fin)\b",
    re.I,
)
_SAN_MOVE_RE = re.compile(r"^[NBRQK]?[a-h]?x?[a-h][1-8](?:=[NBRQ])?[+#]?$")
_MOVE_NUMBER_RE = re.compile(r"^[1-9]\d*\.\.\.?$")
_SECTION_HEADER_RE = re.compile(r"^[AB]\)\s*\d")


def rag_min_score() -> float:
    """Minimum retrieval score for an annotation to be considered (env override)."""
    raw = os.environ.get("RAG_MIN_SCORE", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_MIN_SCORE


def _non_ascii_letter_ratio(text: str) -> float:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) > 127) / len(letters)


def _looks_like_section_header_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _SECTION_HEADER_RE.match(stripped):
        return True
    low = stripped.lower()
    return low.startswith("now we will look") or low.startswith("here are two")


def _is_san_heavy(words: list[str]) -> bool:
    san_like = sum(
        1 for w in words if _SAN_MOVE_RE.match(w) or _MOVE_NUMBER_RE.match(w)
    )
    return san_like >= len(words) * SAN_HEAVY_WORD_RATIO


def rag_annotation_passes_filters(text: str) -> bool:
    """True for clean, English, prose-like annotations worth reusing."""
    raw = (text or "").strip()
    if not raw:
        return False
    words = raw.split()
    if len(words) < MIN_ANNOTATION_WORDS:
        return False
    lines = raw.splitlines()
    if lines and _looks_like_section_header_line(lines[0]):
        return False
    if _non_ascii_letter_ratio(raw) > MAX_NON_ASCII_LETTER_RATIO:
        return False
    if len(_SPANISH_HINT_RE.findall(raw)) >= MAX_SPANISH_HINTS:
        return False
    return not _is_san_heavy(words)


def truncate_annotation_at_sentence(raw: str, cap: int) -> str:
    """Trim ``raw`` to at most ``cap`` chars, preferring a sentence boundary."""
    if len(raw) <= cap:
        return raw
    chunk = raw[:cap]
    for sep in ("\n", ". ", "! ", "? "):
        idx = chunk.rfind(sep)
        if idx > cap // 4:
            if sep == "\n":
                return chunk[:idx].rstrip()
            return chunk[: idx + 1].rstrip()
    return chunk.rstrip()


def rag_idea_overlap_tokens(idea: str, snippets: list[str]) -> bool:
    """True when the idea shares enough content words with the retrieved snippets."""
    idea_tokens = _content_tokens(idea)
    if len(idea_tokens) < MIN_IDEA_OVERLAP_TOKENS:
        return False
    snippet_tokens = _content_tokens(" ".join(snippets))
    return len(idea_tokens & snippet_tokens) >= MIN_IDEA_OVERLAP_TOKENS


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(t) >= MIN_OVERLAP_TOKEN_LEN
    }
