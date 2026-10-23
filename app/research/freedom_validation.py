"""Mechanical and semantic validation for the prose-freedom experiment."""

from __future__ import annotations

import json
import re
from typing import Any

import chess


_PV_RE = re.compile(r"\[pv:([^\]]*)\]", re.IGNORECASE)
_VERDICTS = {"supported", "contradicted", "unsupported", "unassessable"}

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assertions", "overall"],
    "properties": {
        "assertions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "verdict", "evidence_ids", "reason"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "verdict": {"type": "string", "enum": sorted(_VERDICTS)},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        },
        "overall": {"type": "string", "enum": sorted(_VERDICTS)},
    },
}

REVIEW_SYSTEM = """You are a conservative evidence reviewer. Identify and judge every
factual claim in the actual commentary text, including claims about which side is to
move, timing, move order, evaluations, and causal relationships. Compare each claim
directly with the supplied evidence; do not treat the commentary's citations as proof
and do not rely on them without checking the cited evidence. Mark a claim supported,
contradicted, unsupported, or unassessable. Generic chess-pedagogy claims are
unassessable when the supplied evidence does not ground them. Return JSON matching
the supplied schema and include every claim, including failures. A review performed
by the same model that wrote the commentary is only a screening result, not ground
truth."""


def _catalog_ids(evidence: Any) -> set[str]:
    if not isinstance(evidence, dict) or not isinstance(evidence.get("catalog"), dict):
        return set()
    return {key for key in evidence["catalog"] if isinstance(key, str)}


def validate_output(
    output: dict[str, Any], evidence: dict[str, Any], require_citations: bool
) -> dict[str, Any]:
    """Run deterministic checks without pretending to assess prose truth."""
    issues: list[str] = []
    text = output.get("text") if isinstance(output, dict) else None
    citations = output.get("evidence_ids") if isinstance(output, dict) else None

    if not isinstance(text, str) or not text.strip():
        issues.append("text must be a non-empty string")
        text = "" if not isinstance(text, str) else text
    elif len(text.split()) > 1500:
        issues.append("text exceeds 1500 words")

    known_ids = _catalog_ids(evidence)
    valid_citations: list[str] = []
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        issues.append("evidence_ids must be a list of strings")
    else:
        if len(citations) != len(set(citations)):
            issues.append("evidence_ids must be unique")
        unknown = sorted({item for item in citations if item not in known_ids})
        if unknown:
            issues.append(f"unknown evidence_ids: {', '.join(unknown)}")
        valid_citations = [item for item in citations if item in known_ids]
    if require_citations and not valid_citations:
        issues.append("at least one valid evidence citation is required")

    pv_tokens = _PV_RE.findall(text)
    supplied_lines = evidence.get("lines", []) if isinstance(evidence, dict) else []
    exact_lines = {
        tuple(line["san"])
        for line in supplied_lines
        if isinstance(line, dict)
        and isinstance(line.get("san"), list)
        and all(isinstance(move, str) for move in line["san"])
    }
    fen = evidence.get("fen_before") if isinstance(evidence, dict) else None
    for index, token in enumerate(pv_tokens, start=1):
        san = token.split()
        if not san:
            issues.append(f"PV {index} is empty")
            continue
        if tuple(san) not in exact_lines:
            issues.append(f"PV {index} is not an exact supplied line")
        try:
            board = chess.Board(fen)
            for move in san:
                board.push_san(move)
        except (ValueError, TypeError, chess.InvalidMoveError, chess.IllegalMoveError) as exc:
            issues.append(f"PV {index} is not legal from fen_before: {exc}")

    return {
        "passed": not issues,
        "issues": issues,
        "checked_pv_count": len(pv_tokens),
        "semantic_status": "requires_review",
    }


def parse_review(raw: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Parse and strictly validate a semantic review response."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("review must be valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"assertions", "overall"}:
        raise ValueError("review must contain only assertions and overall")
    assertions = value["assertions"]
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("review assertions must be a non-empty list")
    known_ids = _catalog_ids(evidence)
    seen_text: set[str] = set()
    for index, assertion in enumerate(assertions, start=1):
        if not isinstance(assertion, dict) or set(assertion) != {
            "text",
            "verdict",
            "evidence_ids",
            "reason",
        }:
            raise ValueError(f"assertion {index} has invalid fields")
        claim = assertion["text"]
        reason = assertion["reason"]
        verdict = assertion["verdict"]
        ids = assertion["evidence_ids"]
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(f"assertion {index} text must be non-empty")
        normalized = claim.strip()
        if normalized in seen_text:
            raise ValueError(f"duplicate assertion text: {normalized}")
        seen_text.add(normalized)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"assertion {index} reason must be non-empty")
        if verdict not in _VERDICTS:
            raise ValueError(f"assertion {index} has invalid verdict")
        if (
            not isinstance(ids, list)
            or any(not isinstance(item, str) for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError(f"assertion {index} evidence_ids must be unique strings")
        unknown = sorted(set(ids) - known_ids)
        if unknown:
            raise ValueError(f"assertion {index} cites unknown evidence_ids: {', '.join(unknown)}")
        if verdict == "supported" and not ids:
            raise ValueError(f"assertion {index} marked supported without evidence")
    if value["overall"] not in _VERDICTS:
        raise ValueError("overall has invalid verdict")
    value["semantic_pass"] = (
        value["overall"] == "supported"
        and all(item["verdict"] == "supported" for item in assertions)
    )
    return value
