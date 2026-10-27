import json

import pytest

from app.research.freedom_validation import (
    REVIEW_SCHEMA,
    REVIEW_SYSTEM,
    parse_review,
    validate_output,
)


EVIDENCE = {
    "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "move_san": "e4",
    "catalog": {"eval-1": {"cp": 20}, "feature-1": {"center": True}},
    "lines": [{"san": ["e4", "e5", "Nf3"]}, {"san": ["d4", "d5"]}],
}


def test_validate_output_accepts_cited_prose_and_exact_legal_pv():
    result = validate_output(
        {"text": "A possible continuation is [pv:e4 e5 Nf3].", "evidence_ids": ["eval-1"]},
        EVIDENCE,
        require_citations=True,
    )
    assert result == {
        "passed": True,
        "issues": [],
        "checked_pv_count": 1,
        "semantic_status": "requires_review",
    }


def test_validate_output_rejects_structure_citations_and_overlong_text():
    result = validate_output(
        {"text": "word " * 1501, "evidence_ids": ["missing", "missing"]},
        EVIDENCE,
        require_citations=True,
    )
    assert not result["passed"]
    assert any("1500" in issue for issue in result["issues"])
    assert any("unique" in issue for issue in result["issues"])
    assert any("unknown" in issue for issue in result["issues"])
    assert any("required" in issue for issue in result["issues"])


def test_validate_output_allows_uncited_free_arm_but_checks_every_pv():
    result = validate_output(
        {"text": "[pv:e4 e5] and [pv:e4 e5 Qh6]", "evidence_ids": []},
        EVIDENCE,
        require_citations=False,
    )
    assert result["checked_pv_count"] == 2
    assert not result["passed"]
    assert sum("exact supplied line" in issue for issue in result["issues"]) == 2
    assert any("not legal" in issue for issue in result["issues"])


def test_validate_output_does_not_claim_to_verify_factual_prose():
    result = validate_output(
        {"text": "Black has a forced mate because it is Tuesday.", "evidence_ids": ["eval-1"]},
        EVIDENCE,
        require_citations=True,
    )
    assert result["passed"]
    assert result["semantic_status"] == "requires_review"


def _review(assertions, overall="supported"):
    return json.dumps({"assertions": assertions, "overall": overall})


def test_parse_review_passes_only_fully_supported_review_and_preserves_assertions():
    assertions = [
        {"text": "White can play e4.", "verdict": "supported", "evidence_ids": ["eval-1"], "reason": "The evidence supports it."},
        {"text": "The line continues e5.", "verdict": "unsupported", "evidence_ids": [], "reason": "No cited fact establishes the prose claim."},
    ]
    result = parse_review(_review(assertions, "unsupported"), EVIDENCE)
    assert result["assertions"] == assertions
    assert result["semantic_pass"] is False

    assertions[1]["verdict"] = "supported"
    assertions[1]["evidence_ids"] = ["feature-1"]
    assert parse_review(_review(assertions), EVIDENCE)["semantic_pass"] is True


@pytest.mark.parametrize(
    "raw, message",
    [
        (_review([]), "non-empty"),
        (_review([{"text": "x", "verdict": "supported", "evidence_ids": [], "reason": "why"}]), "without evidence"),
        (_review([{"text": "x", "verdict": "supported", "evidence_ids": ["missing"], "reason": "why"}]), "unknown"),
        (_review([{"text": "x", "verdict": "unsupported", "evidence_ids": [], "reason": ""}]), "reason"),
    ],
)
def test_parse_review_rejects_vacuous_or_malformed_reviews(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_review(raw, EVIDENCE)


def test_parse_review_rejects_duplicate_assertions():
    assertion = {"text": "same", "verdict": "unassessable", "evidence_ids": [], "reason": "Not grounded."}
    with pytest.raises(ValueError, match="duplicate"):
        parse_review(_review([assertion, assertion], "unassessable"), EVIDENCE)


def test_review_contract_requires_claim_coverage_and_disclaims_ground_truth():
    assert REVIEW_SCHEMA["properties"]["assertions"]["minItems"] == 1
    lowered = REVIEW_SYSTEM.lower()
    assert "every" in lowered and "causal" in lowered
    assert "same model" in lowered and "not ground\ntruth" in lowered
