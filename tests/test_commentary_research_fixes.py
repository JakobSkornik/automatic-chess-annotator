"""Regression coverage for plan extraction and deterministic wording fixes."""

import chess

from app.core.commentary.features.plan_extractor import (
    PlanTag,
    _side_cluster_label,
    _targets_from_uci,
    build_plan_comparison,
    classify_pv_plan,
)
from app.core.commentary.phases.composer.template import render_facts_template
from app.models.comment_facts import Claim, CommentFacts


class _PM:
    def __init__(self, uci: str) -> None:
        self.move = uci


def _line(*ucis: str) -> list[_PM]:
    return [_PM(uci) for uci in ucis]


def test_wing_labels_use_same_files_for_white_and_black():
    board = chess.Board()
    assert _side_cluster_label(board, ["g4", "h5"], chess.WHITE) == "kingside_activity"
    assert _side_cluster_label(board, ["g4", "h5"], chess.BLACK) == "kingside_activity"
    assert _side_cluster_label(board, ["a4", "b5"], chess.WHITE) == "queenside_activity"
    assert _side_cluster_label(board, ["a4", "b5"], chess.BLACK) == "queenside_activity"


def test_plan_tags_ignore_opponent_moves_in_pv():
    board = chess.Board()
    # Black makes two central piece moves, but White only advances pawns.
    tags = classify_pv_plan(
        board,
        ["a2a3", "g8f6", "a3a4", "f6d5", "d2d3"],
    )
    assert PlanTag.PIECE_MANEUVER.value not in tags
    assert PlanTag.CENTRALIZATION.value not in tags


def test_repeated_targets_ignore_opponent_destinations():
    board = chess.Board()
    # Black's knight reaches f6 twice; White has no repeated destination.
    assert _targets_from_uci(
        board, ["b1c3", "g8f6", "c3b1", "f6g8", "g1f3", "g8f6"]
    ) == []


def test_played_comparison_uses_matching_pv_even_when_ranked_third():
    board = chess.Board()
    pvs = [
        _line("e2e4", "e7e5", "g1f3"),
        _line("d2d4", "d7d5", "c1f4"),
        _line("b1c3", "e7e5", "c3b1", "d7d6", "b1c3", "c7c6"),
    ]
    played_targets, _best_targets, played_seed, _best_seed, _played_chain, _best_chain, _played_tags, _best_tags = build_plan_comparison(
        board.fen(), pvs, "b1c3"
    )
    assert played_seed == "queenside_activity"
    assert played_targets == ["c3"]


def test_missing_played_pv_is_unknown():
    board = chess.Board()
    _pt, _bt, played_seed, _bs, _pc, _bc, _ptags, _btags = build_plan_comparison(
        board.fen(), [_line("e2e4", "e7e5")], "c2c4"
    )
    assert played_seed == "unknown"


def test_tradeoff_concession_has_grammatical_neutral_connector():
    facts = CommentFacts(
        ply=23,
        san="Nf3",
        uci="g1f3",
        mover="White",
        phase="mid",
        verdict="keeps the balance",
        claims=[
            Claim(
                rule_id="concession",
                text="Black leaves the bishop pair.",
                beneficiary="black",
                is_concession=True,
            )
        ],
    )
    text = render_facts_template(facts)
    assert "The trade-off is that Black leaves the bishop pair." in text
    assert "At the cost of Black leaves" not in text
