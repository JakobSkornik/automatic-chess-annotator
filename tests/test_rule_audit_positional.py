"""Regression coverage for positional claims that used to overstate counts."""

import chess
import pytest

from app.core.commentary.rules.context import _Ctx
from app.core.commentary.rules.pawn_rules import (
    rule_doubled_pawns,
    rule_passed_pawn,
    rule_passer_escort,
    rule_rook_behind_passer,
)
from app.core.commentary.rules.piece_rules import (
    rule_bad_bishop,
    rule_bishop_pair,
    rule_intent,
    rule_rook_activity,
    rule_strong_knight,
    rule_piece_activity,
    rule_king_safety,
)
from app.core.commentary.features.guid_features import compute_feature_vector
from app.core.commentary.features.guid_features.squares import outpost_squares
from app.core.commentary.features.envisioned import diff_vectors
from app.models.comment_facts import FeatureDelta, FeatureDiff


def _ctx(
    *deltas: FeatureDelta,
    start_board: chess.Board | None = None,
    leaf_board: chess.Board | None = None,
) -> _Ctx:
    positive = [delta for delta in deltas if delta.delta_cp >= 0]
    negative = [delta for delta in deltas if delta.delta_cp < 0]
    return _Ctx(
        FeatureDiff(positive=positive, negative=negative),
        phase="mid",
        mover="WHITE",
        eval_cp=0,
        start_board=start_board,
        leaf_board=leaf_board,
    )


def _delta(
    name: str,
    before: int,
    after: int,
    *,
    flag_before: int | None = None,
    flag_after: int | None = None,
) -> FeatureDelta:
    return FeatureDelta(
        name=name,
        delta_cp=after - before,
        before_cp=before,
        after_cp=after,
        flag_before=flag_before,
        flag_after=flag_after,
    )


@pytest.mark.parametrize("side,sign", [("WHITE", 1), ("BLACK", -1)])
@pytest.mark.parametrize("before,after,phrase", [(30, 40, "more active"), (40, 30, "less active")])
def test_activity_state_keeps_the_comparative_meaning(side, sign, before, after, phrase):
    claim = rule_piece_activity(_ctx(_delta(
        f"{side}_PIECE_ACTIVITY", sign * before, sign * after,
    )))[0]
    assert phrase in claim.text_state
    assert claim.beneficiary == (side.lower() if after > before else
                                 ("black" if side == "WHITE" else "white"))


@pytest.mark.parametrize("side,sign", [("WHITE", 1), ("BLACK", -1)])
def test_increased_king_pressure_is_comparative_in_both_wordings(side, sign):
    opponent = "BLACK" if side == "WHITE" else "WHITE"
    claim = rule_king_safety(_ctx(
        _delta(f"{side}_KING_DANGER", -10 * sign, -90 * sign),
        _delta(f"{opponent}_KING_TROPISM", -10 * sign, -20 * sign),
    ))[0]
    assert "more pressure" in claim.text
    assert "more pressure" in claim.text_state


def test_rook_activity_mentions_semi_open_files_and_uses_net_weighted_delta():
    claim = rule_rook_activity(
        _ctx(_delta("WHITE_ROOK_HALF_OPEN_FILE", 0, 1))
    )[0]
    assert "semi-open" in claim.text

    ctx = _ctx(
        _delta("WHITE_ROOK_OPEN_FILE", 1, 0),
        _delta("WHITE_ROOK_HALF_OPEN_FILE", 0, 2),
    )

    # Two semi-open rooks gained, one open-file rook lost: 2*10 - 20 = 0 cp.
    # The net count rises, but the weighted activity does not, so it is silent.
    assert rule_rook_activity(ctx) == []


def test_losing_one_strong_knight_does_not_claim_all_are_gone():
    ctx = _ctx(
        _delta(
            "WHITE_KNIGHTS_OUTPOSTS",
            2,
            1,
            flag_before=2,
            flag_after=1,
        )
    )

    claim = rule_strong_knight(ctx)[0]

    assert claim.rule_id == "strong_knight_lost"
    assert "no longer" not in claim.text
    assert "still has a strong knight" in claim.text_state


def test_bad_bishop_reduction_with_one_remaining_is_not_called_solved():
    ctx = _ctx(
        _delta("WHITE_BAD_BISHOP", 0, -40, flag_before=2, flag_after=1)
    )

    claim = rule_bad_bishop(ctx)[0]

    assert claim.rule_id == "bad_bishop_solved"
    assert "reduces" in claim.text
    assert "fewer bad bishops" in claim.text_state


def test_doubled_pawn_reduction_with_one_remaining_is_not_called_gone():
    ctx = _ctx(
        _delta("WHITE_PAWN_DOUBLED", 2, 1, flag_before=2, flag_after=1)
    )

    claim = rule_doubled_pawns(ctx)[0]

    assert claim.rule_id == "doubled_pawns_resolved"
    assert "reduces" in claim.text
    assert "still has doubled-pawn weaknesses" in claim.text_state


def test_passed_pawn_advance_claim_needs_board_evidence_of_forward_progress():
    ctx = _ctx(
        _delta("WHITE_PAWN_PASSED", 20, 35, flag_before=1, flag_after=1),
        start_board=chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"),
        leaf_board=chess.Board("4k3/8/8/4P3/8/8/8/4K3 w - - 0 1"),
    )

    claims = rule_passed_pawn(ctx)

    assert len(claims) == 1
    assert claims[0].rule_id == "passed_pawn_advances"
    assert "dangerously" not in claims[0].text
    assert "further advanced" in claims[0].text_state


def test_passed_pawn_advance_is_suppressed_without_matching_board_evidence():
    ctx = _ctx(
        _delta("WHITE_PAWN_PASSED", 20, 35, flag_before=1, flag_after=1)
    )

    assert rule_passed_pawn(ctx) == []


def test_intent_is_suppressed_without_the_actual_start_board():
    ctx = _ctx(leaf_board=chess.Board())

    assert rule_intent(ctx) == []


def test_same_color_bishops_do_not_form_a_pair_for_either_side():
    white = chess.Board("4k3/8/8/8/8/4B3/8/2B1K3 w - - 0 1")
    for board, side in ((white, "WHITE"), (white.mirror(), "BLACK")):
        vector = compute_feature_vector(board)
        assert vector[f"{side}_BISHOP_PAIR"].value_cp == 0
        assert vector[f"{side}_BISHOP_PAIR"].flag == 1


def test_bishop_pair_loss_tracks_square_color_diversity_for_both_sides():
    before = chess.Board("7k/8/8/8/8/8/8/2BB3K w - - 0 1")
    after = chess.Board("7k/8/8/8/8/4B3/8/2B4K w - - 0 1")

    for start, leaf, side, beneficiary in (
        (before, after, "WHITE", "black"),
        (before.mirror(), after.mirror(), "BLACK", "white"),
    ):
        diff = diff_vectors(
            compute_feature_vector(start), compute_feature_vector(leaf)
        )
        ctx = _Ctx(
            diff,
            phase="mid",
            mover=side,
            eval_cp=0,
            start_board=start,
            leaf_board=leaf,
        )
        claims = rule_bishop_pair(ctx)
        assert len(claims) == 1
        assert claims[0].beneficiary == beneficiary
        assert claims[0].flag_note == "2 -> 1"


def test_outpost_square_uses_the_count_predicate_when_enemy_pawn_is_blocked():
    board = chess.Board("7k/2p5/2P5/3N4/2P5/8/8/K7 w - - 0 1")
    threatened = chess.Board("7k/2p5/8/3N4/2P5/8/8/K7 w - - 0 1")

    for position, side, expected in (
        (board, chess.WHITE, ["d5"]),
        (board.mirror(), chess.BLACK, ["d4"]),
    ):
        assert outpost_squares(position, side) == expected
        prefix = "WHITE" if side == chess.WHITE else "BLACK"
        assert compute_feature_vector(position)[f"{prefix}_KNIGHTS_OUTPOSTS"].flag == 1

    for position, side in (
        (threatened, chess.WHITE),
        (threatened.mirror(), chess.BLACK),
    ):
        assert outpost_squares(position, side) == []
        prefix = "WHITE" if side == chess.WHITE else "BLACK"
        assert compute_feature_vector(position)[f"{prefix}_KNIGHTS_OUTPOSTS"].flag == 0


def test_rook_behind_passer_requires_an_unobstructed_file_for_both_sides():
    clear = chess.Board("7k/8/8/8/4P3/8/8/K3R3 w - - 0 1")
    blocked = chess.Board("7k/8/8/8/4P3/4N3/8/K3R3 w - - 0 1")

    for position, side, prefix in (
        (clear, "WHITE", "WHITE"),
        (clear.mirror(), "BLACK", "BLACK"),
    ):
        assert compute_feature_vector(position)[
            f"{prefix}_ROOK_BEHIND_PASSED_PAWN"
        ].flag == 1
    for position, side, prefix in (
        (blocked, "WHITE", "WHITE"),
        (blocked.mirror(), "BLACK", "BLACK"),
    ):
        assert compute_feature_vector(position)[
            f"{prefix}_ROOK_BEHIND_PASSED_PAWN"
        ].flag == 0

    diff = diff_vectors(compute_feature_vector(blocked), compute_feature_vector(clear))
    ctx = _Ctx(
        diff,
        phase="mid",
        mover="WHITE",
        eval_cp=0,
        start_board=blocked,
        leaf_board=clear,
    )
    assert [claim.rule_id for claim in rule_rook_behind_passer(ctx)] == [
        "rook_behind_passer"
    ]


def test_new_passer_near_stationary_king_does_not_claim_king_moved():
    start = chess.Board("7k/8/8/4P3/8/8/8/4K3 w - - 0 1")
    leaf = chess.Board("7k/8/8/4P3/8/8/5P2/4K3 w - - 0 1")

    for before, after, side, label in (
        (start, leaf, "WHITE", "White"),
        (start.mirror(), leaf.mirror(), "BLACK", "Black"),
    ):
        diff = diff_vectors(
            compute_feature_vector(before), compute_feature_vector(after)
        )
        ctx = _Ctx(
            diff,
            phase="end",
            mover=side,
            eval_cp=0,
            start_board=before,
            leaf_board=after,
        )
        claim = rule_passer_escort(ctx)[0]

        assert claim.text == f"{label} has a passed pawn close to its king."
        assert "moves closer" not in claim.text
