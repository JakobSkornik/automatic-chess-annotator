"""Regression checks for material, threat, and verdict wording audits."""

import chess

from app.core.commentary.features.guid_features.features.threats import (
    hanging,
    pins,
    weak_enemies,
)
from app.core.commentary.rules.context import _Ctx
from app.core.commentary.rules.material import describe_material
from app.core.commentary.rules.threat_rules import rule_threats
from app.core.commentary.rules.verdicts import verdict_for_transition
from app.models.comment_facts import FeatureDelta, FeatureDiff


def test_equal_value_piece_imbalance_does_not_claim_a_material_winner():
    # A rook is equal to a bishop and two pawns at the rule's material scale.
    board = chess.Board("4k3/8/8/8/8/8/pp6/3RK1b1 w - - 0 1")
    assert describe_material(board, changed=True) is None


def test_specialized_material_phrase_does_not_drop_subject_pawn_surplus():
    # White has a rook and pawn against a knight; calling this only an exchange
    # omits a real part of the material difference.
    board = chess.Board("4k1n1/8/8/8/8/8/P7/3RK3 w - - 0 1")
    text, beneficiary, _ = describe_material(board, changed=False)
    assert "rook" in text and "pawn" in text and "knight" in text
    assert "exchange" not in text
    assert beneficiary == "white"


def test_hanging_feature_detects_undefended_piece_and_excludes_pawn_defended_one():
    undefended = chess.Board("q2k4/8/8/8/8/8/8/R3K3 w - - 0 1")
    pawn_defended = chess.Board("4k3/3p4/2q5/8/8/8/8/2R1K3 w - - 0 1")
    assert hanging.count(undefended, chess.WHITE) == 1
    assert hanging.count(pawn_defended, chess.WHITE) == 0


def test_concrete_hanging_rejects_defended_target_even_with_two_attackers():
    # White's bishop and rook both attack the bishop, but Black's rook
    # defends it. Attack count alone does not prove a safe material gain.
    board = chess.Board("k7/8/8/3br3/8/8/6B1/3R3K w - - 0 1")
    bishop = chess.D5
    assert len(board.attackers(chess.WHITE, bishop)) == 2
    assert hanging.is_hanging(board, bishop, chess.WHITE)
    assert not hanging.is_concretely_hanging(board, bishop, chess.WHITE)


def test_concrete_hanging_rejects_a_pinned_pseudolegal_attacker():
    board = chess.Board("k3r3/8/8/8/8/8/n3R3/4K3 w - - 0 1")
    knight = chess.A2
    assert board.attackers(chess.WHITE, knight)
    assert not board.is_legal(chess.Move(chess.E2, knight))
    assert not hanging.is_concretely_hanging(board, knight, chess.WHITE)


def test_threat_counts_never_treat_a_checking_king_as_hanging_or_weak():
    board = chess.Board("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1")
    assert board.is_check()
    assert weak_enemies.count(board, chess.WHITE) == 0
    assert hanging.count(board, chess.WHITE) == 0


def test_concrete_hanging_accepts_legal_capture_of_undefended_piece():
    board = chess.Board("k7/4n3/8/8/8/8/8/4R1K1 w - - 0 1")
    assert hanging.squares(board, chess.WHITE) == ["e7"]


def test_rule_suppresses_heuristic_hanging_claim_without_concrete_capture():
    defended = chess.Board("k7/8/8/3br3/8/8/6B1/3R3K w - - 0 1")
    pinned = chess.Board("k3r3/8/8/8/8/8/n3R3/4K3 w - - 0 1")

    for leaf, victim_square in ((defended, chess.D5), (pinned, chess.A2)):
        start = leaf.copy(stack=False)
        start.remove_piece_at(victim_square)
        before = hanging.count(start, chess.WHITE)
        after = hanging.count(leaf, chess.WHITE)
        assert after > before  # The broad feature alone would fire.
        feature = FeatureDelta(
            name="WHITE_HANGING",
            delta_cp=after - before,
            before_cp=before,
            after_cp=after,
            flag_before=before,
            flag_after=after,
        )
        ctx = _Ctx(
            FeatureDiff(positive=[feature]),
            "mid",
            "WHITE",
            0,
            start_board=start,
            leaf_board=leaf,
        )
        assert rule_threats(ctx) == []


def test_rule_emits_hanging_claim_when_capture_is_legal_and_undefended():
    leaf = chess.Board("k7/4n3/8/8/8/8/8/4R1K1 w - - 0 1")
    start = leaf.copy(stack=False)
    start.remove_piece_at(chess.E7)
    before = hanging.count(start, chess.WHITE)
    after = hanging.count(leaf, chess.WHITE)
    feature = FeatureDelta(
        name="WHITE_HANGING",
        delta_cp=after - before,
        before_cp=before,
        after_cp=after,
        flag_before=before,
        flag_after=after,
    )
    ctx = _Ctx(
        FeatureDiff(positive=[feature]),
        "mid",
        "WHITE",
        0,
        start_board=start,
        leaf_board=leaf,
    )
    claim = rule_threats(ctx)[0]
    assert claim.beneficiary == "white"
    assert claim.text == "Black's knight on e7 is undefended and attacked."


def test_pin_feature_counts_enemy_absolute_pin():
    board = chess.Board("4k3/4n3/8/8/8/8/8/4R1K1 w - - 0 1")
    assert pins.count(board, chess.WHITE) == 1


def test_threat_claim_uses_resulting_position_and_correct_black_direction():
    change = FeatureDelta(
        name="BLACK_HANGING",
        delta_cp=-60,
        before_cp=0,
        after_cp=-60,
        flag_before=0,
        flag_after=1,
    )
    leaf = chess.Board("r2k4/8/8/8/8/8/8/N3K3 b - - 0 1")
    start = chess.Board("3k4/8/8/8/8/8/8/N3K3 b - - 0 1")
    ctx = _Ctx(
        FeatureDiff(negative=[change]),
        "mid",
        "WHITE",
        0,
        start_board=start,
        leaf_board=leaf,
    )
    claim = rule_threats(ctx)[0]
    assert claim.beneficiary == "black"
    assert claim.text == "White's knight on a1 is undefended and attacked."


def test_concrete_threat_accepts_legal_promotion_capture():
    board = chess.Board("k6r/6P1/8/8/8/8/8/K7 w - - 0 1")
    assert hanging.is_concretely_hanging(board, chess.H8, chess.WHITE)


def test_transition_verdict_handles_black_pov_and_equality_boundary():
    assert verdict_for_transition(0, -80, None, "Black") == "seizes a clear advantage for Black"
    assert verdict_for_transition(-80, -20, None, "Black") == "lets the advantage slip away to equality"
