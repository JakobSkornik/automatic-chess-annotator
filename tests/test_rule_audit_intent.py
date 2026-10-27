"""Board-level hard negatives and legal positive examples for intent rules."""

import chess
import pytest

from app.core.commentary.features.envisioned import diff_vectors
from app.core.commentary.features.guid_features import compute_feature_vector
from app.core.commentary.features.intent import (
    build_intent_claim,
    file_opened_for_rook,
    file_semi_opened_for_rook,
    newly_attacked_enemy_piece,
    pawn_break_prepared,
)
from app.core.commentary.rules.catalog import run_rules


def transition(fen, moves, mirror=False):
    before = chess.Board(fen)
    after = before.copy()
    for san in moves:
        after.push_san(san)  # Reject illegal test fixtures.
    return (before.mirror(), after.mirror()) if mirror else (before, after)


@pytest.mark.parametrize("mirror", [False, True])
def test_displayed_five_ply_a6_line_does_not_open_file(mirror):
    before, after = transition(
        "rn1qkb1r/pb3ppp/1p1ppn2/8/2PNP3/2N2P2/PP2B1PP/R1BQK2R b KQkq - 0 8",
        ["Be7", "b3", "O-O", "O-O", "a6"], mirror,
    )
    mover = before.turn
    assert len(chess.SquareSet(after.pawns & chess.BB_FILE_A)) == 2
    assert file_opened_for_rook(before, after, mover) is None
    assert file_semi_opened_for_rook(before, after, mover) is None
    claims = run_rules(
        diff_vectors(compute_feature_vector(before), compute_feature_vector(after)),
        phase="mid", mover="WHITE" if mover else "BLACK", eval_cp=20,
        start_board=before, leaf_board=after, cap=False,
    )
    assert not any(c.rule_id in {"intent_opens_file", "intent_semi_opens_file"} for c in claims)


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("enemy_a_pawn", [False, True])
def test_capture_off_file_distinguishes_open_and_semi_open(mirror, enemy_a_pawn):
    before, after = transition(
        f"4k3/{'p7' if enemy_a_pawn else '8'}/8/1p6/P7/8/8/R3K3 w - - 0 1",
        ["axb5"], mirror,
    )
    mover = before.turn
    assert file_opened_for_rook(before, after, mover) == (None if enemy_a_pawn else 0)
    assert file_semi_opened_for_rook(before, after, mover) == (0 if enemy_a_pawn else None)
    claim = build_intent_claim(before, after, "White" if mover else "Black")
    assert claim is not None
    assert claim.rule_id == ("intent_semi_opens_file" if enemy_a_pawn else "intent_opens_file")


@pytest.mark.parametrize("mirror", [False, True])
def test_remaining_doubled_pawn_keeps_file_closed(mirror):
    before, after = transition(
        "4k3/8/8/1p6/P7/P7/8/R3K3 w - - 0 1", ["axb5"], mirror,
    )
    assert file_opened_for_rook(before, after, before.turn) is None
    assert file_semi_opened_for_rook(before, after, before.turn) is None


def test_already_open_file_is_not_newly_opened():
    before, after = transition("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", ["Kf2"])
    assert file_opened_for_rook(before, after, chess.WHITE) is None


@pytest.mark.parametrize("mirror", [False, True])
def test_supported_pawn_can_prepare_a_legal_contact_advance(mirror):
    before, after = transition(
        "4k3/8/4p3/8/8/2PP4/8/4K3 w - - 0 1", ["d4"], mirror,
    )
    assert pawn_break_prepared(before, after, before.turn) == "d"
    claim = build_intent_claim(before, after, "White" if before.turn else "Black")
    assert claim.rule_id == "intent_pawn_break"
    assert ("d5" if not mirror else "d4") in claim.text


@pytest.mark.parametrize("mirror", [False, True])
def test_directly_blocked_pawn_cannot_prepare_an_advance(mirror):
    before, after = transition(
        "4k3/8/8/3p4/8/2PP4/8/4K3 w - - 0 1", ["d4"], mirror,
    )
    assert pawn_break_prepared(before, after, before.turn) is None


def test_pinned_attacker_does_not_create_a_capture_target():
    before, after = transition(
        "k3r3/8/8/7n/4R3/8/8/4K3 b - - 0 1", ["Nf4"],
    )
    assert after.is_attacked_by(chess.WHITE, chess.F4)
    assert newly_attacked_enemy_piece(before, after, chess.WHITE) is None


def test_legal_new_capture_target_is_retained():
    before, after = transition("3k4/8/8/3n1b2/8/8/8/Q3K3 w - - 0 1", ["Qa5"])
    assert newly_attacked_enemy_piece(before, after, chess.WHITE) == (chess.D5, "knight")
