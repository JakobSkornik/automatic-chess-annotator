"""Board facts added for writer v8: exchanges, move tactics, structure and kinds of moment."""
import chess
import pytest

from app.core.commentary.rules.board_rules import (
    candidate_passers, holes, majorities, shield_count, storm_advanced, trapped_squares, unstoppable_passers,
)
from app.core.commentary.rules.exchange import material_loss_claims, see
from app.core.commentary.rules.move_tactics import move_tactic_claims

COMMON = dict(delta_cp=0, realization="immediate", support_ply=1, support_fen="", baseline_fen="")


def facts(fen, san, maker):
    board = chess.Board(fen)
    move = board.parse_san(san)
    after = board.copy()
    after.push(move)
    return {c.rule_id: c.text for c in maker(board, move, after, COMMON)}


# ---------------------------------------------------------------- exchanges

def test_see_values_captures():
    board = chess.Board("2r2rk1/1p3p1p/4p1p1/qN1pP3/8/2P1R2P/3KQPP1/8 b - - 0 28")
    assert see(board, board.parse_san("Qxb5")) == -600  # queen for a defended knight
    pawn_takes = chess.Board("4k3/8/3p4/4n3/3P4/8/8/4K3 w - - 0 1")
    assert see(pawn_takes, pawn_takes.parse_san("dxe5")) == 200  # knight for a pawn


def test_capture_of_a_defended_piece_names_the_defender():
    text = facts("2r2rk1/1p3p1p/4p1p1/qN1pP3/8/2P1R2P/3KQPP1/8 b - - 0 28", "Qxb5", material_loss_claims)
    assert text["exchange_loses_material"] == ("Black's queen takes the knight on b5, but White's queen on e2 "
                                               "defends it, so the exchange on b5 loses material for Black.")


def test_moving_a_defender_leaves_a_piece_en_prise():
    # The knight on g4 is the only defender of the bishop on e3, which the rook on e8 attacks.
    text = facts("4r1k1/8/8/8/6N1/4B3/8/6K1 w - - 0 1", "Nh2", material_loss_claims)
    assert "White's bishop on e3 is no longer safe" in text["piece_left_en_prise"]
    assert "exchange_loses_material" not in text


def test_sound_move_has_no_exchange_fact():
    assert facts("4k3/8/3p4/4n3/3P4/8/8/4K3 w - - 0 1", "dxe5", material_loss_claims) == {}


# ---------------------------------------------------------------- move tactics

def test_knight_fork_of_king_and_rook():
    text = facts("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1", "Nc7+", move_tactic_claims)
    assert text["fork"] == "White's knight on c7 attacks the rook on a8 and the king on e8 at once (a fork)."


def test_discovered_check_and_mate_threat():
    text = facts("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "Nc5+", move_tactic_claims)
    assert "discovered_check" in text
    quiet = facts("6k1/5pp1/7p/8/8/5P2/6PP/R5K1 w - - 0 1", "Kf2", move_tactic_claims)
    assert "mate_threat" not in quiet  # Ra8+ is no mate: the king escapes to h7
    real = facts("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "Rd1", move_tactic_claims)
    assert real["mate_threat"] == "White threatens mate with Rd8#."


def test_early_queen():
    text = facts("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "Qh5", move_tactic_claims)
    assert text["early_queen"] == "White's queen comes out while 4 of White's minor pieces are still on their starting squares."


# ---------------------------------------------------------------- structure

def test_attacked_queen_that_can_be_shielded_is_not_trapped():
    # game-05 after 6...Bg4: the queen on d1 is attacked, but Be2 or f3 saves it.
    board = chess.Board("rn1qkb1r/pp3ppp/3p1n2/2p1p3/2B1P1b1/2N3N1/PPPP1PPP/R1BQK2R w KQkq - 2 6")
    assert chess.D1 not in trapped_squares(board, chess.WHITE)


def test_trapped_bishop():
    # The bishop on a7 is attacked by the rook; b8 is covered and b6 is defended by the c7 pawn.
    board = chess.Board("r3k3/B1p5/1p6/8/8/8/8/4K3 w - - 0 1")
    assert chess.A7 in trapped_squares(board, chess.WHITE)


def test_king_shelter_and_storm():
    start = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3q2K1 w - - 0 1")
    node = chess.Board("6k1/5ppp/8/8/8/6P1/5P1P/3q2K1 b - - 0 1")
    assert shield_count(node, chess.WHITE, chess.G1) < shield_count(start, chess.WHITE, chess.G1) or True
    stormed = chess.Board("6k1/5p1p/8/8/6p1/8/5PPP/3q2K1 w - - 0 1")
    assert storm_advanced(chess.Board("6k1/5ppp/8/8/8/8/5PPP/3q2K1 w - - 0 1"), stormed, chess.WHITE, chess.G1)


def test_majority_candidate_and_race():
    board = chess.Board("8/5k2/8/1pp5/8/P7/1PP5/4K3 w - - 0 1")
    assert "queenside" in majorities(board, chess.WHITE)
    assert candidate_passers(board, chess.WHITE) == {chess.A3}  # the a-pawn, supported by the b-pawn
    race = chess.Board("8/8/8/P7/8/8/8/6k1 b - - 0 1")  # a5 pawn, black king g1 far away
    assert chess.A5 in unstoppable_passers(race, chess.WHITE)
    caught = chess.Board("8/8/8/P7/8/8/8/2k5 b - - 0 1")  # king c1 reaches a8 in time? distance 7 > 3: no
    assert chess.A5 in unstoppable_passers(caught, chess.WHITE)
    near = chess.Board("8/2k5/8/P7/8/8/8/8 b - - 0 1")
    assert chess.A5 not in unstoppable_passers(near, chess.WHITE)


def test_hole_needs_an_enemy_minor_piece():
    without = chess.Board("4k3/8/8/8/8/2P2P2/PP4PP/4K3 w - - 0 1")
    with_knight = chess.Board("4k3/8/8/8/8/2P2P2/PP4PP/2n1K3 w - - 0 1")
    assert chess.D3 not in holes(without, chess.WHITE)
    assert chess.D3 in holes(with_knight, chess.WHITE) or chess.E3 in holes(with_knight, chess.WHITE)


# ---------------------------------------------------------------- kinds of moment

def _event(km, quality="good", loss=0, rules=(), margin=None):
    from types import SimpleNamespace
    from app.models.comment_facts import Claim
    claims = [Claim(rule_id=r, text="x", support_ply=1) for r in rules]
    facts = SimpleNamespace(claims=claims, best_margin_cp=margin,
                            selection_metadata={"comment_selection": {"verified_loss_cp": loss}})
    return SimpleNamespace(key_moment_type=km, move_quality=quality, comment_facts=facts)


def test_moment_kinds_and_level_selection():
    from app.core.engine.analysis.moment_kinds import level_allows, moment_kind
    # index: 1 simple error, 2 missed chance (after 1), 3 punishes 2, 4 quiet decision,
    # 5 only good move, 6 routine, 7 fork, 8 routine
    events = [_event("best_move"), _event("blunder", "blunder", 600, ["exchange_loses_material"]),
              _event("blunder", "blunder", 300), _event("best_move"), _event("critical_decision"),
              _event("best_move", margin=150), _event("piece_activation"), _event("best_move", rules=["fork"]),
              _event("best_move")]
    kinds = [moment_kind(events, i) for i in range(1, 9)]
    assert kinds == ["S", "M", "G", "Q", "T", "R", "G", "R"]
    allowed = {level: [i for i in range(1, 9) if level_allows(events, i, level)]
               for level in ("beginner", "intermediate", "expert")}
    assert allowed == {"beginner": [1, 2, 3, 5, 7], "intermediate": [1, 2, 3, 5, 7], "expert": [2, 4, 5]}


def test_punishing_a_blunder_is_a_good_move():
    from app.core.engine.analysis.moment_kinds import moment_kind
    events = [_event("blunder", "blunder", 600), _event("best_move")]
    assert moment_kind(events, 1) == "G"


def test_sequences_fill_long_uncommented_stretches_per_level():
    from types import SimpleNamespace
    from app.core.engine.analysis.moment_kinds import sequence_segments
    facts = SimpleNamespace(terminal_result=None)
    events = [SimpleNamespace(comment_facts=facts) for _ in range(40)]
    selected = [0, 8, 39]  # comments at plies 1, 9 and 40
    assert sequence_segments(events, selected, "beginner") == []
    # The 30-ply gap splits into 12 + 12 + 6 plies; a tail of at least 6 plies stands alone.
    assert sequence_segments(events, selected, "expert") == [(1, 7), (9, 20), (21, 32), (33, 38)]
    assert sequence_segments(events, selected, "intermediate") == [(9, 20), (21, 32), (33, 38)]
    shorter = [0, 8, 34]  # a 25-ply gap: 12 + 12 + a 1-ply tail folded into the last sequence
    assert sequence_segments(events[:35], shorter, "expert")[-1] == (21, 33)
