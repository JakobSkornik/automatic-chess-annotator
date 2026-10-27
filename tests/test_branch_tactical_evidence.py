"""Regressions from the Yevchenko–Martinez saved export."""
import chess
import pytest

from app.core.commentary.rules.facts_builder import _material_only, build_comment_facts
from app.core.commentary.rules.tactical_evidence import tactical_claims
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import Claim, EnvisionedLine


def line(fen, sans):
    board = chess.Board(fen)
    ucis, fens = [], []
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return EnvisionedLine(start_fen=fen, line_uci=ucis, line_san=sans,
                          fens=fens, leaf_fen=board.fen())


def test_export_bishop_pin_names_all_three_pieces():
    branch = line('q4b1k/1b4pp/p3Qp2/8/2BN4/1Pr4P/6P1/3R2K1 b - - 1 30', ['Bc5', 'Qg4', 'a5'])
    claims = tactical_claims(branch)
    pin = next(c for c in claims if c.rule_id == 'absolute_pin')
    assert all(word in pin.text for word in ('bishop', 'c5', 'knight', 'd4', 'king', 'g1'))
    assert pin.support_ply == 1 and pin.realization == 'immediate'
    assert pin in _material_only(claims)


def test_export_rook_capture_chain_never_claims_a_forced_gain():
    branch = line('q1r2b1k/1br3pp/p3Qp2/8/3N4/1PN4P/6P1/2RR1BK1 b - - 0 28', ['Rxc3', 'Rxc3', 'Rxc3'])
    claims = tactical_claims(branch)
    exchange = next(c for c in claims if c.rule_id == 'exchange_sequence')
    assert 'c3' in exchange.text
    assert 'White loses a knight and a rook' in exchange.text
    assert 'Black loses a rook' in exchange.text
    assert exchange.support_ply == 3
    assert exchange.baseline_fen == branch.start_fen
    assert all(c.rule_id != 'legal_capture' for c in claims)
    assert 'win' not in exchange.text.lower() and 'forced' not in exchange.text.lower()
    assert 'continuation' not in exchange.text.lower()


@pytest.mark.parametrize('policy', ['current', 'contrastive'])
def test_alternative_preserves_entire_capture_chain_and_reply_context(policy):
    from app.core.commentary.rules.facts_builder import _alternative_merits
    branch = line('q1r2b1k/1br3pp/p3Qp2/8/3N4/1PN4P/6P1/2RR1BK1 b - - 0 28', ['Rxc3', 'Rxc3', 'Rxc3'])
    # Separate captures still retain their full reply/follow-through context
    # when they cannot be summarized as one same-square exchange.
    raw = [Claim(rule_id='legal_capture', text=f'Capture {ply}', support_ply=ply,
                 beneficiary=color) for ply, color in enumerate(['black', 'white', 'black'], 1)]
    claims = _alternative_merits(raw, [], 'Black', policy)
    assert [c.support_ply for c in claims] == [1, 2, 3]
    assert [c.is_concession for c in claims] == [False, True, False]
    summarized = _alternative_merits(tactical_claims(branch), [], 'Black', policy)
    assert summarized[0].rule_id == 'exchange_sequence'
    assert summarized[0].support_ply == 3


def test_exchange_summary_does_not_hide_a_known_fourth_same_square_capture():
    branch = line('3r2k1/3r4/8/3p4/2B5/8/3R4/3R2K1 w - - 0 1',
                  ['Bxd5', 'Rxd5', 'Rxd5', 'Rxd5'])
    claims = tactical_claims(branch)
    assert not any(c.rule_id == 'exchange_sequence' for c in claims)
    assert len([c for c in claims if c.rule_id == 'legal_capture']) == 3


@pytest.mark.parametrize('invalid', ['illegal', 'fen', 'flag'])
def test_corrupt_branch_never_supplies_tactical_claims(invalid):
    branch = line('q4b1k/1b4pp/p3Qp2/8/2BN4/1Pr4P/6P1/3R2K1 b - - 1 30', ['Bc5', 'Qg4'])
    if invalid == 'illegal':
        branch.line_uci[1] = 'a1a8'
    elif invalid == 'fen':
        branch.fens[1] = branch.start_fen
    else:
        branch.evidence_valid = False
    assert tactical_claims(branch) == []


def test_export_pawn_recapture_does_not_assert_temporary_surplus_or_passer(monkeypatch):
    branch = line('rn1qkbnr/pb1p1ppp/1p2p3/2p5/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq - 0 5', ['cxd4', 'Nxd4', 'Nf6'])
    event = MoveEvent(move_index=9, ply=10, san='cxd4', uci=branch.line_uci[0],
                      fen_before=branch.start_fen, fen_after=branch.fens[0], phase='mid',
                      move_quality='good', event_type='quiet', eval_after_cp=0)
    row = AnalyzedMoveData(index=9, ply=10, san=event.san, uci=event.uci,
                           fen_before=event.fen_before, fen_after=event.fen_after,
                           phase_raw='mid', hidden_features={'_engine': {'after_pv_uci': branch.line_uci[1:]}})
    # Force an over-eager immediate horizon to reproduce the historical bug;
    # later legal nodes must veto its temporary material and pawn anchors.
    from app.core.commentary.rules import facts_builder
    monkeypatch.setattr(facts_builder, 'is_quiescent', lambda board: True)
    monkeypatch.setattr(facts_builder, '_first_quiet_fen', lambda branch: branch.fens[0])
    facts = build_comment_facts(row, event)
    assert any(c.rule_id in {'legal_capture', 'exchange_sequence'} for c in facts.claims)
    assert not any(c.rule_id == 'material_won' for c in facts.claims)
    assert not any('passed pawn on d4' in c.text for c in facts.claims)
    assert any(s['reason'] == 'material_state_changes_in_displayed_continuation'
               for s in facts.selection_metadata['evidence_suppressions'])
