"""Regression coverage for evidence preservation in game-wide fact attachment."""

import chess

from app.core.engine.analysis_retriever import _attach_comment_facts, _dedup_claims
from app.models.chess_events import AnalyzedMoveData, MoveEvent, MoveEventType, MoveQuality
from app.models.comment_facts import Claim, CommentFacts


def facts(claims, ply=1):
    return CommentFacts(ply=ply, san="e4", uci="e2e4", mover="White", phase="mid",
                        verdict="equal", claims=claims)


def test_concrete_events_never_share_a_rule_side_budget():
    last = {}
    for rule_id in ("legal_capture", "exchange_sequence", "absolute_pin", "piece_pinned",
                    "material_won", "tactical_material_loss", "hanging_piece", "intent_targets_piece"):
        forecast = Claim(rule_id=rule_id, text="First event", beneficiary="black", realization="envisioned")
        actual = forecast.model_copy(update={"realization": "immediate"})
        different = actual.model_copy(update={"text": "Different event"})
        for ply, claim in enumerate((forecast, actual, different), 1):
            result = _dedup_claims(facts([claim]), ply, last, 6, 24)
            assert result.claims == [claim]
            assert result.muted_claims == []
    assert last == {}


def test_structural_identity_includes_board_and_realization():
    claim = Claim(rule_id="passed_pawn_created", text="White has a passer.", beneficiary="white",
                  support_fen=chess.STARTING_FEN, realization="envisioned")
    last = {}
    assert _dedup_claims(facts([claim]), 1, last, 6, 24).claims
    assert not _dedup_claims(facts([claim]), 2, last, 6, 24).claims
    board = chess.Board()
    board.push_san("e4")
    for changed in (claim.model_copy(update={"realization": "immediate"}),
                    claim.model_copy(update={"support_fen": board.fen()})):
        assert _dedup_claims(facts([changed]), 3, last, 6, 24).claims == [changed]


def test_unselected_moves_do_not_consume_game_comment_budget(monkeypatch):
    import app.core.commentary.rules as rules

    claim = Claim(rule_id="material_standing", text="White is a pawn up.")
    rows, events = [], []
    board = chess.Board()
    for i, san in enumerate(("e4", "e5", "Nf3")):
        before = board.fen()
        move = board.parse_san(san)
        board.push(move)
        row = AnalyzedMoveData(index=i, ply=i + 1, san=san, uci=move.uci(),
                               fen_before=before, fen_after=board.fen())
        rows.append(row)
        events.append(MoveEvent(move_index=i, ply=i + 1, san=san, uci=move.uci(),
                                fen_before=before, fen_after=board.fen(), phase="mid",
                                move_quality=MoveQuality.MISTAKE, event_type=MoveEventType.QUIET,
                                key_moment_type="mistake" if i else None))
    monkeypatch.setattr(rules, "build_comment_facts", lambda row, event, **kw: facts([claim], row.ply))
    _attach_comment_facts(events, rows)
    assert events[0].comment_facts.claims == [claim]
    assert events[1].comment_facts.claims == [claim]
    assert events[2].comment_facts.claims == []


def test_game_attachment_retains_successive_actual_captures():
    """Run the real fact builder for every move, so earlier forecasts also dedup."""
    rows, events = [], []
    board = chess.Board()
    for i, san in enumerate("e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 Nxe5".split()):
        before = board.fen()
        move = board.parse_san(san)
        board.push(move)
        rows.append(AnalyzedMoveData(index=i, ply=i + 1, san=san, uci=move.uci(),
                                    fen_before=before, fen_after=board.fen(), score_cp=0))
        events.append(MoveEvent(move_index=i, ply=i + 1, san=san, uci=move.uci(),
                                fen_before=before, fen_after=board.fen(), phase="mid",
                                move_quality=MoveQuality.MISTAKE, event_type=MoveEventType.QUIET,
                                key_moment_type="mistake", eval_before_cp=0, eval_after_cp=0))
    # Supply a legal continuation to a settled leaf, as the engine pass does.
    continuation = ["d8d4", "e5f3", "d4e4", "d1e2", "e4e2", "e1e2"]
    for i, row in enumerate(rows):
        row.hidden_features = {"_engine": {"after_pv_uci": [r.uci for r in rows[i + 1:]] + continuation}}
    _attach_comment_facts(events, rows)
    for event in events[-3:]:
        assert event.comment_facts is not None
        assert any(c.rule_id in {"legal_capture", "exchange_sequence"} and (c.realization == "immediate" or c.rule_id == "exchange_sequence")
                   for c in event.comment_facts.claims), event.san
