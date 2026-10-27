from types import SimpleNamespace as NS

import chess
import pytest

from app.core.engine.analysis.key_moments import _comment_selection
from app.core.commentary.pipeline.game_pipeline import _key_moment_order


def event(fen="7k/8/8/3q4/4P3/8/8/K7 w - - 0 1", uci="e4d5", **updates):
    board = chess.Board(fen)
    board.push_uci(uci)
    line = NS(leaf_fen=chess.STARTING_FEN, evidence_valid=True)
    facts = NS(display_line=line, terminal_result=None, better_alternative=None, eval_mate=None, selection_metadata={})
    values = dict(fen_before=fen, fen_after=board.fen(), uci=uci,
                  move_quality=NS(value="best"), best_move_uci=uci,
                  best_move_eval_cp=200, eval_after_cp=200,
                  key_moment_type="best_move", comment_facts=facts, ply=1)
    values.update(updates)
    return NS(**values)


def test_capture_evidence_cannot_promote_noisy_sequence():
    move = event()
    result = _comment_selection(move)
    assert not result["allowed"]
    assert result["reason"] == "nonquiescent_start"
    assert result["forcing_played_move"]
    assert _key_moment_order([move], "both") == []


def test_capture_at_quiet_root_is_sequence_anchor():
    move = event(fen="7k/8/8/3p4/4P3/8/8/K7 w - - 0 1")
    assert _comment_selection(move)["allowed"]


@pytest.mark.parametrize("quality,key,loss", [("mistake", "mistake", 150),
                                             ("blunder", "blunder", 300),
                                             ("inaccuracy", "missed_opportunity", 200)])
def test_verified_errors_allow_noisy_root(quality, key, loss):
    move = event(move_quality=NS(value=quality), key_moment_type=key,
                 best_move_uci="a1b1", eval_after_cp=200-loss)
    assert _comment_selection(move)["allowed"]
    assert _key_moment_order([move], "both") == [0]


def test_badge_without_alternative_loss_is_not_exception():
    move = event(move_quality=NS(value="blunder"), key_moment_type="blunder",
                 best_move_uci="a1b1", best_move_eval_cp=None)
    assert not _comment_selection(move)["allowed"]


def test_noisy_leaf_blocks_quiet_root():
    move = event(fen=chess.STARTING_FEN, uci="e2e4")
    move.comment_facts.display_line.leaf_fen = "7k/8/8/3q4/4P3/8/8/K7 w - - 0 1"
    assert _comment_selection(move)["reason"] == "nonquiescent_or_missing_endpoint"


def test_verified_lost_mate_is_exception():
    move = event(move_quality=NS(value="mistake"), key_moment_type="mistake",
                 best_move_uci="a1b1", best_move_eval_cp=None)
    move.comment_facts.better_alternative = NS(eval_mate=3, is_inferior=False, uci="a1b1")
    assert _comment_selection(move)["reason"] == "verified_quality_error"


def test_unknown_alternative_cannot_prove_escape_from_mate():
    move = event(move_quality=NS(value="blunder"), key_moment_type="blunder",
                 best_move_uci="a1b1", best_move_eval_cp=None, eval_after_cp=None)
    move.comment_facts.eval_mate = -3
    move.comment_facts.better_alternative = NS(eval_mate=None, eval_cp=None, is_inferior=False)
    assert not _comment_selection(move)["allowed"]


def test_encoded_mate_uses_mate_ordering_without_fake_cp_loss():
    move = event(move_quality=NS(value="blunder"), key_moment_type="blunder",
                 best_move_uci="a1b1", best_move_eval_cp=100, eval_after_cp=-999997)
    result = _comment_selection(move)
    assert result["allowed"]
    assert result["verified_mate_loss"]
    assert result["verified_loss_cp"] is None


def test_assembly_keeps_inspectable_facts_when_prose_is_blocked(monkeypatch):
    import chess.pgn
    from app.core.engine import analysis_retriever as retriever
    from app.models.chess_events import MoveEvent, MoveQuality, MoveEventType
    from app.models.comment_facts import CommentFacts, EnvisionedLine
    from app.models.Move import Move

    source = event()
    facts = CommentFacts(ply=1, san="exd5", uci=source.uci, mover="White",
                         phase="mid", verdict="equal", display_line=EnvisionedLine(
                             start_fen=source.fen_before, leaf_fen=chess.STARTING_FEN))
    move_event = MoveEvent(move_index=0, ply=1, san="exd5", uci=source.uci,
                           fen_before=source.fen_before, fen_after=source.fen_after,
                           phase="mid", move_quality=MoveQuality.BEST,
                           event_type=MoveEventType.BEST_MOVE_PLAYED,
                           key_moment_type="best_move", comment_facts=facts)
    move = Move(id=1, position=source.fen_after, move=source.uci, context="mainline",
                depth=1, isAnalyzed=True, piece="P", phase="mid",
                hiddenFeatures={"_llm": {"comment": "Stale prose must be suppressed."}})
    game = chess.pgn.Game()
    game.setup(chess.Board(source.fen_before))
    game.add_main_variation(chess.Move.from_uci(source.uci))
    row = NS(analyzed_move=move, pvs=[], fen_before=source.fen_before, phase_raw="mid")
    state = NS(game=game, moves_list=[move], move_events=[move_event],
               metadata=NS(comment_side="both"), llm_done=False)
    monkeypatch.setattr(retriever, "_build_move_debug", lambda analyzed, event, retained, comment, phase: {
        "retained_facts": retained is facts, "selection": retained.selection_metadata["comment_selection"]})
    result = retriever._build_game_move(state, 0, row)
    assert result.comment is None
    assert not result.is_key_moment
    assert not result.final_comment
    assert result.comment_facts is not None
    assert result.debug["retained_facts"]
    assert result.debug["selection"]["reason"] == "nonquiescent_start"
