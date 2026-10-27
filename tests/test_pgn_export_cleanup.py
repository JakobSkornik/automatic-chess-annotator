import chess
import chess.pgn
import io

from app.core.io.pgn_writer import _try_add_line, game_json_to_pgn
from app.models.GameJson import AnalysisInfo, GameJson, GameMetadata, GameMove, MoveScore


def _game(moves):
    return GameJson(
        metadata=GameMetadata(id="test", white="White", black="Black", result="1-0"),
        moves=moves,
        analysis_info=AnalysisInfo(engine="test", depth=16, multipv=1, timestamp=0),
    )


def _move(board, san, comment=None):
    move = board.parse_san(san)
    board.push(move)
    return GameMove(
        mn=board.fullmove_number,
        color="w" if board.turn == chess.BLACK else "b",
        san=san,
        uci=move.uci(),
        fen=board.fen(),
        comment=comment,
    )


def test_identical_terminal_continuation_is_emitted_once():
    board = chess.Board()
    moves = [
        _move(board, "f3"),
        _move(board, "e5"),
        _move(board, "g4", "[pv:g4 Qh4#] [pv:g4 Qh4#]"),
    ]

    pgn = game_json_to_pgn(_game(moves))

    assert pgn.count("Continuation: Qh4#.") == 1


def test_merged_variation_node_keeps_first_eval_only():
    game = chess.pgn.Game()
    board = chess.Board()
    _try_add_line(board, game, ["e4", "e5"], 80, None, 16)
    _try_add_line(board, game, ["e4", "e5"], -120, None, 18)

    leaf = game.variation(game.variations[0].move).variation(
        game.variations[0].variations[0].move
    )
    assert leaf.comment.count("[%eval") == 1
    assert "[%eval 0.80,16]" in leaf.comment
    assert "-1.20" not in leaf.comment
    assert leaf.nags == {16}


def test_whole_annotation_keeps_variations_without_inline_pv_tokens():
    board = chess.Board()
    moves = [_move(board, "e4", "White takes space; d4 was another option."), _move(board, "c5")]
    moves[0].final_comment = True
    moves[0].debug = {"rendering_contract": "grounded_narrative_v3"}
    moves[0].comment_facts = {"display_line": {"san": ["e4", "e5"]},
        "better_alternative": {"san": "d4", "display_line": {"san": ["d4", "d5"]}}}
    game = chess.pgn.read_game(io.StringIO(game_json_to_pgn(_game(moves))))
    assert [m.uci() for m in game.mainline_moves()] == ["e2e4", "c7c5"]
    assert {v.san() for v in game.variations} == {"e4", "d4"}
    assert {v.san() for v in game.variations[0].variations} == {"c5", "e5"}
    assert "White takes space" in game.variations[0].comment
