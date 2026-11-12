"""Changes after the meeting with the supervisor (11 Nov 2026): scaled class limits, the
template without lines, the two-block layout with numbered moves, the writer without claims
and the three versions in the generation path. All are v19 switches."""
import asyncio
import gzip
import json
import math
import sys
from pathlib import Path

import pytest

from app.core.commentary import pipeline_flags
from app.core.commentary.event_extractor import _move_quality_for, scaled_class_limits
from app.core.commentary.phases.composer import (
    comment_archetype, compose_facts_comment, narrative, plain, render_facts_template,
)
from app.models.chess_events import MoveQuality
from app.research.survey_grid import grid_cells, variant_id, versions

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BUNDLE = ROOT / "data" / "surveys" / "comparison-luna-grok-composer-v18-20261105.json.gz"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ACA_PROFILE", raising=False)
    monkeypatch.delenv("COMMENTARY_RENDER_MODE", raising=False)
    for name in pipeline_flags.FLAGS:
        monkeypatch.delenv(f"ACA_V19_{name.upper()}", raising=False)


@pytest.fixture(scope="module")
def stored():
    """Facts of two published positions: 24...Rhd8 (an inaccuracy with the better 24...Rad8)
    and 30...d5 (a pawn move, with the better 30...Qb6+)."""
    from build_dev_version_study import stored_facts

    survey = json.loads(gzip.decompress(BUNDLE.read_bytes()))["survey"]
    out = {}
    for entry in survey["manifest"]["games"]:
        if entry["id"] not in ("game-14", "game-05"):
            continue
        position = entry["comparison_position"]
        moves = survey["variants"][variant_id(entry["id"], "gpt-6-luna", entry["level"])]["game"]["moves"]
        out[entry["id"]] = stored_facts(moves[position["move_index"]], moves[position["move_index"] - 1], position)
    return out


# --- scaled class limits ---

def test_class_limits_scale_with_the_evaluation_before_the_move():
    assert [round(limit) for _, limit in scaled_class_limits(0)] == [14, 35, 70, 147]
    # The same drop in expected score costs more centipawns in a one-sided position.
    assert [round(limit) for _, limit in scaled_class_limits(300)] == [26, 60, 110, 193]
    # At -3.00 the mover's expected score is below 20 points: no loss is a blunder by the bands.
    assert scaled_class_limits(-300)[-1][1] == math.inf


@pytest.mark.parametrize("loss, best, played, fixed, scaled", [
    (60, 30, -30, MoveQuality.INACCURACY, MoveQuality.INACCURACY),
    (40, 20, -20, MoveQuality.GOOD, MoveQuality.INACCURACY),       # 35 cp is the limit near equality
    (120, 0, -120, MoveQuality.MISTAKE, MoveQuality.MISTAKE),
    (160, 0, -160, MoveQuality.MISTAKE, MoveQuality.BLUNDER),      # 147 cp is the limit at equality
    (60, -30, 30, MoveQuality.INACCURACY, MoveQuality.INACCURACY),  # Black to move, White's scores
    (105, 150, 45, MoveQuality.MISTAKE, MoveQuality.MISTAKE),
])
def test_quality_class_with_fixed_and_scaled_limits(monkeypatch, loss, best, played, fixed, scaled):
    assert _move_quality_for(loss, best, played, False) == fixed
    monkeypatch.setenv("ACA_V19_SCALED_CLASS_LIMITS", "1")
    assert _move_quality_for(loss, best, played, False) == scaled
    assert _move_quality_for(loss, best, played, True) == MoveQuality.BEST


def test_decided_positions_stay_good_with_scaled_limits(monkeypatch):
    monkeypatch.setenv("ACA_V19_SCALED_CLASS_LIMITS", "1")
    assert _move_quality_for(290, 500, 210, False) == MoveQuality.GOOD


# --- template ---

def test_template_is_unchanged_without_the_switches(stored):
    text = render_facts_template(stored["game-14"], archetype=comment_archetype(None))
    assert "[pv:Rhd8 " in text and "Stockfish:" in text and "\n" not in text
    assert "Better was Rad8" in text and "pawns better in the engine evaluation" in text


def test_template_without_lines_in_two_blocks(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    text = render_facts_template(stored["game-14"], archetype=comment_archetype(None))
    played, alternative = text.split("\n")
    assert played.startswith("24...Rhd8 lets Black's advantage shrink. ")
    assert alternative.startswith("Better was 24...Rad8, which leads to a clear advantage for Black.")
    assert "[pv:" not in text and "Stockfish" not in text and "pawns better" not in text
    # A statement that starts with its own "After ..." takes no further lead-in.
    assert "After Black's move, Black's pawn on h7" in played and "Now after" not in played
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    played = render_facts_template(stored["game-14"], archetype=comment_archetype(None)).split("\n")[0]
    # A statement behind a lead-in starts in lower case; a side keeps its capital.
    assert "Now after Black's move, Black's pawn on h7" in played
    assert "After 27...Rxh2 in the analysed line, the h-file is open" in played
    # "in the analysed line" is said once per block.
    assert "After 28...Kf6, Black's king is closer to the center." in played


def test_template_joins_statements_that_rest_on_the_same_move(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    text = render_facts_template(stored["game-05"], archetype=comment_archetype(None))
    assert text.count("After 31...Qxc4") == 1
    assert "White's king is under more pressure, Black has a bishop and a pawn for a knight and Black attacks" in text
    # The reply and the first statement about it are one sentence.
    assert "The move is punished by 31.d4 in the analysed line: White's pawn on c3 is backward." in text
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    assert "The move is punished by 31.d4." in render_facts_template(stored["game-05"], archetype=comment_archetype(None))


# --- the full writer: two blocks, numbered moves ---

def _sentences(*rows):
    return json.dumps({"sentences": [{"text": t, "part": p, "main": m, "alternative": a} for t, p, m, a in rows]})


def test_prompt_and_schema_ask_for_the_two_blocks(monkeypatch, stored):
    assert "LAYOUT" not in narrative.system_prompt()
    monkeypatch.setenv("ACA_PROFILE", "v19")
    system = narrative.system_prompt()
    assert "LAYOUT" in system and "one coherent paragraph" not in system and "repeat the move number" not in system
    assert narrative.schema() is narrative.BLOCK_SCHEMA
    user, _ = narrative.prompt(stored["game-14"], "expert")
    assessment = json.loads(user)["evidence"]["assessment"]
    assert assessment["move_label"] == "24...Rhd8" and assessment["alternative"]["move_label"] == "24...Rad8"


def test_writer_text_becomes_two_blocks_with_numbered_moves(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    _, evidence = narrative.prompt(stored["game-14"], "expert")
    raw = _sentences(
        ("24...Rhd8 is an inaccuracy: it leaves Black's pawn on h7 to White's bishop on f5.", "move", [0], []),
        ("After Rxh2 Black's rook is active again.", "move", [1], []),
        ("Better was Rad8, which keeps more of the advantage.", "alternative", [], []))
    proposal, _review = narrative.parse_proposal(raw, evidence)
    played, alternative = narrative.render(proposal, evidence).split("\n")
    assert played.startswith("24...Rhd8 is an inaccuracy") and "After 27...Rxh2 Black's rook" in played
    assert alternative == "Better was 24...Rad8, which keeps more of the advantage."


@pytest.mark.parametrize("rows, message", [
    ([("The rook move is an inaccuracy that leaves Black's pawn on h7 loose.", "move", [0], [])], "does not name its move"),
    ([("24...Rhd8 is an inaccuracy.", "move", [], []), ("The other rook was the right one.", "alternative", [], [])],
     "does not name its move"),
    ([("Better was 24...Rad8.", "alternative", [], []), ("24...Rhd8 is an inaccuracy.", "move", [], [])], "must come before"),
    ([("24...Rhd8 is an inaccuracy; after 26...Rxh2 Black's rook is active.", "move", [1], [])], "does not match the line"),
    ([("Better was 24...Rad8.", "alternative", [], [])], "move part is missing"),
])
def test_writer_layout_is_checked(monkeypatch, stored, rows, message):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    _, evidence = narrative.prompt(stored["game-14"], "expert")
    with pytest.raises(ValueError, match=message):
        narrative.parse_proposal(_sentences(*rows), evidence)


def test_a_pawn_move_at_the_start_of_a_block_gets_its_number(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    _, evidence = narrative.prompt(stored["game-05"], "expert")
    assert narrative.number_moves("d5 lets the advantage shrink.\nQb6+ was better.", evidence) == (
        "30...d5 lets the advantage shrink.\n30...Qb6+ was better.")


# --- the writer without claims ---

def test_writer_without_claims_sees_no_statement_of_the_rules(stored):
    user, evidence = plain.prompt(stored["game-14"], "expert", moves_before=["e4", "e5", "Nf3"])
    view = json.loads(user)["evidence"]
    assert evidence["main"], "the position has claims; they must not reach this writer"
    # What an analysis program shows, and nothing else.
    assert set(view) == {"assessment", "game_before_the_move", "position_before_the_move", "lines", "placeholders"}
    assert not {"kind", "decision_point", "feature_comparison", "horizon_note"} & set(view["assessment"])
    for claim in evidence["main"]:
        assert claim["change"] not in user
    assert view["game_before_the_move"] == "1.e4 e5 2.Nf3"
    assert view["lines"]["played_move"].startswith("24...Rhd8 25.Rc4 ") and view["lines"]["alternative"].startswith("24...Rad8 ")
    assert view["position_before_the_move"]["fen"].startswith("r6r/1pp2k1p/")
    assert view["assessment"]["move_label"] == "24...Rhd8" and view["assessment"]["classification"] == "inaccuracy"
    for word in ("citable", "trace", "anchor", "rejects a move"):
        assert word not in plain.SYSTEM
    assert "LAYOUT" in plain.SYSTEM and "rejects a move" in narrative.LAYOUT_NOTE + narrative._NUMBER_CHECK_NOTE


def test_writer_without_claims_text_is_checked_and_numbered(stored):
    _, evidence = plain.prompt(stored["game-14"], "expert")
    good = json.dumps({"sentences": [
        {"text": "24...Rhd8 is an inaccuracy because Black's pawn on h7 loses its defender.", "part": "move"},
        {"text": "Rad8 was better, since the rook on h8 keeps guarding it.", "part": "alternative"}]})
    assert plain.render(plain.parse_proposal(good, evidence), evidence) == (
        "24...Rhd8 is an inaccuracy because Black's pawn on h7 loses its defender.\n"
        "24...Rad8 was better, since the rook on h8 keeps guarding it.")
    # Form and length only: a move that is not in the lines is not this writer's rejection.
    plain.parse_proposal(json.dumps({"sentences": [{"text": "24...Rhd8 allows Qxa1 with a win.", "part": "move"}]}), evidence)
    for text, message in (("24...Rhd8 loses 0.66 of the advantage.", "placeholders"),
                          ("24...Rhd8 is fine. " + "word " * 260, "Word budget"),
                          ("The rook move is slightly inaccurate.", "does not name its move")):
        with pytest.raises(ValueError, match=message):
            plain.parse_proposal(json.dumps({"sentences": [{"text": text, "part": "move"}]}), evidence)


class _Service:
    """A provider that answers every request with one prepared reply."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        self._provider = type("Provider", (), {"is_configured": staticmethod(lambda: True)})()

    async def _llm_call_json_schema(self, system, user, **kwargs):
        self.requests.append((system, user, kwargs))
        return self.reply


def test_lines_only_mode_uses_the_writer_without_claims(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "lines_only")
    service = _Service(json.dumps({"sentences": [
        {"text": "24...Rhd8 is an inaccuracy because Black's pawn on h7 loses its defender.", "part": "move"}]}))
    result = asyncio.run(compose_facts_comment(service, stored["game-14"], model="m", effort="low", level="expert"))
    assert result["rendering"] == "llm" and result["commentary_status"] == "llm_unreviewed"
    assert result["writer"] == "lines_only" and result["text"].startswith("24...Rhd8 is an inaccuracy")
    system, _user, kwargs = service.requests[0]
    assert system == plain.SYSTEM and kwargs["schema"] is plain.SCHEMA
    # A rejected draft leaves the template, marked as a fallback.
    service.reply = json.dumps({"sentences": [{"text": "A natural developing move.", "part": "move"}]})
    result = asyncio.run(compose_facts_comment(service, stored["game-14"], model="m", effort="low", level="expert"))
    assert result["rendering"] == "template" and result["commentary_status"] == "template_validation_failed"


# --- generation path ---

def test_versions_take_the_place_of_models_in_the_grid():
    three = versions({"id": "gpt-6-luna", "provider": "openai", "label": "GPT-6 Luna"})
    assert [(v["id"], v["writer"], v["llm"]) for v in three] == [
        ("P", "full", "gpt-6-luna"), ("T", "template", "gpt-6-luna"), ("B", "lines_only", "gpt-6-luna")]
    games = [{"id": "game-01", "level": "expert"}]
    cells = grid_cells({"games": games, "models": three, "levels": ["beginner", "intermediate", "expert"]})
    assert [c["id"] for c in cells] == ["game-01__P__expert", "game-01__T__expert", "game-01__B__expert"]
    assert all(c["model"] in "PTB" and c["llm"] == "gpt-6-luna" for c in cells)
    # A model comparison keeps the cells it always had.
    old = grid_cells({"games": games, "models": [{"id": "grok-4.6", "provider": "grok"}], "levels": ["expert"]})
    assert old == [{"id": "game-01__grok-4.6__expert", "game_id": "game-01", "model": "grok-4.6",
                    "provider": "grok", "level": "expert"}]


def test_error_label_and_alternative_follow_the_class(monkeypatch):
    """With scaled limits the key moment's error label is the class of the move, and an
    error always has a better alternative, however small the gap in centipawns."""
    from types import SimpleNamespace
    from app.core.commentary import key_moment_detector as detector
    from app.core.commentary.phases.composer import framing

    monkeypatch.setenv("ACA_PROFILE", "v19")
    alt = SimpleNamespace(eval_cp=20, eval_mate=None, is_inferior=False)
    facts = SimpleNamespace(better_alternative=alt, eval_cp=-20, eval_mate=None, mover="White",
                            move_quality="inaccuracy")
    assert framing.alternative_comparison(facts) == "better"  # 40 cp, below the fixed 50
    facts.move_quality = "good"
    assert framing.alternative_comparison(facts) == "comparable"
    monkeypatch.setenv("ACA_V19_SCALED_CLASS_LIMITS", "0")
    facts.move_quality = "inaccuracy"
    assert framing.alternative_comparison(facts) == "comparable"
    monkeypatch.delenv("ACA_V19_SCALED_CLASS_LIMITS")

    one = detector.KeyMomentDetector()
    previous = SimpleNamespace(score=30, hiddenFeatures={}, depth=20)
    current = SimpleNamespace(score=-110, hiddenFeatures={}, depth=21)
    monkeypatch.setattr(detector, "_mover_is_white", lambda move: True)
    monkeypatch.setattr(one, "_log_score", lambda *a, **k: None)
    monkeypatch.setattr(one, "_critical_decision", lambda *a, **k: [])
    for name in ("_good_defense", "_great_move", "_missed_opportunity"):
        monkeypatch.setattr(detector, name, lambda *a, **k: [])
    # A swing of 140 cp is a "mistake" by the fixed limits; the class of the move decides.
    assert one._score_based(current, previous, None, move_class="blunder")[0] == "blunder"
    assert one._score_based(current, previous, None, move_class="good") == []
    monkeypatch.setenv("ACA_V19_SCALED_CLASS_LIMITS", "0")
    assert one._score_based(current, previous, None, move_class="good") == ["mistake"]


def test_repeated_capture_gets_the_number_of_its_own_move(monkeypatch):
    """'axb5' by White and then by Black: the second one is Black's move, not the first again."""
    monkeypatch.setenv("ACA_PROFILE", "v19")
    evidence = {"traces": {"main": [{"anchor": "18...Nf6"}, {"anchor": "19.e5"}, {"anchor": "19...Bxf3"}],
                           "alternative": [{"anchor": "18...Bc6"}, {"anchor": "19.axb5"}, {"anchor": "19...axb5"},
                                           {"anchor": "20.Ra1"}]}}
    text = "18...Nf6 holds.\n18...Bc6 was comparable; after 19.axb5 axb5 and Ra1 White uses the a-file."
    assert narrative.number_moves(text, evidence).split("\n")[1] == (
        "18...Bc6 was comparable; after 19.axb5 19...axb5 and 20.Ra1 White uses the a-file.")
    assert narrative.number_moves("Nf6 holds.\nBc6: axb5 axb5.", evidence) == (
        "18...Nf6 holds.\n18...Bc6: 19.axb5 19...axb5.")


def test_line_moves_are_counted_in_their_own_block(monkeypatch):
    """A move of the played line that the alternative block names (the alternative's line has
    it too) does not count against the played line."""
    evidence = {"traces": {"main": [{"anchor": "10...Nc6"}, {"anchor": "11.g3"}, {"anchor": "11...hxg3+"},
                                    {"anchor": "12.Nxg3"}, {"anchor": "12...Qb6"}],
                           "alternative": [{"anchor": "10...Qb6"}, {"anchor": "11.g4"}, {"anchor": "11...hxg3+"}]}}
    text = "10...Nc6 develops; 11...hxg3+ and 12.Nxg3 follow.\n10...Qb6 was comparable."
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert narrative.line_moves_named(text, evidence) == {"main": 2, "alternative": 0}
    monkeypatch.setenv("ACA_V19_COMMENT_BLOCKS", "0")
    assert narrative.line_moves_named(text, evidence) == {"main": 3, "alternative": 1}


# --- second round of 11 Nov: parts without claims, wording of statements ---

def test_block_move_repeated_in_its_line_is_not_a_line_move(monkeypatch):
    """14.Nh4+ ... 18.Nh4: naming the block's own move does not use up the two line moves."""
    evidence = {"traces": {"main": [{"anchor": "14.Nh4+"}, {"anchor": "14...Ke8"}, {"anchor": "15.Qxe6+"},
                                    {"anchor": "15...Bxe6"}, {"anchor": "18.Nh4"}]}}
    text = "14.Nh4+ is a blunder: after 14...Ke8 and 15.Qxe6+ the exchanges favour Black."
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert narrative.line_moves_named(text, evidence) == {"main": 2}
    monkeypatch.setenv("ACA_V19_COMMENT_BLOCKS", "0")
    assert narrative.line_moves_named(text, evidence) == {"main": 3}


def test_writer_is_told_which_part_has_no_claims(monkeypatch, stored):
    facts = stored["game-14"].model_copy(deep=True)  # the alternative 24...Rad8 has no claim
    facts.claims = []
    assert "PARTS WITHOUT CLAIMS" not in narrative.system_prompt()
    assert "parts_without_claims" not in json.loads(narrative.prompt(facts, "expert")[0])["evidence"]
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert "PARTS WITHOUT CLAIMS" in narrative.system_prompt()
    assert json.loads(narrative.prompt(facts, "expert")[0])["evidence"]["parts_without_claims"] == ["move", "alternative"]
    assert json.loads(narrative.prompt(stored["game-14"], "expert")[0])["evidence"]["parts_without_claims"] == ["alternative"]
    monkeypatch.setenv("ACA_V19_CLAIMLESS_PARTS", "0")
    assert "PARTS WITHOUT CLAIMS" not in narrative.system_prompt()


def test_move_without_any_claim_is_not_sent_to_the_model(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    facts = stored["game-14"].model_copy(deep=True)
    facts.claims, facts.better_alternative.claims = [], []
    service = _Service("{}")
    result = asyncio.run(compose_facts_comment(service, facts, model="m", effort="low", level="expert"))
    assert result["rendering"] == "template" and not service.requests
    assert result["commentary_status"] == "template" and result["commentary_reason"] == "no_claims"
    monkeypatch.setenv("ACA_V19_CLAIMLESS_PARTS", "0")
    asyncio.run(compose_facts_comment(service, facts, model="m", effort="low", level="expert"))
    assert len(service.requests) == 1


def test_verdict_of_a_sound_move_in_a_worse_position(monkeypatch):
    from app.core.commentary.rules.verdicts import verdict_for_transition

    # Black to move, White better by 2.00 before and 2.60 after.
    assert verdict_for_transition(200, 260, None, "Black", move_class="good").startswith("makes matters worse")
    assert verdict_for_transition(200, 210, None, "Black") == "gives White a winning advantage"
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert verdict_for_transition(200, 260, None, "Black", move_class="good") == "leaves White with a winning advantage"
    assert verdict_for_transition(200, 260, None, "Black", move_class="mistake").startswith("makes matters worse")
    assert verdict_for_transition(200, 210, None, "Black") == "leaves White with a winning advantage"
    assert verdict_for_transition(-200, -150, None, "Black", move_class="good") == "maintains Black's advantage"
    assert verdict_for_transition(-200, -150, None, "Black", move_class="inaccuracy") == "lets Black's advantage shrink"


def test_no_pawn_break_that_the_pawn_could_have_played_at_once(monkeypatch):
    import chess
    from app.core.commentary.features.intent import _prepared_break_move

    def after(fen, san):
        start = chess.Board(fen)
        leaf = start.copy()
        leaf.push_san(san)
        return start, leaf

    one_step = after("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "d6")   # ...d5 was possible
    later = after("rnbqkbnr/ppp2ppp/3pp3/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "d5")     # prepares ...d4
    assert _prepared_break_move(*one_step, chess.BLACK) == chess.Move.from_uci("d6d5")
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert _prepared_break_move(*one_step, chess.BLACK) is None
    assert _prepared_break_move(*later, chess.BLACK) == chess.Move.from_uci("d5d4")


def test_exchange_of_equal_pieces_is_named_as_an_exchange(monkeypatch):
    import chess
    from app.core.commentary.rules.tactical_evidence import tactical_claims
    from app.models.comment_facts import EnvisionedLine

    board = chess.Board("3rk3/8/8/3q4/8/8/3Q4/4K3 w - - 0 1")
    start, ucis, fens, sans = board.fen(), [], [], ["Qxd5", "Rxd5", "Ke2"]
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    line = EnvisionedLine(start_fen=start, line_uci=ucis, line_san=sans, fens=fens, leaf_fen=board.fen())

    def exchange():
        return next(c.text for c in tactical_claims(line) if c.rule_id == "exchange_sequence")

    assert exchange() == "In the displayed exchange on d5, White loses a queen, while Black loses a queen."
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert exchange() == "Queens are exchanged on d5."


def _claim(rule, text, side, **more):
    from app.models.comment_facts import Claim

    return Claim(rule_id=rule, text=text, beneficiary=side, is_concession=side == "white", **more)


def test_template_does_not_say_the_same_thing_twice(monkeypatch, stored):
    """An unsafe pawn and its capture are one statement; at most two statements are joined."""
    monkeypatch.setenv("ACA_PROFILE", "v19")
    facts = stored["game-14"].model_copy(deep=True)
    facts.claims = [
        _claim("piece_left_en_prise", "After Black's move, Black's pawn on h6 is no longer safe: White's rook on h5 "
               "can win it.", "white", entity="en_prise:h6", support_ply=1),
        _claim("legal_capture", "White's rook from h5 captures the pawn on h6.", "white", entity="capture:2:h6",
               support_ply=2),
        _claim("material_won", "White is a pawn up.", "white", support_ply=3)]
    played = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[0]
    # The pawn count holds from a later move of the line (no feature series here: its own ply).
    assert "After Black's move, Black's pawn on h6 is no longer safe: White's rook on h5 can win it. After " in played
    assert played.endswith("White is a pawn up.") and "captures the pawn on h6" not in played
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    played = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[0]
    assert "can win it and White's rook from h5 captures the pawn on h6 and White is a pawn up." in played


def test_template_exchange_hole_and_missed_continuation(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    facts = stored["game-14"].model_copy(deep=True)
    hole = "The square {} becomes a hole in White's position: no White pawn can attack it any more."
    facts.claims = [
        _claim("legal_capture", "White's bishop from c4 captures the knight on d5.", "white", realization="envisioned",
               entity="capture:2:d5", support_ply=2),
        _claim("material_won", "White is up a bishop.", "white", realization="envisioned", persistence="transient",
               support_ply=2),
        _claim("legal_capture", "Black's pawn from e6 captures the bishop on d5.", "black", realization="envisioned",
               entity="capture:3:d5", support_ply=3),
        _claim("hole_created", hole.format("f3"), "black", realization="envisioned", support_ply=3),
        _claim("hole_created", hole.format("d4"), "black", realization="envisioned", support_ply=4)]
    text = render_facts_template(facts, archetype=comment_archetype(None))
    assert "is up a bishop" not in text and "missed a better continuation" not in text
    assert text.count("no White pawn can attack it any more") == 1 and "the square d4 becomes a hole in White's position." in text
    # The statement with its explanation closes the sentence.
    assert "captures the bishop on d5 and the square f3 becomes a hole in White's position: no White pawn" in text
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    text = render_facts_template(facts, archetype=comment_archetype(None))
    assert "is up a bishop" in text and "missed a better continuation" in text
    assert text.count("no White pawn can attack it any more") == 2


def test_weaker_alternative_that_still_wins_says_still(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    facts = stored["game-14"].model_copy(deep=True)
    facts.move_quality, facts.eval_cp = "best", -450
    alt = facts.better_alternative
    alt.is_inferior, alt.eval_cp, alt.verdict, alt.claims = True, -300, "gives Black a winning advantage", []
    alternative = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[1]
    assert alternative == "Weaker was 24...Rad8, which still gives Black a winning advantage."
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    alternative = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[1]
    assert alternative == "Weaker was 24...Rad8, which gives Black a winning advantage."


def test_parried_threat_is_one_plain_sentence(monkeypatch):
    import chess
    from types import SimpleNamespace
    from app.core.commentary.rules.prophylaxis import rule_threat_parried

    start = chess.Board("2k5/8/8/7R/8/6K1/1P6/1r6 w - - 0 1")   # Black's rook attacks the pawn on b2
    leaf = start.copy()
    leaf.push_san("Rh2")
    ctx = SimpleNamespace(start_board=start, leaf_board=leaf, mover="WHITE",
                          color=lambda side: chess.WHITE if side == "WHITE" else chess.BLACK)
    old = rule_threat_parried(ctx)[0].text
    assert old.startswith("White defends the pawn on b2: so that") and "(the threat was" in old
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert rule_threat_parried(ctx)[0].text == (
        "White defends the pawn on b2, so Black's rook on b1 can no longer win it.")


def test_statement_names_the_move_it_first_holds_after(monkeypatch, stored):
    """43...c6 44.Rxh6 Kb5: the facts mark the whole forcing stretch as immediate. The pawn is
    lost after the second move and the king steps up after the third; what the played move
    itself changed stays without a lead-in."""
    monkeypatch.setenv("ACA_PROFILE", "v19")
    facts = stored["game-14"].model_copy(deep=True)
    line = facts.display_line
    second, third = (f"{'25.' if i == 2 else '25...'}{line.line_san[i - 1]}" for i in (2, 3))
    plies = len(line.line_san)
    line.feature_series = {"MATERIAL_BALANCE": [0, 0] + [100] * (plies - 1),
                           "BLACK_KING_ACTIVITY": [0, 0, 0] + [-5] * (plies - 2),
                           "BLACK_ROOK_ACTIVITY": [0] + [-9] * plies}
    facts.claims = [
        _claim("rooks_activated", "Black's rooks become more active.", "black", support_ply=3,
               features_involved=["BLACK_ROOK_ACTIVITY"]),
        _claim("king_activated", "Black's king moves closer to the center.", "black", support_ply=3,
               features_involved=["BLACK_KING_ACTIVITY"]),
        _claim("material_won", "White is a pawn up.", "white", support_ply=3, features_involved=["MATERIAL_BALANCE"])]
    played = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[0]
    assert played == (f"24...Rhd8 lets Black's advantage shrink. Black's rooks become more active. After {second} in the "
                      f"analysed line, White is a pawn up. After {third}, Black's king moves closer to the center.")
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    played = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[0]
    assert "Black's rooks become more active. Black's king moves closer to the center. Now White is a pawn up." in played


def test_bishop_rule_needs_a_change_of_the_pawns_on_its_colour(monkeypatch):
    """5.Nc3 in front of the c2 pawn raises the bishop-pawns score without moving a pawn."""
    import chess
    from types import SimpleNamespace
    from app.core.commentary.rules.feature_rules import rule_bishop_color_complex

    start = chess.Board("rnb1k1nr/ppp2ppp/3p1q2/2b1pP2/4P3/3P4/PPP3PP/RNBQKBNR w KQkq - 0 5")
    leaf = start.copy()
    leaf.push_san("Nc3")
    ctx = SimpleNamespace(start_board=start, leaf_board=leaf,
                          color=lambda side: chess.WHITE if side == "WHITE" else chess.BLACK,
                          flag_pair=lambda name: (24, 32) if name.startswith("WHITE") else None)
    assert [c.rule_id for c in rule_bishop_color_complex(ctx)] == ["bishop_hemmed_in"]
    monkeypatch.setenv("ACA_PROFILE", "v19")
    assert rule_bishop_color_complex(ctx) == []
    leaf.push_san("Nc6")
    leaf.push_san("g3")   # a pawn steps onto the colour of the bishop on c1
    assert [c.rule_id for c in rule_bishop_color_complex(ctx)] == ["bishop_hemmed_in"]


def test_template_leaves_out_the_statement_that_the_lines_part(monkeypatch, stored):
    monkeypatch.setenv("ACA_PROFILE", "v19")
    facts = stored["game-14"].model_copy(deep=True)
    facts.better_alternative.claims = [
        _claim("lines_diverge", "The lines part at White's next move: after Rad8 it continues Rc4.", "black",
               realization="envisioned", support_ply=2, supplementary=True)]
    alternative = render_facts_template(facts, archetype=comment_archetype(None)).split("\n")[1]
    assert alternative == "Better was 24...Rad8, which leads to a clear advantage for Black."
    monkeypatch.setenv("ACA_V19_STATEMENT_WORDING", "0")
    assert "lines part" in render_facts_template(facts, archetype=comment_archetype(None))
