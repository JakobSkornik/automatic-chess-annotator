"""Writer evidence added in grounded_narrative_v12: claim timing after a forced sequence,
matched-depth comparison, decision-point framing and the per-claim square whitelist."""
import json

import chess
import pytest

from app.core.commentary.phases.composer import narrative, sentence_plan
from app.core.commentary.phases.composer.prompts import build_facts_user_prompt
from app.core.commentary.features.guid_features.evidence import evidence_for
from app.models.comment_facts import BestAlternative, Claim, CommentFacts, EnvisionedLine, FeatureAdvantage

# game-03 of the survey before 13.Bg5: the engine line is forced until 14...Qxg5+ (the test
# records the immediate horizon at ply 4 and lets the line run two quiet plies further).
BG5 = "r2qkb1r/pp3p1p/4p1p1/2PpP3/1n1n1BQ1/2NB3P/PPP2PP1/2KR3R w kq - 2 13"


def line(fen, sans, **fields):
    board = chess.Board(fen)
    ucis, fens = [], []
    for san in sans:
        move = board.parse_san(san)
        ucis.append(move.uci())
        board.push(move)
        fens.append(board.fen())
    return EnvisionedLine(start_fen=fen, line_uci=ucis, line_san=list(sans), fens=fens, leaf_fen=board.fen(), **fields)


def claim(rule_id, text, ply, branch, realization, squares=(), roles=None):
    node = chess.Board(branch.fens[ply - 1])
    return Claim(rule_id=rule_id, text=text, text_state=text, beneficiary="white", support_ply=ply,
                 support_fen=branch.fens[ply - 1], baseline_fen=branch.start_fen, realization=realization,
                 evidence=evidence_for(node, [chess.parse_square(s) for s in squares], roles=roles))


def bg5_facts(**update):
    main = line(BG5, ["Bg5", "h5", "Qxd4", "Qxg5+", "f4", "Nc6"], immediate_horizon_ply=4,
                horizon_reason="settled")
    alt = line(BG5, ["Bxg6", "fxg6", "Qxg6+", "Kd7"], immediate_horizon_ply=4, horizon_reason="persistence")
    claims = [
        claim("discovered_attack", "Moving the bishop uncovers White's queen on g4 against the knight on d4.",
              1, main, "immediate", ["g4", "d4", "g5"], {chess.D4: "target"}),
        claim("material_won", "White is a pawn up.", 4, main, "immediate", ["d4", "g5"]),
        claim("king_safer", "White's king is now safer.", 6, main, "envisioned", ["b1", "c1"], {chess.C1: "king"}),
    ]
    better = BestAlternative(san="Bxg6", uci="d3g6", eval_cp=40, display_line=alt, matched_ply=4,
                             feature_advantages=[FeatureAdvantage(name="BLACK_KING_DANGER", alt_delta_cp=80,
                                                                  played_delta_cp=0, advantage_cp=80)],
                             feature_advantages_matched=[FeatureAdvantage(name="W_MOBILITY", alt_delta_cp=0,
                                                                          played_delta_cp=30, advantage_cp=-30)],
                             horizon_note="The lines settle at different depths.",
                             claims=[claim("exchange_sequence", "In the exchange on g6 White loses a bishop.",
                                           2, alt, "immediate", ["g6"])])
    facts = CommentFacts(ply=24, san="Bg5", uci="f4g5", mover="White", phase="mid", verdict="keeps an edge",
                         eval_cp=60, eval_before_cp=50, move_quality="good", display_line=main, claims=claims,
                         better_alternative=better, selection_metadata={"moment_kind": "Q"})
    return facts.model_copy(update=update)


def writer_view(facts, level="expert", key_moment_type=None):
    user, evidence = narrative.prompt(facts, level, key_moment_type=key_moment_type)
    return json.loads(user)["evidence"], evidence


def test_claims_are_tagged_by_when_they_hold():
    view, _ = writer_view(bg5_facts())
    timings = [(c["rule_id"], c["timing"], c["holds_from"], c["caused_by"]) for c in view["main"]]
    assert timings == [("discovered_attack", "after_move", "13.Bg5", None),
                       ("material_won", "after_forced_sequence", "14...Qxg5+", None),
                       ("king_safer", "in_line", "15...Nc6", "15...Nc6")]
    assert view["lines"]["main"] == {"forced_moves": ["13...h5", "14.Qxd4", "14...Qxg5+"],
                                     "forced_until": "14...Qxg5+", "settles_at": "15...Nc6",
                                     "horizon_reason": "settled", "length_plies": 6}
    assert view["lines"]["alternative"]["forced_until"] == "14...Kd7"
    assert view["main"][0]["evidence_pieces"] == ["target: Black's knight on d4", "subject: White's queen on g4",
                                                  "subject: White's bishop on g5"]


def test_default_plan_states_forced_sequence_claims_rather_than_changes():
    plan = sentence_plan.default_plan(sentence_plan.catalog(bg5_facts()))
    assert [g["form"] for g in plan["main"]] == ["change", "state", "state"]


def test_prompt_explains_the_three_timings_and_decision_points():
    for word in ("after_move", "after_forced_sequence", "in_line", "forced_moves", "holds_from",
                 "DECISION POINTS", "compared_at", "evidence_squares", "parried"):
        assert word in narrative.SYSTEM, word
    assert "after_forced_sequence" in narrative.REVIEW_SYSTEM and "horizon_note" in narrative.REVIEW_SYSTEM
    assert "<immediate>" not in build_facts_user_prompt(bg5_facts())
    assert "<after the forced sequence>" in build_facts_user_prompt(bg5_facts())


def test_comparison_is_taken_at_the_matched_depth_when_lines_settle_differently():
    view, _ = writer_view(bg5_facts())
    assert view["assessment"]["feature_comparison"] == {
        "favours_played_move": ["piece activity"], "favours_alternative": [],
        "compared_at": "both lines 4 plies after the move (matched depth)"}
    assert view["assessment"]["horizon_note"] == "The lines settle at different depths."
    # Without a matched comparison the settled endpoints are compared, as before.
    plain = bg5_facts()
    plain = plain.model_copy(update={"better_alternative": plain.better_alternative.model_copy(
        update={"matched_ply": None, "feature_advantages_matched": [], "horizon_note": None})})
    view, _ = writer_view(plain)
    assert view["assessment"]["feature_comparison"] == {
        "favours_played_move": [], "favours_alternative": ["king safety"],
        "compared_at": "the settled end of each line"}


@pytest.mark.parametrize("kind, quality, key_moment, expected", [
    ("Q", "good", None, True), ("Q", "best", "decision_point", True), ("G", "good", "decision_point", True),
    ("Q", "mistake", None, False), ("R", "good", None, False), ("T", "blunder", None, False)])
def test_decision_point_flag_needs_a_quiet_choice_without_an_error(kind, quality, key_moment, expected):
    facts = bg5_facts(move_quality=quality, selection_metadata={"moment_kind": kind})
    view, _ = writer_view(facts, key_moment_type=key_moment)
    assert view["assessment"]["decision_point"] is expected


def test_text_may_name_only_squares_of_cited_claims_and_trace_moves():
    _, evidence = writer_view(bg5_facts())

    def check(text, main):
        return narrative.parse_proposal(json.dumps({"text": text, "main": main, "alternative": []}), evidence)

    check("The bishop move uncovers the queen on g4 against the knight on d4.", [0])  # cited claim squares
    check("After the exchanges on d4 and g5 White is a pawn up.", [1])  # trace squares and evidence squares
    check("White's king on c1 is safer later in the line.", [2])  # evidence of the cited claim
    with pytest.raises(ValueError, match="Square absent"):
        check("White's king on c1 is safer.", [0])  # c1 belongs to an uncited claim, not a trace move
    with pytest.raises(ValueError, match="Square absent"):
        check("Black's rook on a8 is passive.", [0])  # occupied square from a board snapshot only


def test_castling_in_a_trace_allows_the_rook_squares():
    start = "r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 8"
    main = line(start, ["O-O-O", "O-O"])
    facts = CommentFacts(ply=15, san="O-O-O", uci="e1c1", mover="White", phase="mid", verdict="keeps the balance",
                         eval_cp=10, display_line=main,
                         claims=[claim("king_safer", "White's king is safer.", 1, main, "immediate", ["c1"])])
    _, evidence = narrative.prompt(facts, "expert")
    narrative.parse_proposal(json.dumps({"text": "The rook comes to d1 and Black castles with the rook on f8.",
                                         "main": [0], "alternative": []}), evidence)


def test_reviewer_receives_the_forced_sequence_summary():
    _, evidence = writer_view(bg5_facts())
    _, reviews = narrative.parse_proposal(json.dumps(
        {"text": "The bishop move uncovers the queen on g4.", "main": [0], "alternative": []}), evidence)
    assert reviews[0]["lines"]["main"]["forced_until"] == "14...Qxg5+"
    assert reviews[0]["claims"]["main"][0]["timing"] == "after_move"


def test_internal_vocabulary_and_line_retelling_are_rejected():
    _, evidence = writer_view(bg5_facts())

    def check(text):
        return narrative.parse_proposal(json.dumps({"text": text, "main": [0], "alternative": []}), evidence)

    check("The bishop move uncovers the queen on g4; after Qxd4 and Qxg5+ White is a pawn up.")  # two line moves
    with pytest.raises(ValueError, match="engine-line moves"):
        check("After h5, Qxd4, Qxg5+, f4 and Nc6 the queen on g4 is gone.")  # h5 is a square-like push, 4 others
    with pytest.raises(ValueError, match="Internal vocabulary"):
        check("At a decision point, the queen on g4 hits d4.")
    with pytest.raises(ValueError, match="Internal vocabulary"):
        check("At the matched depth the queen on g4 hits d4.")
    assert narrative.line_moves_named("after Qxd4 and ...Qxg5+", evidence) == {"main": 2, "alternative": 0}
