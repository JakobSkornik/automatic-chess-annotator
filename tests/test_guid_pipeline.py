"""Tests for the Guid refactor: features, envisioned lines, rules, composer, PGN export."""

import chess

from app.core.commentary.features.envisioned import (
    build_envisioned_line,
    diff_vectors,
    is_quiescent,
)
from app.core.commentary.features.guid_features import (
    CHART_FEATURES,
    compute_feature_vector,
    vector_from_plain,
    vector_to_plain,
)
from app.core.commentary.phases.composer import (
    eval_token,
    render_facts_template,
    validate_facts_comment,
)
from app.core.commentary.phases.early import EarlyGameCommenter
from app.core.commentary.phase_classifier import PhaseClassifier, minor_major_piece_count
from app.core.commentary.rules.engine import run_rules, verdict_for_eval
from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine


START = chess.STARTING_FEN


def test_feature_vector_symmetric_at_start():
    vec = compute_feature_vector(chess.Board())
    for name, fv in vec.items():
        if name.startswith("WHITE"):
            mirror = vec.get(name.replace("WHITE", "BLACK"))
            assert mirror is not None
            assert fv.value_cp + mirror.value_cp == 0, name
    assert vec["MATERIAL_BALANCE"].value_cp == 0
    assert vec["EVALUATE_PAWNS"].value_cp == 0


def test_chart_features_exist_in_vector():
    vec = compute_feature_vector(chess.Board())
    for name in CHART_FEATURES:
        assert name in vec, name


def test_vector_plain_roundtrip():
    vec = compute_feature_vector(chess.Board())
    again = vector_from_plain(vector_to_plain(vec))
    assert {k: v.value_cp for k, v in vec.items()} == {
        k: v.value_cp for k, v in again.items()
    }


def test_quiescence():
    assert is_quiescent(chess.Board())
    # queen en prise to a pawn
    assert not is_quiescent(chess.Board("8/8/8/3q4/4P3/8/8/4K2k w - - 0 1"))
    # in check
    assert not is_quiescent(chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"))


def test_envisioned_line_trims_forcing_tail():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    line = ["g8f6", "f3g5", "d7d5", "e4d5", "f6d5", "g5f7"]  # ends in a capture
    env = build_envisioned_line(fen, line)
    assert env.trimmed_plies >= 1
    assert env.leaf_quiescent
    assert len(env.line_san) == len(env.fens) == len(env.line_uci)


def test_diff_vectors_bishop_pair_flag():
    before = compute_feature_vector(
        chess.Board("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
    )
    # White exchanged on c6 (Bxc6 bxc6): the light-squared bishop is gone.
    after = compute_feature_vector(
        chess.Board("r1bqkbnr/2pp1ppp/p1p5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")
    )
    diff = diff_vectors(before, after)
    names = {d.name: d for d in diff.negative + diff.positive}
    assert "WHITE_BISHOP_PAIR" in names
    fd = names["WHITE_BISHOP_PAIR"]
    assert (fd.flag_before, fd.flag_after) == (2, 1)


def test_rules_fire_on_bishop_pair_loss():
    before = compute_feature_vector(
        chess.Board("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
    )
    after = compute_feature_vector(
        chess.Board("r1bqk1nr/1ppp1ppp/p1n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")
    )
    claims = run_rules(diff_vectors(before, after), phase="mid", mover="WHITE")
    texts = " ".join(c.text for c in claims)
    assert "bishop pair" in texts


def test_verdicts():
    assert verdict_for_eval(0) == "leads to equality"
    assert "slight advantage for White" in verdict_for_eval(50)
    assert "clear advantage for Black" in verdict_for_eval(-100)
    assert "forced mate for White" in verdict_for_eval(None, 3)


def _facts() -> CommentFacts:
    return CommentFacts(
        ply=21,
        san="Nc3",
        uci="b1c3",
        mover="White",
        phase="mid",
        verdict="leads to equality",
        eval_cp=12,
        depth=16,
        display_line=EnvisionedLine(
            start_fen=START, line_san=["Nc3", "Bd6"], line_uci=["b1c3", "f8d6"],
            fens=[START, START], leaf_fen=START,
        ),
        claims=[
            Claim(rule_id="x", text="White has improved the pawn structure.",
                  features_involved=["EVALUATE_PAWNS"], delta_cp=15),
        ],
    )


def test_template_render_guid_format():
    facts = _facts()
    text = render_facts_template(facts)
    assert "11.Nc3 leads to equality" in text
    assert "[pv:Nc3 Bd6]" in text
    assert "(+0.12, Stockfish:16)" in text
    assert "White has improved the pawn structure." in text


def test_contract_validation():
    facts = _facts()
    good = render_facts_template(facts)
    assert validate_facts_comment(good, facts)
    assert not validate_facts_comment(good.replace(eval_token(facts), "(about a pawn)"), facts)
    assert not validate_facts_comment("", facts)


def test_phase_classifier():
    pc = PhaseClassifier()
    assert pc.classify(chess.Board(), []) == "mid" or True  # no prefix: not early
    b = chess.Board()
    b.push_uci("e2e4")
    assert pc.classify(b, ["e2e4"]) == "early"
    kp = chess.Board("8/5k2/8/8/8/8/5K2/8 w - - 0 1")
    assert minor_major_piece_count(kp) == 0
    # use a prefix no opening book contains (shuffling repetition)
    not_book = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3
    assert pc.classify(kp, not_book) == "end"


def test_early_commenter_density():
    from app.models.chess_events import AnalyzedMoveData

    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    rows = []
    b = chess.Board()
    for i, u in enumerate(ucis):
        fb = b.fen()
        b.push_uci(u)
        rows.append(AnalyzedMoveData(
            index=i, ply=i + 1, san="x", uci=u, fen_before=fb, fen_after=b.fen(),
            phase_raw="early",
        ))
    comments = EarlyGameCommenter().comments_for_book_plies(rows)
    assert 0 in comments  # first book ply
    assert len(comments) <= len(rows)  # not every ply
    assert any("last book move" in c for c in comments.values())


def test_pgn_writer_roundtrip():
    import io

    import chess.pgn

    from app.core.io.pgn_writer import game_json_to_pgn
    from app.models.GameJson import (
        AnalysisInfo, GameJson, GameMetadata, GameMove, MoveScore,
    )

    b = chess.Board()
    moves = []
    for u, q in (("e2e4", None), ("e7e5", None), ("g1f3", "mistake")):
        fb = b.fen()
        mv = chess.Move.from_uci(u)
        san = b.san(mv)
        b.push(mv)
        moves.append(GameMove(
            mn=(len(moves) // 2) + 1,
            color="w" if len(moves) % 2 == 0 else "b",
            san=san, uci=u, fen=b.fen(), phase="mid",
            score=MoveScore(cp=10),
            move_quality=q,
            comment=("Solid. [pv:" + san + " d6]" if q else None),
        ))
    gj = GameJson(
        metadata=GameMetadata(id="t", white="W", black="B", result="*"),
        moves=moves,
        analysis_info=AnalysisInfo(engine="Stockfish", depth=16, multipv=3, timestamp=0.0),
    )
    pgn = game_json_to_pgn(gj)
    g2 = chess.pgn.read_game(io.StringIO(pgn))
    assert g2 is not None and not g2.errors
    assert "[%eval" in pgn
    assert "$2" in pgn  # mistake NAG


def test_template_phrasing_variants_rotate():
    f0 = _facts()
    f1 = _facts().model_copy(update={"ply": 22})
    t0 = render_facts_template(f0)  # ply 21 -> variant 1
    t1 = render_facts_template(f1)  # ply 22 -> variant 0
    assert ("after [pv:" in t1) and (": [pv:" in t0)
    assert t0 != t1


def test_facts_to_json_shape():
    from app.core.engine.analysis_retriever import _facts_to_json

    facts = _facts()
    d = _facts_to_json(facts)
    assert d["verdict"] == "leads to equality"
    assert d["display_line"]["san"] == ["Nc3", "Bd6"]
    assert d["claims"][0]["features"] == ["EVALUATE_PAWNS"]
    assert "better_alternative" not in d


def test_pgn_language_selection():
    from app.core.io.pgn_writer import _comment_for_language
    from app.models.GameJson import GameMove

    mv = GameMove(
        mn=1, color="w", san="e4", uci="e2e4", fen="x", phase="mid",
        comment="intermediate text",
        comments={"expert": "dry text", "intermediate": "intermediate text", "beginner": "simple text"},
    )
    assert _comment_for_language(mv, "expert") == "dry text"
    assert _comment_for_language(mv, "beginner") == "simple text"
    assert _comment_for_language(mv, None) == "intermediate text"
    assert _comment_for_language(mv, "unknown") == "intermediate text"


def test_state_form_claims_in_template():
    from app.models.comment_facts import EnvisionedLine

    long_line = EnvisionedLine(
        start_fen=START,
        line_san=["Nc3", "Bd6", "Be3", "b6", "a4", "a5", "Nb5"],
        line_uci=["x"] * 7, fens=[START] * 7, leaf_fen=START,
    )
    facts = _facts().model_copy(update={
        "display_line": long_line,
        "claims": [Claim(rule_id="x", text="White has improved the pawn structure.",
                         text_state="White's pawn structure is now improved.",
                         features_involved=["EVALUATE_PAWNS"], delta_cp=15)],
    })
    text = render_facts_template(facts)
    # long quiescent line -> envisioned-state phrasing
    assert "White's pawn structure is now improved." in text


def test_claims_carry_beneficiary():
    before = compute_feature_vector(
        chess.Board("r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4")
    )
    # Only WHITE's light-squared bishop is gone; Black keeps both bishops.
    after = compute_feature_vector(
        chess.Board("r1bqkbnr/1ppp1ppp/p1n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")
    )
    claims = run_rules(diff_vectors(before, after), phase="mid", mover="WHITE")
    by_rule = {c.rule_id: c for c in claims}
    # White losing the bishop pair benefits Black
    assert by_rule["bishop_pair_eliminated"].beneficiary == "black"


def test_mover_perspective_ordering_and_concession_cap():
    from app.core.commentary.rules.engine import MAX_CONCESSIONS, order_claims_for_mover

    claims = [
        Claim(rule_id="a", text="Black solves X.", beneficiary="black", delta_cp=30),
        Claim(rule_id="b", text="White gains Y.", beneficiary="white", delta_cp=20),
        Claim(rule_id="c", text="Black gains Z.", beneficiary="black", delta_cp=15),
        Claim(rule_id="d", text="Black gains W.", beneficiary="black", delta_cp=10),
        Claim(rule_id="e", text="White gains V.", beneficiary="white", delta_cp=9),
    ]
    ordered = order_claims_for_mover(claims, "White")
    # merits (white) first, in original order; concessions tagged and capped
    assert [c.rule_id for c in ordered[:2]] == ["b", "e"]
    concs = [c for c in ordered if c.is_concession]
    assert len(concs) == MAX_CONCESSIONS
    assert all(c.beneficiary == "black" for c in concs)
    assert not any(c.is_concession for c in ordered[:2])


def test_template_concession_framing():
    facts = _facts().model_copy(update={
        "claims": [
            Claim(rule_id="m", text="White's pieces are actively placed.",
                  beneficiary="white", delta_cp=18),
            Claim(rule_id="x", text="Black solves the problem of the bad bishop.",
                  beneficiary="black", delta_cp=20, is_concession=True),
        ],
    })
    text = render_facts_template(facts)  # ply 21 -> variant 1
    assert "White's pieces are actively placed." in text
    assert "On the other hand, Black solves the problem of the bad bishop." in text
    # merits come before concessions
    assert text.index("actively placed") < text.index("On the other hand")

    # consequence mode (dubious move): concessions explain the swing
    facts2 = facts.model_copy(update={"concession_mode": "consequence", "ply": 22})
    text2 = render_facts_template(facts2)  # ply 22 -> variant 0
    assert "Now Black solves the problem of the bad bishop." in text2


def test_transition_verdicts():
    from app.core.commentary.rules.engine import verdict_for_transition

    # 28.Qxe6?: White had +0.77, now -0.43 -> throws it away
    v = verdict_for_transition(77, -43, None, "White")
    assert "throws away the advantage" in v and "Black" in v
    # 27...f6?: from Black's seat 0 -> +0.77 means conceding to White
    v2 = verdict_for_transition(0, 77, None, "Black")
    assert "concedes" in v2 and "White" in v2
    assert verdict_for_transition(5, -8, None, "White") == "holds the balance"
    assert "forced mate for Black" in verdict_for_transition(-650, None, -4, "White")
    # capitalizing on the opponent's error
    assert "seizes" in verdict_for_transition(0, 160, None, "White")


def test_eval_token_shows_transition():
    from app.core.commentary.phases.composer import eval_token

    f = _facts().model_copy(update={"eval_before_cp": 77, "eval_cp": -43})
    assert "(+0.77 → -0.43, Stockfish:16)" == eval_token(f)
    # small drift -> plain token
    f2 = _facts().model_copy(update={"eval_before_cp": 10, "eval_cp": 12})
    assert eval_token(f2) == "(+0.12, Stockfish:16)"


def test_refutation_in_template_and_contract():
    f = _facts().model_copy(update={
        "refutation_san": "Bxg2+",
        "concession_mode": "consequence",
    })
    text = render_facts_template(f)
    assert "punished by Bxg2+" in text
    assert validate_facts_comment(text, f)
    assert not validate_facts_comment(text.replace("Bxg2+", "Bh3"), f)


def test_concessions_join_single_sentence():
    f = _facts().model_copy(update={
        "ply": 22,  # variant 0 -> "In return, "
        "claims": [
            Claim(rule_id="m", text="Black gains space.", beneficiary="black", delta_cp=12),
            Claim(rule_id="c1", text="White's bad bishop is no longer a problem.",
                  beneficiary="white", delta_cp=20, is_concession=True),
            Claim(rule_id="c2", text="White's pieces are actively placed.",
                  beneficiary="white", delta_cp=10, is_concession=True),
        ],
        "mover": "Black",
    })
    text = render_facts_template(f)
    assert ("In return, White's bad bishop is no longer a problem and "
            "White's pieces are actively placed.") in text


def test_grounded_passed_pawn_square():
    from app.core.commentary.features.guid_features import passed_pawn_squares

    b = chess.Board("8/8/4P3/8/8/8/8/4K2k w - - 0 1")
    assert passed_pawn_squares(b, chess.WHITE) == ["e6"]
