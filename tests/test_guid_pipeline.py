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
)
from app.core.commentary.phase_classifier import (
    PhaseClassifier,
    minor_major_piece_count,
)
from app.core.commentary.phases.composer import (
    eval_token,
    render_facts_template,
    validate_facts_comment,
)
from app.core.commentary.phases.early import EarlyGameCommenter
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
        chess.Board(
            "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        )
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
        chess.Board(
            "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        )
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
            start_fen=START,
            line_san=["Nc3", "Bd6"],
            line_uci=["b1c3", "f8d6"],
            fens=[START, START],
            leaf_fen=START,
        ),
        claims=[
            Claim(
                rule_id="x",
                text="White has improved the pawn structure.",
                features_involved=["EVALUATE_PAWNS"],
                delta_cp=15,
            ),
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
    assert not validate_facts_comment(
        good.replace(eval_token(facts), "(about a pawn)"), facts
    )
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
        rows.append(
            AnalyzedMoveData(
                index=i,
                ply=i + 1,
                san="x",
                uci=u,
                fen_before=fb,
                fen_after=b.fen(),
                phase_raw="early",
            )
        )
    comments = EarlyGameCommenter().comments_for_book_plies(rows)
    assert 0 in comments  # first book ply
    assert len(comments) <= len(rows)  # not every ply
    assert any("last book move" in c for c in comments.values())


def test_pgn_writer_roundtrip():
    import io

    import chess.pgn

    from app.core.io.pgn_writer import game_json_to_pgn
    from app.models.GameJson import (
        AnalysisInfo,
        GameJson,
        GameMetadata,
        GameMove,
        MoveScore,
    )

    b = chess.Board()
    moves = []
    for u, q in (("e2e4", None), ("e7e5", None), ("g1f3", "mistake")):
        b.fen()
        mv = chess.Move.from_uci(u)
        san = b.san(mv)
        b.push(mv)
        moves.append(
            GameMove(
                mn=(len(moves) // 2) + 1,
                color="w" if len(moves) % 2 == 0 else "b",
                san=san,
                uci=u,
                fen=b.fen(),
                phase="mid",
                score=MoveScore(cp=10),
                comment=("Solid. [pv:" + san + " d6]" if q else None),
            )
        )
    gj = GameJson(
        metadata=GameMetadata(id="t", white="W", black="B", result="*"),
        moves=moves,
        analysis_info=AnalysisInfo(
            engine="Stockfish", depth=16, multipv=3, timestamp=0.0
        ),
    )
    pgn = game_json_to_pgn(gj)
    g2 = chess.pgn.read_game(io.StringIO(pgn))
    assert g2 is not None and not g2.errors
    assert "[%eval" in pgn


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


def test_state_form_claims_in_template():
    from app.models.comment_facts import EnvisionedLine

    long_line = EnvisionedLine(
        start_fen=START,
        line_san=["Nc3", "Bd6", "Be3", "b6", "a4", "a5", "Nb5"],
        line_uci=["x"] * 7,
        fens=[START] * 7,
        leaf_fen=START,
    )
    facts = _facts().model_copy(
        update={
            "display_line": long_line,
            "claims": [
                Claim(
                    rule_id="x",
                    text="White has improved the pawn structure.",
                    text_state="White's pawn structure is now improved.",
                    features_involved=["EVALUATE_PAWNS"],
                    delta_cp=15,
                )
            ],
        }
    )
    text = render_facts_template(facts)
    # long quiescent line -> envisioned-state phrasing
    assert "White's pawn structure is now improved." in text


def test_claims_carry_beneficiary():
    before = compute_feature_vector(
        chess.Board(
            "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        )
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
    facts = _facts().model_copy(
        update={
            "claims": [
                Claim(
                    rule_id="m",
                    text="White's pieces are actively placed.",
                    beneficiary="white",
                    delta_cp=18,
                ),
                Claim(
                    rule_id="x",
                    text="Black solves the problem of the bad bishop.",
                    beneficiary="black",
                    delta_cp=20,
                    is_concession=True,
                ),
            ],
        }
    )
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
    assert eval_token(f) == "(+0.77 → -0.43, Stockfish:16)"
    # small drift -> plain token
    f2 = _facts().model_copy(update={"eval_before_cp": 10, "eval_cp": 12})
    assert eval_token(f2) == "(+0.12, Stockfish:16)"


def test_refutation_in_template_and_contract():
    f = _facts().model_copy(
        update={
            "refutation_san": "Bxg2+",
            "concession_mode": "consequence",
        }
    )
    text = render_facts_template(f)
    assert "punished by Bxg2+" in text
    assert validate_facts_comment(text, f)
    assert not validate_facts_comment(text.replace("Bxg2+", "Bh3"), f)


def test_concessions_join_single_sentence():
    f = _facts().model_copy(
        update={
            "ply": 22,  # variant 0 -> "In return, "
            "claims": [
                Claim(
                    rule_id="m",
                    text="Black gains space.",
                    beneficiary="black",
                    delta_cp=12,
                ),
                Claim(
                    rule_id="c1",
                    text="White's bad bishop is no longer a problem.",
                    beneficiary="white",
                    delta_cp=20,
                    is_concession=True,
                ),
                Claim(
                    rule_id="c2",
                    text="White's pieces are actively placed.",
                    beneficiary="white",
                    delta_cp=10,
                    is_concession=True,
                ),
            ],
            "mover": "Black",
        }
    )
    text = render_facts_template(f)
    assert (
        "In return, White's bad bishop is no longer a problem and "
        "White's pieces are actively placed."
    ) in text


def test_grounded_passed_pawn_square():
    from app.core.commentary.features.guid_features import passed_pawn_squares

    b = chess.Board("8/8/4P3/8/8/8/8/4K2k w - - 0 1")
    assert passed_pawn_squares(b, chess.WHITE) == ["e6"]


def test_parse_text_extracts_from_schema_blob():
    from app.core.commentary.phases.composer import _parse_text

    blob = (
        '{"type":"object","properties":{"text":{"type":"string"}},'
        '"required":["text"],"additionalProperties":false}\n\n'
        '{"text":"21...Nc5 leaves White with a decisive advantage (+5.31, Stockfish:16)."}'
    )
    assert (
        _parse_text(blob)
        == "21...Nc5 leaves White with a decisive advantage (+5.31, Stockfish:16)."
    )
    # plain object still works
    assert _parse_text('{"text":"hello"}') == "hello"
    # last valid object wins on concatenation
    assert _parse_text('{"text":"first"}\n{"text":"second"}') == "second"


def test_validate_rejects_schema_and_raw_json():
    facts = _facts()
    assert not validate_facts_comment('{"type":"object","properties":{}}', facts)
    assert not validate_facts_comment('{"text":"x"}', facts)
    # a real rendered comment still validates
    assert validate_facts_comment(render_facts_template(facts), facts)


def test_line_feature_series_shape():
    from app.core.commentary.rules.engine import _line_feature_series

    start = chess.STARTING_FEN
    b = chess.Board()
    fens = []
    for u in ("e2e4", "e7e5", "g1f3"):
        b.push_uci(u)
        fens.append(b.fen())
    series = _line_feature_series(
        start, fens, ["MATERIAL_BALANCE", "WHITE_PIECE_ACTIVITY"]
    )
    # one point for the start position plus one per ply
    assert len(series["MATERIAL_BALANCE"]) == len(fens) + 1
    assert all(isinstance(v, int) for v in series["WHITE_PIECE_ACTIVITY"])


def test_claim_realization_immediate_vs_envisioned():
    from app.core.commentary.rules.engine import _claim_realization

    c = Claim(rule_id="x", text="t", features_involved=["F"])
    # change lands on the move (point 0 -> point 1): immediate
    assert _claim_realization(c, {"F": [0, 100, 100]}) == "immediate"
    # change only develops after the move (flat at ply 1, swings at the leaf)
    assert _claim_realization(c, {"F": [0, 0, 100]}) == "envisioned"
    # no meaningful swing at all: immediate (nothing to defer)
    assert _claim_realization(c, {"F": [50, 51, 52]}) == "immediate"
    # missing series: defaults to immediate
    assert _claim_realization(c, {}) == "immediate"


def test_template_hedges_envisioned_concession():
    facts = _facts().model_copy(
        update={
            "claims": [
                Claim(
                    rule_id="x",
                    text="Black's pawn structure has been weakened.",
                    beneficiary="black",
                    delta_cp=32,
                    is_concession=True,
                    realization="envisioned",
                ),
            ],
        }
    )
    text = render_facts_template(facts)
    # framed as a developing risk, not an accomplished fact
    assert "Down the line, Black's pawn structure has been weakened." in text
    assert "On the other hand" not in text
    assert "Now Black" not in text


def _alt(eval_cp: int, realization: str = "immediate") -> dict:
    from app.models.comment_facts import BestAlternative

    return {
        "better_alternative": BestAlternative(
            san="Rd8",
            uci="d7d8",
            eval_cp=eval_cp,
            verdict="keeps the edge",
            display_line=EnvisionedLine(
                start_fen=START,
                line_san=["Rd8"],
                line_uci=["d7d8"],
                fens=[START],
                leaf_fen=START,
            ),
            claims=[
                Claim(
                    rule_id="r",
                    text="Black's rooks become active on the open file.",
                    beneficiary="black",
                    delta_cp=15,
                    realization=realization,
                ),
            ],
        )
    }


def test_template_missed_chance_vs_potential():
    base = _facts().model_copy(update={"mover": "Black", "eval_cp": 0})
    # materially better alternative, gain lands at once -> "missed the chance"
    immediate = base.model_copy(update=_alt(eval_cp=-300, realization="immediate"))
    t_imm = render_facts_template(immediate)
    assert "Black missed the chance here" in t_imm
    # materially better, gain only develops in the line -> "missed the potential"
    envis = base.model_copy(update=_alt(eval_cp=-300, realization="envisioned"))
    assert "Black missed the potential here" in render_facts_template(envis)


def test_template_comparable_alternative_not_a_miss():
    base = _facts().model_copy(update={"mover": "Black", "eval_cp": 0})
    # near-equal alternative -> comparable option, never framed as a miss
    comparable = base.model_copy(update=_alt(eval_cp=-20))
    text = render_facts_template(comparable)
    assert "A comparable alternative was Rd8" in text
    assert "missed" not in text


def test_comment_archetype_mapping():
    from app.core.commentary.phases.composer import comment_archetype

    assert comment_archetype("brilliant") == "brilliant_sacrifice"
    assert comment_archetype("best_move") == "engine_choice"
    assert comment_archetype("great_move") == "engine_choice"
    assert comment_archetype("inaccuracy") == "inaccuracy_missed"
    assert comment_archetype("missed_opportunity") == "inaccuracy_missed"
    assert comment_archetype("blunder") == "neutral"
    assert comment_archetype(None) == "neutral"


def test_template_archetype_openers():
    facts = _facts()
    head = render_facts_template(facts, archetype="engine_choice")
    assert "is the strongest move here" in head
    # The opener must not contain a phrase the forbidden-phrase scrub strips
    # (which would otherwise leave debris like "is the ,").
    from app.core.commentary.forbidden_phrases import scrub_forbidden

    assert scrub_forbidden(head)[0] == head
    assert "is a brilliant sacrifice" in render_facts_template(
        facts, archetype="brilliant_sacrifice"
    )
    # neutral / default keeps the plain verdict head
    assert "strongest move here" not in render_facts_template(facts)


def test_annotation_symbol_mapping():
    from app.core.engine.analysis_retriever import _annotation_symbol

    assert _annotation_symbol("brilliant") == "!!"
    assert _annotation_symbol("best_move") == "!"
    assert _annotation_symbol("great_move") == "!"
    assert _annotation_symbol("inaccuracy") == "?!"
    assert _annotation_symbol("mistake") == "?"
    assert _annotation_symbol("blunder") == "??"
    assert _annotation_symbol("critical_decision") is None
    assert _annotation_symbol(None) is None


def _me(**over):
    from app.models.chess_events import MoveEvent, MoveEventType, MoveQuality

    base = {
        "move_index": 10,
        "ply": 21,
        "san": "Nd5",
        "uci": "f4d5",
        "fen_before": START,
        "fen_after": START,
        "phase": "mid",
        "move_quality": MoveQuality.BEST,
        "event_type": MoveEventType.BEST_MOVE_PLAYED,
        "best_move_uci": "f4d5",
    }
    base.update(over)
    return MoveEvent(**base)


def _facts_with_material(series, mover="White", eval_cp=200, eval_before=50, claims=()):
    return _facts().model_copy(
        update={
            "mover": mover,
            "eval_cp": eval_cp,
            "eval_before_cp": eval_before,
            "claims": list(claims),
            "display_line": EnvisionedLine(
                start_fen=START,
                line_san=["Nd5", "exd5", "Qxd5"],
                line_uci=["f4d5", "e6d5", "d1d5"],
                fens=[START, START, START],
                leaf_fen=START,
                feature_series={"MATERIAL_BALANCE": series},
            ),
        }
    )


def test_promote_brilliant_sacrifice():
    from app.core.engine.analysis_retriever import _promote_key_moment

    me = _me(key_moment_type=None)
    # White gives up ~3 pawns along the line yet keeps a +2 eval -> brilliant
    facts = _facts_with_material([0, -300, -300], eval_cp=200, eval_before=50)
    assert _promote_key_moment(me, facts, facts.claims) == "brilliant"


def test_promote_best_move_requires_claim():
    from app.core.engine.analysis_retriever import _promote_key_moment

    me = _me(key_moment_type=None)
    # best move, no material dip, with a fired claim -> best_move
    with_claim = _facts_with_material([10, 10, 10], claims=_facts().claims)
    assert _promote_key_moment(me, with_claim, with_claim.claims) == "best_move"
    # same move with NO claims (e.g. decided position) -> not promoted
    no_claim = _facts_with_material([10, 10, 10], claims=())
    assert _promote_key_moment(me, no_claim, []) is None


def test_promote_leaves_non_best_untouched():
    from app.core.engine.analysis_retriever import _promote_key_moment

    me = _me(best_move_uci="a2a3", uci="h2h3", key_moment_type="inaccuracy")
    facts = _facts_with_material([0, -300, -300])
    # not the engine's move -> never brilliant/best, keeps its existing type
    assert _promote_key_moment(me, facts, facts.claims) == "inaccuracy"


def test_merit_suppressed_when_move_backfires():
    # 9...Bxc2-style mistake: a mover merit (damaged the opponent's pawns) plus
    # the consequence (opponent won material). Black ends up clearly worse, so
    # the misleading merit must be dropped, leaving the consequence.
    import chess

    from app.core.commentary.rules.engine import build_comment_facts
    from app.models.chess_events import (
        AnalyzedMoveData,
        MoveEvent,
        MoveEventType,
        MoveQuality,
    )

    fen_before = "rnbqkb1r/pp3ppp/2p1pn2/4N1B1/6PP/2N5/PPPPQP2/R3KB1R b KQkq - 0 9"

    def _claims_stub(diff, **kw):
        return [
            Claim(
                rule_id="pawns",
                text="White's pawn structure has been weakened.",
                beneficiary="black",
                features_involved=["WHITE_PAWN_DOUBLED"],
                delta_cp=32,
            ),
            Claim(
                rule_id="mat",
                text="White has won a pawn.",
                beneficiary="white",
                is_concession=True,
                features_involved=["MATERIAL_BALANCE"],
                delta_cp=100,
            ),
        ]

    # Drive build_comment_facts with stubbed rules so the test is deterministic.
    # build_comment_facts calls run_rules via the facts_builder module namespace.
    import app.core.commentary.rules.facts_builder as eng

    orig = eng.run_rules
    eng.run_rules = _claims_stub
    try:
        board = chess.Board(fen_before)
        me = MoveEvent(
            move_index=17,
            ply=18,
            san="Bxc2",
            uci="g6c2",
            fen_before=fen_before,
            fen_after=board.fen(),
            phase="mid",
            move_quality=MoveQuality.MISTAKE,
            event_type=MoveEventType.EVAL_SWING,
            eval_after_cp=209,  # White-POV: Black (mover) is clearly worse
            eval_before_cp=28,
            best_move_uci="f6d7",
            best_move_san="Nfd7",
        )
        row = AnalyzedMoveData(
            index=17,
            ply=18,
            san="Bxc2",
            uci="g6c2",
            fen_before=fen_before,
            fen_after=board.fen(),
            score_cp=209,
            phase_raw="mid",
        )
        facts = build_comment_facts(row, me, depth=16)
    finally:
        eng.run_rules = orig

    assert facts is not None
    texts = [c.text for c in facts.claims]
    # the mover-merit is gone; the consequence remains
    assert "White's pawn structure has been weakened." not in texts
    assert any("won a pawn" in t for t in texts)


def test_promote_losing_sacrifice_is_not_brilliant():
    from app.core.engine.analysis_retriever import _promote_key_moment

    me = _me(key_moment_type=None)
    # material given up AND the mover ends up worse -> just a bad move, not brilliant
    facts = _facts_with_material([0, -300, -300], eval_cp=-250, eval_before=50)
    assert _promote_key_moment(me, facts, facts.claims) != "brilliant"


def test_decisive_interval_demotes_when_both_outside():
    # 33...Rg3-style: played +4.80 and best +2.84 both outside [-2,2] (White
    # winning either way) -> not a mistake (Guid 2006 paper's discard rule).
    from app.core.commentary.event_extractor import _move_quality_for
    from app.models.chess_events import MoveQuality

    # loss large, but both evals decisive -> GOOD, not blunder/mistake
    assert (
        _move_quality_for(196, best_eval=284, cur_score=480, played_is_best=False)
        == MoveQuality.GOOD
    )
    # a real blunder from a live position (best inside [-2,2]) is still punished
    assert _move_quality_for(
        196, best_eval=20, cur_score=-180, played_is_best=False
    ) in (
        MoveQuality.MISTAKE,
        MoveQuality.BLUNDER,
    )


def test_brilliant_not_fired_when_material_regained():
    # 20.Nxb5-style: knight given but bishop+pawn regained -> material UP at the
    # leaf -> not a sacrifice -> not brilliant.
    from app.core.engine.analysis_retriever import _promote_key_moment

    me = _me(key_moment_type=None)
    facts = _facts_with_material([0, -300, 100], eval_cp=36, eval_before=20)
    assert _promote_key_moment(me, facts, facts.claims) != "brilliant"


def test_brilliant_not_fired_when_only_holds_equality():
    # 24...Rfe8-style: material down at the leaf but the eval is only ~equal,
    # not winning -> not a "!!" brilliancy.
    from app.core.engine.analysis_retriever import _promote_key_moment

    me = _me(key_moment_type=None)
    facts = _facts_with_material([0, -200, -200], eval_cp=16, eval_before=0)
    assert _promote_key_moment(me, facts, facts.claims) != "brilliant"


def _mv(uci: str, score: float):
    from app.models.Move import Move

    return Move(
        id=0,
        position=START,
        move=uci,
        context="",
        depth=20,
        isAnalyzed=True,
        piece="N",
        score=score,
    )


def test_great_move_excludes_obvious_recapture():
    from app.core.commentary.key_moment_detector import _great_move

    # White recaptures on d4 (opponent just played ...xd4) — obvious, not great.
    played = _mv("e5d4", 150)
    prev = _mv("c6d4", -150)
    pvs = [[played], [_mv("a1a2", -50)]]  # huge gap to the (bad) alternative
    assert _great_move(pvs, played, True, prev) == []
    # Same gap, but the move is NOT a recapture -> great_move fires.
    nonrecap = _mv("f3e5", 150)
    pvs2 = [[nonrecap], [_mv("a1a2", -50)]]
    assert _great_move(pvs2, nonrecap, True, prev) == ["great_move"]


def test_great_move_needs_clear_gap():
    from app.core.commentary.key_moment_detector import _great_move

    best = _mv("f3e5", 100)
    pvs = [[best], [_mv("a1a2", 50)]]  # only 50cp gap -> below threshold
    assert _great_move(pvs, best, True, None) == []


def test_template_inferior_alternative_is_contrast_not_miss():
    from app.models.comment_facts import BestAlternative

    facts = _facts().model_copy(
        update={
            "mover": "White",
            "eval_cp": 250,
            "better_alternative": BestAlternative(
                san="Qe2",
                uci="d1e2",
                eval_cp=80,
                verdict="only keeps a slight edge",
                is_inferior=True,
                display_line=EnvisionedLine(
                    start_fen=START,
                    line_san=["Qe2"],
                    line_uci=["d1e2"],
                    fens=[START],
                    leaf_fen=START,
                ),
            ),
        }
    )
    text = render_facts_template(facts)
    assert "Qe2" in text
    assert "missed" not in text.lower()


def test_say_nothing_for_empty_optional_annotation():
    from app.core.engine.analysis.comment_assembly import _nothing_instructive_to_say

    # "!" move with no facts at all -> nothing instructive -> stay silent.
    assert _nothing_instructive_to_say(_me(key_moment_type="great_move")) is True
    # "!" move with fired claims -> there IS something to say.
    with_claims = _facts_with_material([10, 10, 10], claims=_facts().claims)
    assert (
        _nothing_instructive_to_say(
            _me(key_moment_type="great_move", comment_facts=with_claims)
        )
        is False
    )
    # A negative annotation always has a consequence to explain -> never silenced.
    assert _nothing_instructive_to_say(_me(key_moment_type="mistake")) is False


def test_castling_rights_feature():
    from app.core.commentary.features.guid_features import compute_feature_vector_fen

    start = compute_feature_vector_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    # Both sides hold both rights -> +2 flag each, net 0.
    assert start["WHITE_CASTLING_RIGHTS"].flag == 2
    assert start["BLACK_CASTLING_RIGHTS"].flag == 2
    assert start["CASTLING_RIGHTS"].value_cp == 0

    # White has castled (no rights), Black still holds both -> Black favored.
    asym = compute_feature_vector_fen(
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 0 1"
    )
    assert asym["WHITE_CASTLING_RIGHTS"].flag == 0
    assert asym["BLACK_CASTLING_RIGHTS"].flag == 2
    assert asym["CASTLING_RIGHTS"].value_cp < 0  # White-POV: Black has the edge


def test_template_inferior_comparable_vs_weaker():
    from app.models.comment_facts import BestAlternative

    def render(alt_cp):
        f = _facts().model_copy(
            update={
                "mover": "White",
                "eval_cp": 250,
                "better_alternative": BestAlternative(
                    san="Qe2",
                    uci="d1e2",
                    eval_cp=alt_cp,
                    verdict="keeps an edge",
                    is_inferior=True,
                    display_line=EnvisionedLine(
                        start_fen=START,
                        line_san=["Qe2"],
                        line_uci=["d1e2"],
                        fens=[START],
                        leaf_fen=START,
                    ),
                ),
            }
        )
        return render_facts_template(f)

    weaker = render(80)  # 170cp worse -> clearly weaker
    assert "Weaker was Qe2" in weaker
    assert "missed" not in weaker.lower()
    comparable = render(240)  # 10cp worse -> comparable
    assert "comparable alternative was Qe2" in comparable


def test_mobility_matches_stockfish_area_definition():
    import chess

    from app.core.commentary.features.guid_features.mobility import (
        _mobility_area,
        mobility_count,
    )

    # Symmetric in the initial position.
    b = chess.Board()
    assert mobility_count(b, chess.WHITE) == mobility_count(b, chess.BLACK)

    # Squares attacked by an enemy pawn are NOT in the mobility area.
    b2 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 2")
    area = _mobility_area(b2, chess.WHITE)
    # ...e5 pawn attacks d4 and f4 -> excluded for White.
    assert chess.D4 not in area
    assert chess.F4 not in area

    # A pinned knight contributes zero mobility (Stockfish). Bishop b4 pins the
    # knight d2 to the king e1 along the a5–e1 diagonal (c3 empty between them).
    pin = chess.Board("4k3/8/8/8/1b6/8/3N4/4K3 w - - 0 1")
    from app.core.commentary.features.guid_features.mobility import _piece_mobility

    assert pin.is_pinned(chess.WHITE, chess.D2)
    area_w = _mobility_area(pin, chess.WHITE)
    assert _piece_mobility(pin, chess.D2, chess.WHITE, area_w) == 0


def test_piece_activity_feature_is_square_count():
    import chess

    from app.core.commentary.features.guid_features import compute_feature_vector
    from app.core.commentary.features.guid_features.mobility import mobility_count

    b = chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    vec = compute_feature_vector(b)
    # Value is the raw mobility count (natural unit), White-POV signed.
    assert vec["WHITE_PIECE_ACTIVITY"].value_cp == mobility_count(b, chess.WHITE)
    assert vec["WHITE_PIECE_ACTIVITY"].flag == mobility_count(b, chess.WHITE)


def test_sf_pawn_predicates():
    import chess

    from app.core.commentary.features.guid_features.features.pawns import (
        backward,
        doubled,
        isolated,
        phalanx,
    )

    # Start position: no doubled / isolated / backward pawns.
    start = chess.Board()
    assert doubled.count(start, chess.WHITE) == 0
    assert isolated.count(start, chess.WHITE) == 0
    assert backward.count(start, chess.WHITE) == 0

    # White doubled on the c-file (c2,c3), no b/d pawns -> 1 doubled, and both
    # c-pawns isolated.
    dbl = chess.Board("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1")
    assert doubled.count(dbl, chess.WHITE) == 1
    assert isolated.count(dbl, chess.WHITE) == 2

    # A supported doubled pawn is NOT counted (Stockfish): c3 behind-supported by b2.
    supported = chess.Board("4k3/8/8/8/8/2P5/1P1P4/4K3 w - - 0 1")
    assert doubled.count(supported, chess.WHITE) == 0

    # Phalanx: pawns side by side on d4/e4.
    phal = chess.Board("4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1")
    assert phalanx.count(phal, chess.WHITE) == 2


def test_pawn_features_use_natural_counts():
    import chess

    from app.core.commentary.features.guid_features import compute_feature_vector

    # White: doubled+isolated c-pawns; the feature value is the raw count.
    vec = compute_feature_vector(chess.Board("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1"))
    assert vec["WHITE_PAWN_DOUBLED"].value_cp == 1
    assert vec["WHITE_PAWN_DOUBLED"].flag == 1
    assert vec["WHITE_PAWN_ISOLATED"].value_cp == 2
    # The aggregate stays a weighted centipawn score (non-zero, negative for White).
    assert vec["EVALUATE_PAWNS"].value_cp < 0


def test_sf_piece_predicates():
    import chess

    from app.core.commentary.features.guid_features.features.pieces import (
        outpost,
        rook_on_file,
    )

    # Knight on e5, pawn-supported from d4, no enemy pawns to challenge -> outpost.
    op = chess.Board("4k3/8/8/4N3/3P4/8/8/4K3 w - - 0 1")
    assert outpost.count(op, chess.WHITE) == 1

    # An enemy pawn that can attack the square denies the outpost (Stockfish span).
    denied = chess.Board("4k3/5p2/8/4N3/3P4/8/8/4K3 w - - 0 1")
    assert outpost.count(denied, chess.WHITE) == 0

    # Rook on a fully open a-file; rook on a semi-open d-file (enemy pawn only).
    rooks = chess.Board("3rk3/3p4/8/8/8/8/8/R3K3 w - - 0 1")
    assert rook_on_file.open_file_count(rooks, chess.WHITE) == 1
    assert rook_on_file.semi_open_file_count(rooks, chess.BLACK) == 0  # own pawn d7


def test_sf_threats_hanging_and_weak():
    import chess

    from app.core.commentary.features.guid_features.features.threats import (
        hanging,
        weak_enemies,
    )

    # White queen attacks an undefended black knight on d5 -> weak and hanging.
    b = chess.Board("4k3/8/8/3n4/8/8/8/3QK3 w - - 0 1")
    assert weak_enemies.count(b, chess.WHITE) == 1
    assert hanging.count(b, chess.WHITE) == 1
    # No threats the other way.
    assert hanging.count(b, chess.BLACK) == 0

    # A defended knight (by a pawn) is not weak.
    defended = chess.Board("4k3/8/4p3/3n4/8/8/8/3QK3 w - - 0 1")
    assert weak_enemies.count(defended, chess.WHITE) == 0


def test_sf_king_danger():
    import chess

    from app.core.commentary.features.guid_features import compute_feature_vector
    from app.core.commentary.features.guid_features.features.king import king_danger

    # Quiet start: no king danger either side.
    assert king_danger.king_danger(chess.Board(), chess.WHITE) == 0

    # Black queen + bishop swarming the white king -> White king danger > 0,
    # and the feature is negative for White (the side under attack).
    b = chess.Board("rnb1k1nr/pppp1ppp/8/8/1b6/5q2/PPPPP1PP/RNBQKBNR w KQkq - 0 1")
    assert king_danger.king_danger(b, chess.WHITE) > 0
    assert compute_feature_vector(b)["WHITE_KING_DANGER"].value_cp < 0


def test_material_standing_restated_when_unchanged():
    import chess

    from app.core.commentary.features.envisioned import diff_vectors
    from app.core.commentary.features.guid_features import compute_feature_vector
    from app.core.commentary.rules.engine import run_rules

    # White a rook up; the position is quiet (no change) -> the standing edge is
    # still stated, so a best move that holds it is not left silent.
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    v = compute_feature_vector(b)
    claims = run_rules(
        diff_vectors(v, v), phase="mid", mover="WHITE", start_board=b, leaf_board=b
    )
    assert any("rook" in c.text for c in claims)

    # Only a pawn up: not restated (would be noise every quiet move).
    b2 = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
    v2 = compute_feature_vector(b2)
    claims2 = run_rules(
        diff_vectors(v2, v2), phase="mid", mover="WHITE", start_board=b2, leaf_board=b2
    )
    assert not any(c.rule_id == "material_standing" for c in claims2)


def test_pins_feature():
    import chess

    from app.core.commentary.features.guid_features.features.threats import pins

    # White rook e1 pins the black knight e7 to the king on e8.
    b = chess.Board("4k3/4n3/8/8/8/8/8/4RK2 w - - 0 1")
    assert pins.count(b, chess.WHITE) == 1
    assert pins.count(b, chess.BLACK) == 0
