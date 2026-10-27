"""Tests for compact prompt evidence bound to legally replayed branches."""

import chess

from app.core.commentary.phases.composer.evidence import (
    EVIDENCE_VERSION,
    build_evidence_context,
)
from app.models.comment_facts import (
    BestAlternative,
    Claim,
    CommentFacts,
    EnvisionedLine,
)


def _line(start: chess.Board, sans: list[str]) -> EnvisionedLine:
    board = start.copy(stack=False)
    uci, san, fens = [], [], []
    for move_san in sans:
        move = board.parse_san(move_san)
        uci.append(move.uci())
        san.append(board.san(move))
        board.push(move)
        fens.append(board.fen())
    return EnvisionedLine(
        start_fen=start.fen(),
        line_uci=uci,
        line_san=san,
        fens=fens,
        leaf_fen=board.fen(),
    )


def _facts(line: EnvisionedLine, *, san: str, uci: str, mover: str, **kwargs):
    return CommentFacts(
        ply=1,
        san=san,
        uci=uci,
        mover=mover,
        phase="mid",
        verdict="keeps the position level",
        display_line=line,
        **kwargs,
    )


def test_valid_line_exposes_displayed_boards_and_closed_a_file_after_a6():
    start = chess.Board()
    start.push_san("a3")
    line = _line(start, ["a6"])
    facts = _facts(line, san="a6", uci="a7a6", mover="Black", eval_cp=-4)

    context = build_evidence_context(facts)
    branch = context["branches"]["main"]
    file_a = branch["board_after_move"]["pawn_files"][0]

    assert context["version"] == EVIDENCE_VERSION == "chess_evidence_v1"
    assert context["status"] == branch["status"] == "available"
    assert branch["line_labels"] == ["1...a6"]
    assert branch["board_before_move"]["pawn_files"][0]["status"] == "closed"
    assert file_a["status"] == "closed"
    assert file_a["pawns"] == {"white": ["a3"], "black": ["a6"]}
    assert file_a["rooks"] == {"white": ["a1"], "black": ["a8"]}
    assert branch["board_after_move"]["pieces"]["black"]["pawn"] == [
        "a6", "b7", "c7", "d7", "e7", "f7", "g7", "h7"
    ]


def test_file_status_distinguishes_open_and_each_sides_semi_open_files():
    start = chess.Board("4k3/1p6/8/p7/8/8/P1P5/R2K4 w - - 0 1")
    line = _line(start, ["Kc1"])
    facts = _facts(line, san="Kc1", uci="d1c1", mover="White")

    files = build_evidence_context(facts)["branches"]["main"]["board_after_move"][
        "pawn_files"
    ]

    assert [files[ord(file) - ord("a")]["status"] for file in "abcd"] == [
        "closed",
        "semi_open_white",
        "semi_open_black",
        "open",
    ]


def test_claims_are_bound_to_unique_replayed_support_nodes_and_catalog_units():
    start = chess.Board()
    line = _line(start, ["e4", "e5"])
    claims = [
        Claim(
            rule_id="immediate_claim",
            text="White advances a pawn.",
            beneficiary="white",
            features_involved=["PAWN_ADVANCES"],
            support_ply=1,
            support_fen=line.fens[0],
            baseline_fen=line.start_fen,
            realization="immediate",
        ),
        Claim(
            rule_id="activity_improved",
            text="White improves piece activity.",
            text_state="White's pieces are more active.",
            beneficiary="white",
            features_involved=["WHITE_PIECE_ACTIVITY", "WHITE_PAWN_DOUBLED"],
            support_ply=2,
            support_fen=line.fens[1],
            baseline_fen=line.start_fen,
            realization="envisioned",
        ),
        Claim(
            rule_id="space_gained",
            text="White gains space.",
            beneficiary="white",
            features_involved=["WHITE_PIECE_ACTIVITY"],
            support_ply=2,
            support_fen=line.fens[1],
            baseline_fen=line.start_fen,
            realization="envisioned",
        ),
    ]
    facts = _facts(
        line,
        san="e4",
        uci="e2e4",
        mover="White",
        claims=claims,
    )

    branch = build_evidence_context(facts)["branches"]["main"]

    assert branch["line_labels"] == ["1.e4", "1...e5"]
    assert [position["support_ply"] for position in branch["support_positions"]] == [1, 2]
    assert branch["support_positions"][0]["board_ref"] == "board_after_move"
    assert branch["support_positions"][1]["board"]["fen"] == line.fens[1]
    assert [feature["unit"] for feature in branch["claims"][1]["features"]] == [
        "squares",
        "pawns",
    ]
    assert branch["claims"][1]["node_anchor"] == "1...e5"
    assert branch["claims"][1]["realization"] == "envisioned"


def test_alternative_replays_its_own_move_and_board():
    start = chess.Board()
    main_line = _line(start, ["e4"])
    alt_line = _line(start, ["d4"])
    alternative = BestAlternative(
        san="d4",
        uci="d2d4",
        eval_cp=17,
        verdict="keeps the position level",
        display_line=alt_line,
    )
    facts = _facts(
        main_line,
        san="e4",
        uci="e2e4",
        mover="White",
        eval_cp=9,
        better_alternative=alternative,
    )

    branches = build_evidence_context(facts)["branches"]

    assert branches["main"]["move"] == {"san": "e4", "uci": "e2e4"}
    assert branches["alternative"]["move"] == {"san": "d4", "uci": "d2d4"}
    assert branches["main"]["eval_white_pov_cp"] == 9
    assert branches["alternative"]["eval_white_pov_cp"] == 17
    assert branches["main"]["board_after_move"]["fen"] != branches[
        "alternative"
    ]["board_after_move"]["fen"]


def test_illegal_displayed_pv_is_rejected_without_board_evidence():
    start = chess.Board()
    invalid = EnvisionedLine(
        start_fen=start.fen(),
        line_uci=["e2e5"],
        line_san=["e5"],
        fens=[start.fen()],
        leaf_fen=start.fen(),
    )
    facts = _facts(invalid, san="e5", uci="e2e5", mover="White")

    branch = build_evidence_context(facts)["branches"]["main"]

    assert branch["status"] == "invalid"
    assert branch["reason"] == "illegal_or_mismatched_move"
    assert "board_after_move" not in branch
    assert branch["claims"] == []


def test_legacy_claim_without_support_metadata_is_explicitly_unavailable():
    start = chess.Board()
    line = _line(start, ["e4"])
    facts = _facts(
        line,
        san="e4",
        uci="e2e4",
        mover="White",
        claims=[Claim(rule_id="old_claim", text="An old claim.")],
    )

    branch = build_evidence_context(facts)["branches"]["main"]

    assert branch["status"] == "partially_available"
    assert branch["claims"] == []
    assert branch["unavailable_claims"] == [
        {
            "claim_index": 0,
            "rule_id": "old_claim",
            "evidence_status": "unavailable",
            "reason": "missing_support_ply",
        }
    ]


def test_missing_line_with_no_claims_has_explicit_unavailable_status():
    facts = CommentFacts(
        ply=1,
        san="e4",
        uci="e2e4",
        mover="White",
        phase="mid",
        verdict="keeps the position level",
    )

    context = build_evidence_context(facts)

    assert context["status"] == "unavailable"
    assert context["branches"]["main"]["status"] == "unavailable"
    assert context["branches"]["main"]["reason"] == "missing_display_line"
    assert context["branches"]["main"]["move"] == {"san": "e4", "uci": "e2e4"}
    assert context["branches"]["main"]["line_labels"] == []
    assert context["branches"]["main"]["claims"] == []


def test_claim_without_a_displayed_line_is_rejected_with_reason():
    facts = CommentFacts(
        ply=1,
        san="e4",
        uci="e2e4",
        mover="White",
        phase="mid",
        verdict="keeps the position level",
        claims=[Claim(rule_id="legacy", text="An unsupported claim.")],
    )

    branch = build_evidence_context(facts)["branches"]["main"]

    assert branch["status"] == "unavailable"
    assert branch["claims"] == []
    assert branch["unavailable_claims"][0]["reason"] == "missing_display_line"


def test_claim_with_wrong_support_fen_is_not_certified():
    start = chess.Board()
    line = _line(start, ["e4"])
    facts = _facts(
        line,
        san="e4",
        uci="e2e4",
        mover="White",
        claims=[
            Claim(
                rule_id="bad_anchor",
                text="Unsupported location claim.",
                support_ply=1,
                support_fen=start.fen(),
                baseline_fen=line.start_fen,
            )
        ],
    )

    branch = build_evidence_context(facts)["branches"]["main"]

    assert branch["claims"] == []
    assert branch["unavailable_claims"][0]["reason"] == "support_position_mismatch"
