"""Tests for the annotation-quality phases:

P1 intent claims (targets piece / pawn break / open file)
P2 king-zone-attacks feature + kingside_attack key moment
P3 endgame_transition firing + deterministic sentence
P4 prose variety (template variants rotate; win-note budget)
"""

import chess

from app.core.commentary.features.envisioned import diff_vectors
from app.core.commentary.features.guid_features import compute_feature_vector
from app.core.commentary.features.intent import (
    build_intent_claim,
    file_opened_for_rook,
    newly_attacked_enemy_piece,
    pawn_break_prepared,
)
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.phases.composer import render_facts_template
from app.core.commentary.rules.catalog import run_rules
from app.models.Move import Move
from app.models.chess_events import MoveEvent, MoveEventType, MoveQuality

START = chess.STARTING_FEN


def _move(depth: int, uci: str, score: int | None = None, hidden=None) -> Move:
    return Move(
        id=depth,
        position=START,
        move=uci,
        context="mainline",
        isAnalyzed=True,
        depth=depth,
        piece="wP",
        score=score,
        hiddenFeatures=hidden or {},
    )


# ---------------------------------------------------------------------------
# P1: intent predicates
# ---------------------------------------------------------------------------


def test_newly_attacked_enemy_piece():
    # White queen lifts a1-a5 -> attacks the d5 knight that was not attacked.
    start = chess.Board("3k4/8/8/3n1b2/8/8/8/Q3K3 w - - 0 1")
    leaf = start.copy(stack=False)
    leaf.push(chess.Move.from_uci("a1a5"))
    got = newly_attacked_enemy_piece(start, leaf, chess.WHITE)
    assert got is not None
    sq, label = got
    assert label == "knight"
    assert chess.square_name(sq) == "d5"


def test_newly_attacked_ignores_already_attacked():
    # Bishop already attacked by the c4 pawn — a quiet knight move must not
    # report it as newly attacked.
    start = chess.Board("4k3/8/8/3b4/2P5/8/1N6/4K3 w - - 0 1")
    leaf = start.copy(stack=False)
    leaf.push(chess.Move.from_uci("b2d3"))  # knight moves away from b2
    got = newly_attacked_enemy_piece(start, leaf, chess.WHITE)
    if got is not None:
        assert chess.square_name(got[0]) != "d5"


def test_pawn_break_prepared_supported_phalanx():
    # White plays d4: phalanx c3-d4 facing d5 enemy pawn? Use a clean setup:
    # white pawns c3+e4? Simplest true break: white pawns on b2,c2? Actually a
    # supported single pawn advance facing an enemy pawn: white d3-d4 with c3
    # support vs black d5.
    start = chess.Board("rn1qkbnr/ppp1pppp/8/3p4/8/2PP4/PP2PPPP/RNBQKB1R w - - 0 1")
    leaf = start.copy(stack=False)
    leaf.push(chess.Move.from_uci("d3d4"))
    got = pawn_break_prepared(start, leaf, chess.WHITE)
    assert got == "d"


def test_file_opened_for_rook():
    start = chess.Board("4k3/8/8/8/8/P7/8/R3K3 w - - 0 1")
    leaf = start.copy(stack=False)
    leaf.push(chess.Move.from_uci("a3a4"))  # pawn leaves the a-file
    assert file_opened_for_rook(start, leaf, chess.WHITE) == 0


def test_rule_intent_fires_in_rules_pass():
    start = chess.Board("3k4/8/8/3n1b2/8/8/8/Q3K3 w - - 0 1")
    leaf = start.copy(stack=False)
    leaf.push(chess.Move.from_uci("a1a5"))
    vb = compute_feature_vector(start)
    va = compute_feature_vector(leaf)
    claims = run_rules(
        diff_vectors(vb, va),
        phase="mid",
        mover="WHITE",
        eval_cp=10,
        start_board=start,
        leaf_board=leaf,
    )
    assert any(c.rule_id == "intent_targets_piece" for c in claims)


# ---------------------------------------------------------------------------
# P2: king zone attacks + kingside_attack key moment
# ---------------------------------------------------------------------------


def test_king_zone_attacks_feature_counts_attackers():
    from app.core.commentary.features.guid_features import vector_to_plain

    b = chess.Board("rnbqkb1r/pppp1ppp/8/8/8/5Q2/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    vec = compute_feature_vector(b)
    plain = vector_to_plain(vec)
    # Qf3 aims at the black king's zone via the f-file/diagonal region.
    assert plain["WHITE_KING_ZONE_ATTACKS"]["flag"] >= 1


def _guid_hidden(zone_attacks: int, side_prefix: str) -> dict:
    return {
        "_guid": {
            f"{side_prefix}_KING_ZONE_ATTACKS": {"v": zone_attacks, "flag": zone_attacks},
            "MATERIAL_BALANCE": {"v": 0, "flag": 32},
        }
    }


def test_kingside_attack_arc_fires_on_jump():
    det = KeyMomentDetector()
    prev = _move(20, "e2e4", score=10, hidden=_guid_hidden(1, "WHITE"))
    curr = _move(21, "a2a4", score=12, hidden=_guid_hidden(3, "WHITE"))
    got = det.detect(curr, prev, pvs_for_move=None)
    assert got == "kingside_attack"


def test_kingside_attack_no_fire_without_jump():
    det = KeyMomentDetector()
    prev = _move(20, "e2e4", score=10, hidden=_guid_hidden(2, "WHITE"))
    curr = _move(21, "a2a4", score=12, hidden=_guid_hidden(2, "WHITE"))
    got = det.detect(curr, prev, pvs_for_move=None)
    assert got != "kingside_attack"


# ---------------------------------------------------------------------------
# Pin motif: only a pin the move actually created
# ---------------------------------------------------------------------------


def test_pre_existing_pin_is_not_credited_to_the_move():
    """Qe2 already pins the e6 pawn to Ke8 before 10.Nxf7; the capture must not
    be reported as creating a pin (and a pinned pawn is not a pinned piece)."""
    from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
    from app.models.chess_events import TacticalMotif

    before = chess.Board(
        "rn1qkb1r/pp3ppp/2p1pn2/4N3/2B3PP/2N5/PPbPQP2/R1B1K2R w KQkq - 0 10"
    )
    move = chess.Move.from_uci("e5f7")
    assert move in before.legal_moves
    after = before.copy(stack=False)
    after.push(move)
    motifs = detect_tactical_motifs(before, after, move, 300, 300)
    assert TacticalMotif.PIN not in motifs


def test_new_pin_is_detected():
    from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
    from app.models.chess_events import TacticalMotif

    # Re1 pins the e6 knight to the king on e8 down the open e-file.
    before = chess.Board("4k3/8/4n3/8/8/8/8/3R2K1 w - - 0 1")
    move = chess.Move.from_uci("d1e1")
    assert move in before.legal_moves
    after = before.copy(stack=False)
    after.push(move)
    assert TacticalMotif.PIN in detect_tactical_motifs(before, after, move, 0, 0)


# ---------------------------------------------------------------------------
# P3: endgame transition
# ---------------------------------------------------------------------------


def test_endgame_transition_detected_on_queen_trade():
    from app.core.commentary.event_extractor import ChessEventExtractor

    ext = ChessEventExtractor()
    before = chess.Board(
        "r4rk1/pp3ppp/2n1b3/3p4/3P4/2N1B3/PP3PPP/R2Q1RK1 w - - 0 10"
    )
    after = before.copy(stack=False)
    after.push(chess.Move.from_uci("d1d8"))  # Qxd8 ... simplified queens off
    after.push(chess.Move.from_uci("f8d8"))
    got = ext._transition_for(before, after)
    assert got == "queens_off"


def _transition_event(**over) -> MoveEvent:
    base = dict(
        move_index=0,
        ply=40,
        san="Qxd8",
        uci="d1d8",
        fen_before=START,
        fen_after=START,
        phase="middlegame",
        move_quality=MoveQuality.GOOD,
        event_type=MoveEventType.QUIET,
        eval_after_cp=-60,
        eval_swing_cp=-120,
        phase_transition="queens_off",
    )
    base.update(over)
    return MoveEvent(**base)


def test_endgame_transition_sentence_renders():
    from app.core.engine.analysis.comment_assembly import _transition_sentence

    text = _transition_sentence(_transition_event(), "queens_off")
    assert "queens are off" in text
    assert "+1.20" in text or "-1.20" in text


def test_transition_sentence_appends_to_existing_prose():
    """The queen-trade ply usually carries its own comment; the transition line
    must ride along with it rather than be dropped."""
    from app.core.engine.analysis.comment_assembly import (
        _MoveComment,
        _append_transition_sentence,
    )

    mc = _MoveComment(comment="Recaptures the queen.")
    _append_transition_sentence(mc, _transition_event(), True, {})
    assert mc.comment.startswith("Recaptures the queen.")
    assert "queens are off" in mc.comment


def test_transition_sentence_emitted_once_per_boundary():
    from app.core.engine.analysis.comment_assembly import (
        _MoveComment,
        _append_transition_sentence,
    )

    seen: dict[str, int] = {}
    first, second = _MoveComment(), _MoveComment()
    _append_transition_sentence(first, _transition_event(), True, seen)
    _append_transition_sentence(second, _transition_event(ply=60), True, seen)
    assert first.comment and second.comment is None


def test_adjacent_boundaries_yield_one_sentence():
    """A queen trade drops the piece count through the endgame threshold a ply
    later; the second boundary would only restate the first."""
    from app.core.engine.analysis.comment_assembly import (
        _MoveComment,
        _append_transition_sentence,
    )

    seen: dict[str, int] = {}
    queens, pieces = _MoveComment(), _MoveComment()
    _append_transition_sentence(queens, _transition_event(ply=40), True, seen)
    _append_transition_sentence(
        pieces, _transition_event(ply=41, phase_transition="endgame"), True, seen
    )
    assert queens.comment and pieces.comment is None


def test_distant_boundaries_both_speak():
    from app.core.engine.analysis.comment_assembly import (
        _MoveComment,
        _append_transition_sentence,
    )

    seen: dict[str, int] = {}
    queens, pieces = _MoveComment(), _MoveComment()
    _append_transition_sentence(queens, _transition_event(ply=40), True, seen)
    _append_transition_sentence(
        pieces, _transition_event(ply=70, phase_transition="endgame"), True, seen
    )
    assert queens.comment and pieces.comment


def test_transition_survives_a_quality_tag():
    """A queen trade that is also a mistake keeps its transition signal: the two
    live on separate fields instead of competing for key_moment_type."""
    me = _transition_event(key_moment_type="mistake")
    from app.core.engine.analysis.comment_assembly import (
        _MoveComment,
        _append_transition_sentence,
    )

    mc = _MoveComment(comment="A mistake that loses the thread.")
    _append_transition_sentence(mc, me, True, {})
    assert "queens are off" in (mc.comment or "")


# ---------------------------------------------------------------------------
# P4: template variety
# ---------------------------------------------------------------------------


def _facts(ply: int):
    line = EnvisionedLine(
        start_fen=START,
        line_san=["Nc3", "Bd6"],
        line_uci=["b1c3", "f8d6"],
        fens=[START, START],
        leaf_fen=START,
        root_eval_cp=10,
    )
    return CommentFacts(
        ply=ply,
        san="Nc3",
        uci="b1c3",
        mover="White",
        phase="mid",
        verdict="holds the balance",
        eval_cp=10,
        depth=16,
        display_line=line,
        claims=[],
    )


def test_template_head_rotates_over_three_plies():
    texts = [render_facts_template(_facts(p)) for p in (11, 12, 13)]
    shapes = set()
    for t in texts:
        for shape in ("after [pv:", ": [pv:", "([pv:"):
            if shape in t:
                shapes.add(shape)
    assert len(shapes) == 3


from app.models.comment_facts import CommentFacts, EnvisionedLine  # noqa: E402
