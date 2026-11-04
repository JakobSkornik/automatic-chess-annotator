"""Which key moments each commentary level comments on.

Every key moment is sorted into one kind, from engine data and board facts:

S  simple error     loses >= 100 cp, and the move itself loses material in a
                    capture sequence (a static-exchange fact fires)
T  tactical         loses >= 100 cp otherwise, or only one move works (first
                    choice >= 100 cp ahead of the second), or brilliant/great
M  missed chance    the opponent's previous move was a mistake or blunder and
                    this move does not punish it (loses >= 100 cp)
G  good move        about as good as the engine's best (loses < 50 cp) and
                    concrete: it punishes the opponent's mistake or blunder,
                    wins material in a capture sequence, or executes a tactic
                    (fork, discovered attack or check, double check, mate threat)
Q  quiet decision   a decision point without an error: critical decision,
                    hidden turning point, structural or king-safety turning point
R  routine          everything else (a sound move with nothing to decide);
                    no level comments it

Beginner and intermediate comment S, T, M and G (they differ in how the comment
is written); expert comments T, M, Q and good moves that are decision points.
"""
from __future__ import annotations

from typing import Any

import chess

ERROR_CP = 100
GOOD_MOVE_LOSS_CP = 50
UNIQUE_MARGIN_CP = 100
SIMPLE_FACTS = {"exchange_loses_material", "piece_left_en_prise"}
TACTIC_FACTS = {"fork", "discovered_attack", "discovered_check", "double_check", "mate_threat"}
LEVEL_KINDS = {"beginner": {"S", "T", "M", "G"}, "intermediate": {"S", "T", "M", "G"}, "expert": {"T", "M", "Q"}}
ALWAYS = {"checkmate", "game_end"}
DECISIONS = {"critical_decision", "hidden_inflection", "structural_transformation", "king_safety_crisis",
             "initiative_shift", "kingside_attack", "good_defense", "feature_event", "decision_point"}


def _loss(event: Any) -> int | None:
    selection = ((getattr(event.comment_facts, "selection_metadata", None) or {}).get("comment_selection") or {})
    return selection.get("verified_loss_cp")


def moment_kind(events: list[Any], index: int) -> str:
    event = events[index]
    facts = event.comment_facts
    loss = _loss(event) or 0
    claims = getattr(facts, "claims", None) or []
    simple = any(c.rule_id in SIMPLE_FACTS and c.support_ply == 1 for c in claims)
    previous = events[index - 1] if index > 0 else None
    opponent_erred = bool(previous and getattr(previous.move_quality, "value", previous.move_quality)
                          in ("mistake", "blunder"))
    if loss >= ERROR_CP and opponent_erred:
        return "M"
    if loss >= ERROR_CP:
        return "S" if simple else "T"
    if event.key_moment_type == "missed_opportunity":
        return "M"
    margin = getattr(facts, "best_margin_cp", None)
    if event.key_moment_type in ("brilliant", "great_move") or (margin is not None and margin >= UNIQUE_MARGIN_CP):
        return "T"
    if loss < GOOD_MOVE_LOSS_CP and (opponent_erred or _wins_material(event) or event.key_moment_type == CONCRETE_GAIN
                                     or any(c.rule_id in TACTIC_FACTS and c.support_ply == 1 for c in claims)):
        return "G"
    return "Q" if event.key_moment_type in DECISIONS or event.key_moment_type == DECISION_POINT else "R"


def _wins_material(event: Any) -> bool:
    """The played move is a capture that comes out ahead in the exchange on its square."""
    from app.core.commentary.rules.exchange import MATERIAL_GAIN_CP, see
    try:
        board = chess.Board(event.fen_before)
        move = chess.Move.from_uci(event.uci)
    except (ValueError, AttributeError):
        return False
    return move in board.legal_moves and board.is_capture(move) and see(board, move) >= MATERIAL_GAIN_CP


def level_allows(events: list[Any], index: int, level: str | None) -> bool:
    """Whether ``level`` comments this key moment (no level: every key moment)."""
    event = events[index]
    if level not in LEVEL_KINDS or event.key_moment_type in ALWAYS:
        return True
    if event.key_moment_type == SEQUENCE:
        return False  # sequence comments are added per level by attach_sequences
    kind = moment_kind(events, index)
    if level == "expert" and kind == "G" and event.key_moment_type in DECISIONS:
        return True  # a good move that is also a decision point stays an expert moment
    if level == "expert" and kind in ("T", "S") and already_lost(event):
        return False  # losing more in a lost position teaches an expert nothing
    return kind in LEVEL_KINDS[level]


LOST_FOR_MOVER_CP = 200  # the mover was already this far behind before the move


def has_mover_merit(event: Any) -> bool:
    """At least one kept claim benefits the side that moved and is not a concession."""
    facts = getattr(event, "comment_facts", None)
    if facts is None:
        return False
    mover = "white" if chess.Board(event.fen_before).turn == chess.WHITE else "black"
    return any(getattr(c, "beneficiary", None) == mover and not getattr(c, "is_concession", False)
               for c in facts.claims)


def already_lost(event: Any) -> bool:
    """The mover was clearly lost before the move (mover POV), so an error here only cashes in."""
    before = getattr(event, "eval_before_cp", None)
    if before is None:
        return False
    mover_white = chess.Board(event.fen_before).turn == chess.WHITE
    return (before if mover_white else -before) <= -LOST_FOR_MOVER_CP


# ---------------------------------------------------------------- moments from facts (A)

# Facts that make a sound move worth a comment even when the key-moment
# detector did not flag it: the mover gains something concrete.
CONCRETE_GAIN_FACTS = TACTIC_FACTS | {"material_won", "passed_pawn_advances", "passed_pawn_created",
                                      "unstoppable_passer", "strong_knight_established",
                                      "rook_reaches_seventh", "piece_trapped"}
CONCRETE_GAIN = "concrete_gain"


def _mover_loss(event: Any) -> int | None:
    """Mover-POV centipawns the played move gives up against the engine's choice (None: unknown)."""
    best, played = event.best_move_eval_cp, event.eval_after_cp
    if best is None or played is None or abs(best) > 90000 or abs(played) > 90000:
        return None
    sign = 1 if chess.Board(event.fen_before).turn == chess.WHITE else -1
    return max(0, sign * (best - played))


def _concrete_gain(event: Any) -> bool:
    facts = event.comment_facts
    if facts is None or facts.terminal_result:
        return False
    mover = facts.mover.lower()
    return any(c.rule_id in CONCRETE_GAIN_FACTS and c.beneficiary == mover and c.realization == "immediate"
               for c in facts.claims)


def promote_concrete_moments(events: list[Any]) -> None:
    """Make a sound, unflagged move with a concrete gain for the mover a key moment."""
    for i, event in enumerate(events):
        # A move the back-to-back rule demoted (brief_commentary) stays a short reference.
        if getattr(event, "key_moment_type", None) or getattr(event, "brief_commentary", False)                 or getattr(event, "comment_facts", None) is None or not _concrete_gain(event):
            continue
        loss = _mover_loss(event)
        if loss is not None and loss < GOOD_MOVE_LOSS_CP:
            events[i] = event.model_copy(update={"key_moment_type": CONCRETE_GAIN})


# ---------------------------------------------------------------- sequences (B)

# Uncommented plies after which a level gets one comment on the sequence, and
# the longest sequence one comment covers. Beginners get no sequence comments.
SEQUENCE_GAP = {"expert": 6, "intermediate": 10}
SEQUENCE_MAX_PLIES = 12
SEQUENCE_FACTS_PER_MOVE = 3
SEQUENCE = "sequence"


def _label(event: Any) -> str:
    board = chess.Board(event.fen_before)
    return f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}{event.san}"


def sequence_segments(events: list[Any], selected: list[int], level: str | None) -> list[tuple[int, int]]:
    """(first, last) event indices of uncommented stretches that get one sequence comment.

    Only middlegame and endgame moves count (they have facts); a long stretch is
    split into sequences of at most SEQUENCE_MAX_PLIES plies."""
    gap = SEQUENCE_GAP.get(level or "")
    if not gap:
        return []
    commented = set(selected)
    segments, run = [], []
    for i, event in enumerate(events):
        facts = getattr(event, "comment_facts", None)
        eligible = facts is not None and not getattr(facts, "terminal_result", None)
        if i in commented or not eligible:
            if len(run) >= gap:
                segments.extend(_split(run))
            run = []
            continue
        run.append(i)
    if len(run) >= gap:
        segments.extend(_split(run))
    return segments


def _split(run: list[int]) -> list[tuple[int, int]]:
    parts = [run[k:k + SEQUENCE_MAX_PLIES] for k in range(0, len(run), SEQUENCE_MAX_PLIES)]
    if len(parts) > 1 and len(parts[-1]) < SEQUENCE_MAX_PLIES // 2:
        tail = parts.pop()
        parts[-1] = parts[-1] + tail  # fold a short tail into the previous sequence
    return [(part[0], part[-1]) for part in parts]


def describe_sequence(events: list[Any], first: int, last: int) -> dict:
    """The moves of a sequence with what each move itself did (its immediate facts)."""
    moves = []
    for event in events[first:last + 1]:
        facts = event.comment_facts
        said = [c.text for c in (facts.claims if facts else []) if c.realization == "immediate"]
        moves.append({"move": _label(event), "facts": said[:SEQUENCE_FACTS_PER_MOVE]})
    return {"from": moves[0]["move"], "to": moves[-1]["move"], "moves": moves}


def attach_sequences(events: list[Any], selected: list[int], level: str | None) -> list[int]:
    """Mark the last move of each sequence for a comment; returns those indices."""
    anchors = []
    for first, last in sequence_segments(events, selected, level):
        event = events[last]
        event.comment_facts.selection_metadata["sequence"] = describe_sequence(events, first, last)
        update = {"brief_commentary": False, "commentary_stub_ref_ply": None}
        if not event.key_moment_type:
            update["key_moment_type"] = SEQUENCE
        events[last] = event.model_copy(update=update)
        anchors.append(last)
    return anchors


# ---------------------------------------------------------------- coverage selection (1 Nov 2026, revised 2 Nov 2026)
#
# A move is commented when it brings something new. Walking the game forward:
#   error          loses >= 100 cp (every level, always);
#   only move      the engine's first choice is >= 100 cp ahead of its second (intermediate, expert);
#   decision point the choice mattered: the first choice is >= DECISION_MARGIN_CP ahead of the
#                  second, or the top two are close but lead to different feature stories
#                  (feature divergence), or the detector flagged a decision (intermediate, expert);
#   feature event  a rule fires on the move itself (immediate facts): beginner a concrete gain
#                  or tactic; intermediate a concrete tactic or material gain. Expert never
#                  comments a move only because a feature changed (2 Nov 2026).
# After a comment, game moves that follow that comment's engine line are covered: the
# comment already explained them, so feature events there are not commented again (an
# error never is covered). Feature events also keep a minimum distance from the last comment.

FEATURE_EVENT = "feature_event"
ONLY_MOVE = "only_move"
DECISION_POINT = "decision_point"
MIN_SPACING = {"beginner": 4, "intermediate": 3, "expert": 2}
DECISION_MARGIN_CP = 75  # first choice this far ahead of the second: the choice mattered
CLOSE_MARGIN_CP = 20  # top two this close: a real choice between plans
DIVERGENCE_CP = 30  # a feature group where the two lines differ by this much
DIVERGENT_GROUPS = 2  # how many such groups make the lines different plans
# Concrete facts: what a beginner learns from (gains, tactics, losses on the board).
CONCRETE_FACTS = CONCRETE_GAIN_FACTS | SIMPLE_FACTS | {"hanging_piece", "absolute_pin", "piece_pinned",
                                                       "back_rank_weakness", "castling_rights_lost"}
# Facts that are bookkeeping rather than a change worth a comment of its own.
NOT_EVENTS = {"legal_capture", "exchange_sequence", "material_standing", "eval_concession",
              "tactical_material_loss"}


def feature_events(event: Any) -> tuple[list[str], list[str]]:
    """(concrete, positional) rule ids that fire on the move itself (immediate facts)."""
    facts = getattr(event, "comment_facts", None)
    if facts is None:
        return [], []
    immediate = [c.rule_id for c in facts.claims if c.realization == "immediate" and c.rule_id not in NOT_EVENTS]
    return ([r for r in immediate if r in CONCRETE_FACTS], [r for r in immediate if r not in CONCRETE_FACTS])


def feature_divergence(event: Any) -> int:
    """Feature groups where the played line and the alternative differ by DIVERGENCE_CP or more."""
    facts = getattr(event, "comment_facts", None)
    alt = getattr(facts, "better_alternative", None) if facts else None
    if alt is None:
        return 0
    advantages = list(getattr(alt, "feature_advantages_matched", None) or []) or list(alt.feature_advantages or [])
    groups = set()
    for advantage in advantages:
        if abs(advantage.advantage_cp) >= DIVERGENCE_CP:
            name = advantage.name.removeprefix("WHITE_").removeprefix("BLACK_")
            groups.add(name)
    return len(groups)


def is_decision_point(event: Any) -> bool:
    """The move was a real choice: a clear margin, or close options leading to different plans."""
    facts = getattr(event, "comment_facts", None)
    if facts is None:
        return False
    loss = _loss(event) or 0
    margin = getattr(facts, "best_margin_cp", None)
    if loss < GOOD_MOVE_LOSS_CP and margin is not None and margin >= DECISION_MARGIN_CP:
        return True
    if margin is not None and margin <= CLOSE_MARGIN_CP and feature_divergence(event) >= DIVERGENT_GROUPS:
        return True
    # A detector-flagged decision counts; a bare feature event does not.
    return event.key_moment_type in DECISIONS - {"feature_event"}


def _event_trigger(event: Any, level: str) -> bool:
    concrete, positional = feature_events(event)
    if level == "beginner":
        return bool(concrete)
    tactic = any(r in TACTIC_FACTS or r in ("material_won", "threat_parried", "mate_threat_parried") for r in concrete)
    if level == "intermediate":
        return tactic or is_decision_point(event)
    return is_decision_point(event)


def select_by_coverage(events: list[Any], candidates: list[int], level: str | None) -> tuple[list[int], set[int]]:
    """(indices to comment, latest first; indices covered by an earlier comment's line).

    ``candidates`` are the moves that may be commented: the requested side's moves that
    passed the comment gate (quiet positions, or a verified error)."""
    level = level if level in MIN_SPACING else "intermediate"
    allowed = set(candidates)
    selected, covered, last_ply = [], set(), None
    expected: list[str] = []  # the engine line of the last comment, still followed by the game
    for i, event in enumerate(events):
        facts = getattr(event, "comment_facts", None)
        follows = bool(expected) and expected[0] == event.uci
        expected = expected[1:] if follows else []
        if follows:
            covered.add(i)
        if i not in allowed or facts is None:
            continue
        loss = _loss(event) or 0
        margin = getattr(facts, "best_margin_cp", None)
        error = loss >= ERROR_CP or event.key_moment_type in ALWAYS
        only = level != "beginner" and loss < GOOD_MOVE_LOSS_CP and margin is not None and margin >= UNIQUE_MARGIN_CP
        spaced = last_ply is None or event.ply - last_ply >= MIN_SPACING[level]
        trigger = error or (only and not follows) or (not follows and spaced and _event_trigger(event, level))
        if not trigger:
            continue
        selected.append(i)
        last_ply = event.ply
        line = facts.display_line.line_uci if facts.display_line else []
        expected = list(line[1:])
    return sorted(selected, key=lambda i: -events[i].ply), covered - set(selected)


def mark_selected(events: list[Any], selected: list[int]) -> None:
    """Give selected moves without a key-moment type one, so the commentary gate admits them."""
    for i in selected:
        event = events[i]
        if event.key_moment_type:
            continue
        margin = getattr(event.comment_facts, "best_margin_cp", None)
        kind = (ONLY_MOVE if margin is not None and margin >= UNIQUE_MARGIN_CP
                else DECISION_POINT if is_decision_point(event) else FEATURE_EVENT)
        events[i] = event.model_copy(update={"key_moment_type": kind, "brief_commentary": False,
                                             "commentary_stub_ref_ply": None})


# ---------------------------------------------------------------- complexity for survey positions (1 Nov 2026)

DECIDED_CP = 300  # a game beyond +-3 pawns before the move is no longer open
NARROW_MARGIN_CP = 50
# A tactic the player had to see; a loose pawn is not one (en-prise facts removed 2 Nov 2026).
TACTIC_IN_LINE = TACTIC_FACTS | {"piece_trapped", "threat_parried", "mate_threat_parried"}
# Feature events that count toward complexity: not the bookkeeping of a loose piece.
NOT_COMPLEX = {"hanging_piece", "piece_left_en_prise", "exchange_loses_material"}


def complexity(events: list[Any], index: int) -> int | None:
    """How demanding a position is for a 1500-2000 player (higher: more complex); None
    when unsuitable: the game is decided, or the move is a simple one-move capture loss.

    Points: a narrow choice (only a few good moves, or an error that is not a simple
    capture loss) 3; a tactic in the better line the player missed or had to find 3;
    each distinct feature event at the move, up to 4; a quiet start position 1."""
    from app.core.commentary.features.envisioned import is_quiescent

    event = events[index]
    facts = getattr(event, "comment_facts", None)
    if facts is None or event.eval_before_cp is None or abs(event.eval_before_cp) > DECIDED_CP:
        return None
    kind = moment_kind(events, index)
    if kind == "S" or (kind == "T" and already_lost(event)):
        return None
    if not facts.claims:
        return None  # nothing to explain the move with
    if kind in ("G", "R") and not has_mover_merit(event):
        return None  # a sound move whose facts only describe the opponent's gains
    loss = _loss(event) or 0
    margin = getattr(facts, "best_margin_cp", None)
    score = 0
    if loss >= ERROR_CP or (margin is not None and margin >= NARROW_MARGIN_CP):
        score += 3
    alternative = facts.better_alternative
    lines = [facts] + ([alternative] if alternative is not None else [])
    if any(c.rule_id in TACTIC_IN_LINE for record in lines for c in record.claims):
        score += 3
    concrete, positional = feature_events(event)
    score += min(4, len((set(concrete) | set(positional)) - NOT_COMPLEX))
    if is_quiescent(chess.Board(event.fen_before)):
        score += 1
    return score
