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
             "initiative_shift", "kingside_attack", "good_defense"}


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
    return "Q" if event.key_moment_type in DECISIONS else "R"


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
    return kind in LEVEL_KINDS[level]


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
