"""Shared sacrifice-detection core (Guid): a real sacrifice is a material
deficit for the mover that PERSISTS to a settled/quiescent position, with
real compensation (the position stays objectively fine, not merely lost).

This is used by both:

- the general-purpose tactical-motif layer
  (``TacticalMotif.SACRIFICE`` / ``POSITIONAL_PAWN_SAC`` in
  ``tactical_motifs/detectors.py``), and
- the brilliancy-specific key-moment promotion in
  ``engine.analysis.key_moments`` (which layers its own extra gates —
  played-is-best, clearly *winning* not merely fine — on top).

Both previously computed "material lost" by diffing the mover's material one
ply apart on the SAME half-move (before vs. immediately after the mover's own
move). Since a single legal move can never reduce the *mover's own* material
total (a capture only removes the opponent's material), that comparison could
never fire — the bug this module fixes. The correct comparison walks forward
past the opponent's reply (and further, while the position keeps trading) to
a settled/quiescent leaf, mirroring the PV-leaf concept in ``envisioned.py``,
then compares the NET (White-minus-Black) material balance there against the
balance before the move. A transient dip that a recapture immediately erases
(an ordinary trade) does not count; a deficit that lasts to the settled leaf
does.

Material is scored net, White-POV, in centipawns — the same convention as
``MATERIAL_BALANCE`` in ``guid_features/vector.py`` (and the same piece
values, ``weights.py``'s ``material_*`` entries) — rather than a one-sided
"mover's own piece count", so an exchange sac (rook for knight) reads as the
~170cp net swing it actually is, not the full rook's face value.
"""

from __future__ import annotations

from collections.abc import Sequence

import chess

from app.core.commentary.features.envisioned import is_quiescent

# Net-material piece values (cp), matching guid_features/weights.py's
# material_pawn/knight/bishop/rook/queen. Kings are excluded — they can never
# be captured, so they would only ever cancel out between the two sides.
PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# Persistent net deficit (mover-POV cp) required to call a move a
# general-purpose sacrifice — roughly "an exchange or more", not a mere pawn.
SACRIFICE_MATERIAL_MIN_CP = 200
# A "positional pawn sac" is the same mechanism narrowed to a deficit that
# reads as "about one pawn" rather than a bigger sacrifice.
POSITIONAL_PAWN_SAC_MIN_CP = 80
POSITIONAL_PAWN_SAC_MAX_CP = SACRIFICE_MATERIAL_MIN_CP
# Mover-POV cp floor: the position must still be objectively fine (not
# clearly losing) despite the persistent deficit, i.e. real compensation.
# Mirrors the "at least this much worse is misleading" semantics of
# MERIT_SUPPRESS_CP in commentary/rules/constants.py.
SACRIFICE_EVAL_FLOOR_CP = -100
# Depth budget for the local settle-to-quiescence walk below.
SETTLE_MAX_PLIES = 6


def material_balance_cp(board: chess.Board) -> int:
    """Net White-minus-Black material balance, in centipawns."""
    total = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None:
            continue
        v = PIECE_VALUES_CP.get(p.piece_type)
        if v is None:
            continue
        total += v if p.color == chess.WHITE else -v
    return total


def _greedy_recapture(board: chess.Board) -> chess.Move | None:
    """Largest available capture — a local, engine-free stand-in for "the
    forced recapture/follow-up trade a PV would play out", used only to
    settle material once no PV is available at the call site."""
    best: chess.Move | None = None
    best_val = -1
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        victim = board.piece_at(move.to_square)
        victim_val = (
            PIECE_VALUES_CP.get(victim.piece_type, 0)
            if victim is not None
            else PIECE_VALUES_CP[chess.PAWN]  # en passant: victim square is empty
        )
        if victim_val > best_val:
            best_val = victim_val
            best = move
    return best


def settle_position(
    board: chess.Board, *, max_plies: int = SETTLE_MAX_PLIES
) -> chess.Board:
    """Walk forward from ``board`` (already one ply past the played move)
    playing the largest available capture each ply, until the position is
    quiescent (``envisioned.is_quiescent``) or the ply budget runs out.

    This always represents at least the played move's opponent reply (the
    caller passes the post-move board) and extends further only while the
    position is still tactically unsettled, so a transient one-ply dip that
    gets recaptured back to equality does not read as a persistent deficit.
    """
    b = board.copy(stack=False)
    plies = 0
    while plies < max_plies and not is_quiescent(b):
        move = _greedy_recapture(b)
        if move is None:
            break
        b.push(move)
        plies += 1
    return b


def persistent_deficit(values: Sequence[float], sign: int) -> float:
    """Signed persistent deficit (mover-POV) between the first and last of a
    value series. A transient dip the series regains by its end reads as
    ~0 — this is what separates a real sacrifice from an ordinary trade or
    combination, whether ``values`` is a MATERIAL_BALANCE cp series across a
    full envisioned line (key_moments) or a two-point [before, settled-leaf]
    net-balance sample (the tactical-motif detectors below)."""
    if len(values) < 2:
        return 0.0
    return sign * (values[-1] - values[0])


def settled_material_deficit(
    board_before: chess.Board,
    board_after: chess.Board,
    mover_color: chess.Color,
    *,
    settle_plies: int = SETTLE_MAX_PLIES,
) -> float:
    """Persistent NET material deficit (mover-POV cp, positive = mover down)
    between the position before the move and the settled/quiescent leaf
    reached after it."""
    bal_before = material_balance_cp(board_before)
    leaf = settle_position(board_after, max_plies=settle_plies)
    bal_leaf = material_balance_cp(leaf)
    mover_sign = 1 if mover_color == chess.WHITE else -1
    # mover-POV deficit = (sign*bal_before) - (sign*bal_leaf)
    #                   = -sign * (bal_leaf - bal_before)
    return persistent_deficit([bal_before, bal_leaf], sign=-mover_sign)


def is_sacrifice(
    board_before: chess.Board,
    board_after: chess.Board,
    mover_color: chess.Color,
    eval_after_cp: int | None,
    *,
    min_deficit_cp: float = SACRIFICE_MATERIAL_MIN_CP,
    eval_floor_cp: int = SACRIFICE_EVAL_FLOOR_CP,
    settle_plies: int = SETTLE_MAX_PLIES,
) -> bool:
    """General-purpose sacrifice predicate: a persistent material deficit at
    the settled leaf, with real compensation — the position must still be
    objectively fine for the mover (not clearly losing), so an ordinary
    blunder that loses material and stays lost is not mislabeled a
    sacrifice.

    Deliberately omits the brilliancy-specific gates (played move must be the
    engine's literal top choice; position must be clearly *winning*, not
    merely fine) — those live in ``engine.analysis.key_moments`` layered on
    top of this same core.
    """
    if eval_after_cp is None:
        return False
    deficit = settled_material_deficit(
        board_before, board_after, mover_color, settle_plies=settle_plies
    )
    if deficit < min_deficit_cp:
        return False
    sign = 1 if mover_color == chess.WHITE else -1
    return sign * eval_after_cp >= eval_floor_cp
