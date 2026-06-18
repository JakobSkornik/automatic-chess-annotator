"""Envisioned-position machinery (Guid §5.3).

The envisioned position is the end of a *shortened* principal variation:
the engine PV is capped to a displayable length and its tail is trimmed
while the leaf is not quiescent (last move a capture, promotion or check),
to dodge horizon-effect noise. The feature-difference vector between the
starting position and the envisioned position — split into positive
(favoring White) and negative (favoring Black) components — is the raw
material every positional comment is built from (Tables 5.1/5.2).

Everything here is engine-free: PVs and evals come from searches the
engine pass already paid for.
"""

from __future__ import annotations

import os

import chess

from app.core.commentary.features.guid_features import (
    FeatureVector,
    compute_feature_vector,
)
from app.models.comment_facts import EnvisionedLine, FeatureDelta, FeatureDiff


def base_display_plies() -> int:
    """Standard PV length (Guid): 5 plies, extended only until quiescent."""
    try:
        return int(os.environ.get("ENVISIONED_BASE_PLIES", "5"))
    except ValueError:
        return 5


def max_display_plies() -> int:
    """Hard safety cap when extending the base length to reach a quiescent leaf."""
    try:
        return int(os.environ.get("ENVISIONED_MAX_PLIES", "12"))
    except ValueError:
        return 12


_SEE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 99,
}


def is_quiescent(board: chess.Board) -> bool:
    """Static quiescence: not in check and no obviously en-prise piece.

    En-prise = a non-pawn piece of the side *not* to move that the mover can
    capture with a cheaper attacker, or that is attacked and undefended.
    """
    if board.is_check():
        return False
    mover = board.turn
    victim_color = not mover
    for sq, piece in board.piece_map().items():
        if piece.color != victim_color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(mover, sq)
        if not attackers:
            continue
        defenders = board.attackers(victim_color, sq)
        victim_v = _SEE_VALUES[piece.piece_type]
        cheapest = min(_SEE_VALUES[board.piece_at(a).piece_type] for a in attackers)
        if cheapest < victim_v:
            return False
        if not defenders and piece.piece_type != chess.PAWN:
            return False
    return True


def _is_forcing(board_before: chess.Board, move: chess.Move) -> bool:
    """Capture, promotion, or check — moves a quiescent line must not end on."""
    if board_before.is_capture(move) or move.promotion:
        return True
    b = board_before.copy(stack=False)
    b.push(move)
    return b.is_check()


def build_envisioned_line(
    start_fen: str,
    line_uci: list[str],
    *,
    root_eval_cp: int | None = None,
    depth: int | None = None,
    max_plies: int | None = None,
) -> EnvisionedLine:
    """Walk the PV to a standard length, extending only until a quiescent leaf.

    Guid's spec: 5 plies as the standard PV length, or until a quiescent move
    after 5 — so a line never ends mid-capture/-check, but also isn't padded
    out to an arbitrary horizon.
    """
    hard_cap = (
        max_plies
        if max_plies is not None
        else max(base_display_plies(), max_display_plies())
    )
    base = min(base_display_plies(), hard_cap)
    board = chess.Board(start_fen)
    start_q = is_quiescent(board)

    moves: list[chess.Move] = []
    sans: list[str] = []
    fens: list[str] = []
    boards_before: list[chess.Board] = []
    for uci in line_uci[:hard_cap]:
        try:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                break
        except Exception:
            break
        boards_before.append(board.copy(stack=False))
        sans.append(board.san(mv))
        board.push(mv)
        moves.append(mv)
        fens.append(board.fen())

    walked = len(moves)

    def _unsettled(k: int) -> bool:
        """Ply k ends on a forcing move or a non-quiescent leaf."""
        return _is_forcing(boards_before[k - 1], moves[k - 1]) or not is_quiescent(
            chess.Board(fens[k - 1])
        )

    # Standard length is `base` plies; extend forward until the leaf settles
    # (Guid: "5 plies, or until a quiescent move after 5").
    keep = min(base, walked)
    while keep < walked and _unsettled(keep):
        keep += 1
    # If the PV ran out mid-tactic, trim back so the line still ends quietly.
    while keep > 1 and _unsettled(keep):
        keep -= 1
    moves = moves[:keep]
    sans = sans[:keep]
    fens = fens[:keep]
    boards_before = boards_before[:keep]
    trimmed = walked - keep

    leaf_fen = fens[-1] if fens else start_fen
    leaf_q = is_quiescent(chess.Board(leaf_fen))
    return EnvisionedLine(
        start_fen=start_fen,
        line_uci=[m.uci() for m in moves],
        line_san=sans,
        fens=fens,
        leaf_fen=leaf_fen,
        root_eval_cp=root_eval_cp,
        depth=depth,
        trimmed_plies=trimmed,
        start_quiescent=start_q,
        leaf_quiescent=leaf_q,
    )


def feature_diff(
    start_fen: str,
    leaf_fen: str,
    *,
    min_abs_cp: int = 1,
) -> FeatureDiff:
    """Guid diff vector between two positions, sorted by |delta| descending."""
    start_vec = compute_feature_vector(chess.Board(start_fen))
    leaf_vec = compute_feature_vector(chess.Board(leaf_fen))
    return diff_vectors(start_vec, leaf_vec, min_abs_cp=min_abs_cp)


def diff_vectors(
    start_vec: FeatureVector,
    leaf_vec: FeatureVector,
    *,
    min_abs_cp: int = 1,
) -> FeatureDiff:
    positive: list[FeatureDelta] = []
    negative: list[FeatureDelta] = []
    for name in leaf_vec:
        before = start_vec.get(name)
        after = leaf_vec.get(name)
        if before is None or after is None:
            continue
        delta = after.value_cp - before.value_cp
        flag_changed = (
            before.flag is not None
            and after.flag is not None
            and before.flag != after.flag
        )
        if abs(delta) < min_abs_cp and not flag_changed:
            continue
        fd = FeatureDelta(
            name=name,
            delta_cp=delta,
            before_cp=before.value_cp,
            after_cp=after.value_cp,
            flag_before=before.flag,
            flag_after=after.flag,
        )
        if delta > 0:
            positive.append(fd)
        elif delta < 0:
            negative.append(fd)
        elif flag_changed:
            # value unchanged but the count moved (rare) — file under positive
            positive.append(fd)
    positive.sort(key=lambda d: -abs(d.delta_cp))
    negative.sort(key=lambda d: -abs(d.delta_cp))
    return FeatureDiff(positive=positive, negative=negative)


def envisioned_for_played_move(
    fen_before: str,
    played_uci: str,
    after_pv_uci: list[str],
    *,
    played_eval_cp: int | None,
    depth: int | None,
) -> EnvisionedLine:
    """Envisioned line for the played move: played move + engine continuation."""
    return build_envisioned_line(
        fen_before,
        [played_uci, *list(after_pv_uci or [])],
        root_eval_cp=played_eval_cp,
        depth=depth,
    )


def envisioned_for_best_move(
    fen_before: str,
    best_pv_uci: list[str],
    *,
    best_eval_cp: int | None,
    depth: int | None,
) -> EnvisionedLine:
    """Envisioned line for the engine's best move: PV1 from the pre-move position."""
    return build_envisioned_line(
        fen_before,
        list(best_pv_uci or []),
        root_eval_cp=best_eval_cp,
        depth=depth,
    )
