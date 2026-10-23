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
from collections.abc import Callable

import chess

from app.core.commentary.features.guid_features import (
    FeatureVector,
)
from app.models.comment_facts import (
    EnvisionedLine,
    FeatureAdvantage,
    FeatureDelta,
    FeatureDiff,
)


def base_display_plies() -> int:
    """Application display target; Guid does not prescribe a fixed ply count."""
    try:
        return max(1, int(os.environ.get("ENVISIONED_BASE_PLIES", "5")))
    except ValueError:
        return 5


def max_display_plies() -> int:
    """Hard safety cap when extending the base length to reach a quiescent leaf."""
    try:
        return max(1, int(os.environ.get("ENVISIONED_MAX_PLIES", "12")))
    except ValueError:
        return 12


def probe_trim_enabled() -> bool:
    """Opt-out knob for the probe-based horizon-effect trim (Improvement 1),
    following the same env-var convention as ``ENVISIONED_BASE_PLIES`` /
    ``ENVISIONED_MAX_PLIES``. Defaults to enabled."""
    return os.environ.get("ENVISIONED_PROBE_TRIM_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


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
    """Choose a bounded quiet endpoint, preserving terminal forcing lines.

    If no quiet endpoint exists, retain the available tactical continuation
    and mark its evidence unsuitable for strategic claims.
    """
    hard_cap = (
        max(1, max_plies)
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
        if k == 0 or chess.Board(fens[k - 1]).is_game_over():
            return False
        return _is_forcing(boards_before[k - 1], moves[k - 1]) or not is_quiescent(
            chess.Board(fens[k - 1])
        )

    # Extend past the display target to finish forcing play.
    keep = min(base, walked)
    while keep < walked and _unsettled(keep):
        keep += 1
    if keep and _unsettled(keep):
        quiet_prefixes = [k for k in range(1, keep) if not _unsettled(k)]
        if quiet_prefixes:
            keep = quiet_prefixes[-1]
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
        evidence_valid=bool(keep) and not _unsettled(keep),
    )


def trim_envisioned_line_by_probe(
    line: EnvisionedLine,
    prober: Callable[[str], int | None],
    tolerance_cp: int = 150,
) -> EnvisionedLine:
    """Trim the line's tail while a shallow sanity probe contradicts it.

    A statically quiescent leaf can still be tactically loaded (discovered
    attack, skewer). ``prober(fen)`` returns the engine's cheap eval of a
    position; when the leaf's probe disagrees IN SIGN with the line's backed-up
    eval beyond ``tolerance_cp``, tail plies are dropped one at a time (each
    re-probed) and the longest prefix whose eval story survives is kept.
    Engine-free lines and missing evals pass through untouched."""
    if (
        line.root_eval_cp is None
        or abs(line.root_eval_cp) >= 999000
        or len(line.fens) < 2
        or chess.Board(line.leaf_fen).is_game_over()
    ):
        return line

    def _contradicts(idx: int) -> bool:
        leaf_eval = prober(line.fens[idx])
        if leaf_eval is None:
            return False
        # Both scores are White-POV. Equality has no winning-side story.
        return (line.root_eval_cp > tolerance_cp and leaf_eval < -tolerance_cp) or (
            line.root_eval_cp < -tolerance_cp and leaf_eval > tolerance_cp
        )

    if not _contradicts(len(line.fens) - 1):
        return line
    keep = 0
    for k in range(len(line.fens) - 1, 0, -1):
        before = chess.Board(line.start_fen if k == 1 else line.fens[k - 2])
        move = chess.Move.from_uci(line.line_uci[k - 1])
        leaf = chess.Board(line.fens[k - 1])
        if is_quiescent(leaf) and not _is_forcing(before, move) and not _contradicts(k - 1):
            keep = k
            break
    if not keep:
        return line.model_copy(update={"evidence_valid": False})
    trimmed_line = line.model_copy(
        update={
            "line_uci": line.line_uci[:keep],
            "line_san": line.line_san[:keep],
            "fens": line.fens[:keep],
            "leaf_fen": line.fens[keep - 1],
            "trimmed_plies": line.trimmed_plies + (len(line.fens) - keep),
            "leaf_quiescent": is_quiescent(chess.Board(line.fens[keep - 1])),
            "feature_series": {name: values[:keep + 1] for name, values in line.feature_series.items()},
        }
    )
    return trimmed_line


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


def diff_of_diffs(
    played: FeatureDiff,
    alt: FeatureDiff,
    *,
    min_abs_cp: int = 1,
    max_features: int | None = None,
) -> list[FeatureAdvantage]:
    """Feature-diff-of-diffs: for each named feature present in either the
    played move's diff or the alternative's diff, how many more (White-POV)
    centipawns does the ALTERNATIVE'S own start->leaf diff move that feature
    by, compared to the PLAYED move's start->leaf diff on the same feature?

    Every feature name here is already White-POV signed (``WHITE_*``/``BLACK_*``
    are separate, oppositely-signed features per ``vector.py``), so the two
    diffs' ``delta_cp`` values are directly comparable by name without any
    extra POV bookkeeping — unlike claim text, which is deduplicated by rule
    id/beneficiary, this is a genuine number: "the alternative gains N more
    cp of rook activity than the played move does", grounded in the actual
    feature vectors rather than in claim-text absence.

    ``min_abs_cp`` mirrors ``diff_vectors``'s own materiality convention
    (the smallest cp swing worth reporting at all); callers doing user-facing
    surfacing should pass a higher threshold (e.g. Guid's ``min_claim_cp``)
    so only claim-worthy swings show up. Results are sorted by |advantage|
    descending and optionally capped to ``max_features``."""

    def _by_name(diff: FeatureDiff) -> dict[str, int]:
        return {d.name: d.delta_cp for d in (diff.positive + diff.negative)}

    played_map = _by_name(played)
    alt_map = _by_name(alt)
    names = set(played_map) | set(alt_map)

    advantages: list[FeatureAdvantage] = []
    for name in names:
        played_cp = played_map.get(name, 0)
        alt_cp = alt_map.get(name, 0)
        advantage_cp = alt_cp - played_cp
        if abs(advantage_cp) < min_abs_cp:
            continue
        advantages.append(
            FeatureAdvantage(
                name=name,
                alt_delta_cp=alt_cp,
                played_delta_cp=played_cp,
                advantage_cp=advantage_cp,
            )
        )
    advantages.sort(key=lambda a: (-abs(a.advantage_cp), a.name))
    if max_features is not None:
        advantages = advantages[:max_features]
    return advantages


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
