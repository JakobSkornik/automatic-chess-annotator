"""Prophylaxis and pawn-defect rules (added 2 Nov 2026).

``rule_threat_parried``: what the move prevents. In the starting position the
opponent, if it were to move, could win material on a square (SEE) or mate in
one; at the support node that threat is gone while the threatened piece is
still there (so the move defended, blocked or removed the attacker, rather
than simply moving the piece away). Engine-free and exact for the stated
threat; it does not claim the threat was the opponent's plan.

``rule_pawn_defects``: isolated and backward pawns named with their square, so
a structural weakness is a concrete statement, not an aggregate score.
"""
from __future__ import annotations

import chess

from app.core.commentary.features.guid_features.evidence import evidence_for
from app.core.commentary.features.guid_features.features.pawns import backward as backward_feature
from app.core.commentary.features.guid_features.features.pawns import isolated as isolated_feature
from app.core.commentary.features.see import VALUE, as_side_to_move, best_capture_gain
from app.models.comment_facts import Claim
from app.core.commentary import pipeline_flags

from .context import _Ctx, _side_label

PARRY_CP = 45
DEFECT_CP = 20


def _label(color: chess.Color) -> str:
    return "White" if color else "Black"


def _mate_in_one(board: chess.Board, color: chess.Color) -> chess.Move | None:
    """A mating move for ``color`` if it were ``color``'s turn (None when none or in check)."""
    view = as_side_to_move(board, color)
    if view.is_check():
        return None
    for move in view.legal_moves:
        view.push(move)
        mate = view.is_checkmate()
        view.pop()
        if mate:
            return move
    return None


def opponent_threats(board: chess.Board, mover: chess.Color) -> dict[int, tuple[int, chess.Move]]:
    """Squares of ``mover``'s pieces the opponent could win on (gain, capture) if it were to move."""
    view = as_side_to_move(board, not mover)
    threats = {}
    for square in chess.SquareSet(board.occupied_co[mover]):
        if board.piece_type_at(square) == chess.KING:
            continue
        gain, capture = best_capture_gain(view, square)
        if gain >= VALUE[chess.PAWN] and capture is not None:
            threats[square] = (gain, capture)
    return threats


def rule_threat_parried(ctx: _Ctx) -> list[Claim]:
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    start, node = ctx.start_board, ctx.leaf_board
    mover = ctx.color(ctx.mover)
    side = _side_label(ctx.mover)
    out: list[Claim] = []
    after_threats = opponent_threats(node, mover)
    for square, (gain, capture) in sorted(opponent_threats(start, mover).items(), key=lambda kv: -kv[1][0]):
        piece = start.piece_at(square)
        if node.piece_at(square) != piece:
            continue  # the piece moved away or was exchanged: not a parry
        if square in after_threats:
            continue  # still winnable
        attacker = start.piece_at(capture.from_square)
        if node.piece_at(capture.from_square) == attacker:
            how = "defends"
            verb = f"so that {_label(not mover)}'s {chess.piece_name(attacker.piece_type)} on " \
                   f"{chess.square_name(capture.from_square)} can no longer win it"
        else:
            how = "removes the attacker of"
            verb = f"the {chess.piece_name(attacker.piece_type)} that attacked it is gone"
        text = (f"{side} {how} the {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}: {verb} "
                f"(the threat was {start.san(capture) if start.turn == (not mover) else _san_for(start, not mover, capture)}).")
        if pipeline_flags.enabled("statement_wording"):
            # One plain sentence, without the bracket and its unnumbered move.
            subject = f"{side} {how} the {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}"
            text = f"{subject}, so {verb[len('so that '):]}." if how == "defends" else f"{subject}: {verb}."
        out.append(Claim(rule_id="threat_parried", beneficiary=ctx.mover.lower(), text=text, text_state=text,
                         features_involved=["TACTICAL_EXCHANGE"], delta_cp=PARRY_CP,
                         entity=f"parry:{chess.square_name(square)}",
                         evidence=evidence_for(node, [square, capture.from_square],
                                               roles={square: "subject", capture.from_square: "attacker"},
                                               baseline_squares=[square, capture.from_square])))
        break  # one parry claim per pass: the most valuable threat
    mate_before = _mate_in_one(start, not mover)
    if mate_before is not None and _mate_in_one(node, not mover) is None and not node.is_check():
        king = node.king(mover)
        text = f"{side} parries the mate threat {_san_for(start, not mover, mate_before)}."
        out.append(Claim(rule_id="mate_threat_parried", beneficiary=ctx.mover.lower(), text=text, text_state=text,
                         features_involved=["TACTIC_MATE_THREAT"], delta_cp=PARRY_CP + 20,
                         entity="parry:mate",
                         evidence=evidence_for(node, [mate_before.from_square, mate_before.to_square,
                                                      *([king] if king is not None else [])],
                                               roles={mate_before.to_square: "target",
                                                      **({king: "king"} if king is not None else {})})))
    return out


def _san_for(board: chess.Board, color: chess.Color, move: chess.Move) -> str:
    view = as_side_to_move(board, color)
    try:
        return view.san(move)
    except (ValueError, AssertionError):
        return move.uci()


def rule_pawn_defects(ctx: _Ctx) -> list[Claim]:
    """New isolated or backward pawns, named with their squares (both sides)."""
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out: list[Claim] = []
    start, node = ctx.start_board, ctx.leaf_board
    for color in (chess.WHITE, chess.BLACK):
        side = _label(color)
        for kind, predicate in (("isolated", _isolated), ("backward", _backward)):
            before = {chess.square_file(sq) for sq in predicate(start, color)}
            new = [sq for sq in predicate(node, color) if chess.square_file(sq) not in before]
            if not new:
                continue
            names = ", ".join(chess.square_name(sq) for sq in new)
            noun = "pawn" if len(new) == 1 else "pawns"
            text = f"{side}'s {noun} on {names} {'is' if len(new) == 1 else 'are'} {kind}."
            feature = f"{'WHITE' if color else 'BLACK'}_PAWN_{kind.upper()}"
            out.append(Claim(rule_id=f"{kind}_pawn_created", beneficiary=_label(not color).lower(),
                             text=text, text_state=text, features_involved=[feature], delta_cp=DEFECT_CP,
                             entity=f"{kind}:{''.join(chess.FILE_NAMES[chess.square_file(sq)] for sq in new)}",
                             evidence=evidence_for(node, new, roles={sq: "subject" for sq in new})))
    return out


def _isolated(board: chess.Board, color: chess.Color) -> list[int]:
    return [sq for sq in board.pieces(chess.PAWN, color)
            if not isolated_feature._has_neighbour_on_adjacent_file(board, chess.square_file(sq), color)]


def _backward(board: chess.Board, color: chess.Color) -> list[int]:
    return [sq for sq in board.pieces(chess.PAWN, color) if backward_feature.is_backward(board, sq, color)]


PROPHYLAXIS_RULES = [rule_threat_parried, rule_pawn_defects]
