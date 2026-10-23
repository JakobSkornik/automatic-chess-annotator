"""Compact, replay-checked board evidence for commentary prompts.

This module exposes only facts recoverable from the displayed legal lines and
approved claim metadata. It does not infer tactics, plans, or causal relations.
"""

from __future__ import annotations

import chess

from app.core.commentary.features.guid_features.feature_catalog import CATALOG
from app.models.comment_facts import BestAlternative, Claim, CommentFacts, EnvisionedLine

EVIDENCE_VERSION = "chess_evidence_v1"

_PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_COLORS = ((chess.WHITE, "white"), (chess.BLACK, "black"))


def _board_summary(board: chess.Board) -> dict:
    pieces: dict[str, dict[str, list[str]]] = {}
    material_counts: dict[str, dict[str, int]] = {}
    for color, color_name in _COLORS:
        pieces[color_name] = {}
        material_counts[color_name] = {}
        for piece_type, piece_name in _PIECE_NAMES.items():
            squares = sorted(board.pieces(piece_type, color))
            pieces[color_name][piece_name] = [
                chess.square_name(square) for square in squares
            ]
            if piece_type != chess.KING:
                material_counts[color_name][piece_name] = len(squares)

    pawn_files = []
    for file_index, file_name in enumerate(chess.FILE_NAMES):
        white_pawns = sorted(
            chess.square_name(square)
            for square in board.pieces(chess.PAWN, chess.WHITE)
            if chess.square_file(square) == file_index
        )
        black_pawns = sorted(
            chess.square_name(square)
            for square in board.pieces(chess.PAWN, chess.BLACK)
            if chess.square_file(square) == file_index
        )
        white_rooks = sorted(
            chess.square_name(square)
            for square in board.pieces(chess.ROOK, chess.WHITE)
            if chess.square_file(square) == file_index
        )
        black_rooks = sorted(
            chess.square_name(square)
            for square in board.pieces(chess.ROOK, chess.BLACK)
            if chess.square_file(square) == file_index
        )
        if not white_pawns and not black_pawns:
            status = "open"
        elif not white_pawns:
            status = "semi_open_white"
        elif not black_pawns:
            status = "semi_open_black"
        else:
            status = "closed"
        pawn_files.append(
            {
                "file": file_name,
                "status": status,
                "pawns": {"white": white_pawns, "black": black_pawns},
                "rooks": {"white": white_rooks, "black": black_rooks},
            }
        )

    return {
        "fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "pieces": pieces,
        "material_counts": material_counts,
        "pawn_files": pawn_files,
        "in_check": board.is_check(),
    }


def _feature_evidence(claim: Claim) -> list[dict[str, str]]:
    result = []
    for name in claim.features_involved:
        base_name = name.removeprefix("WHITE_").removeprefix("BLACK_")
        feature = CATALOG.get(base_name)
        result.append(
            {
                "name": name,
                "unit": feature.unit if feature is not None else "unknown",
            }
        )
    return result


def _claim_unavailable(claim: Claim, index: int, reason: str) -> dict:
    return {
        "claim_index": index,
        "rule_id": claim.rule_id,
        "evidence_status": "unavailable",
        "reason": reason,
    }


def _claim_reason(claim: Claim, line: EnvisionedLine) -> tuple[int | None, str | None]:
    ply = claim.support_ply
    if ply is None:
        return None, "missing_support_ply"
    if type(ply) is not int or not 1 <= ply <= len(line.line_uci):
        return None, "invalid_support_ply"
    if claim.support_fen is None:
        return None, "missing_support_fen"
    if claim.support_fen != line.fens[ply - 1]:
        return None, "support_position_mismatch"
    if claim.baseline_fen is None:
        return None, "missing_baseline_fen"
    if claim.baseline_fen != line.start_fen:
        return None, "baseline_position_mismatch"
    if claim.realization not in ("immediate", "envisioned"):
        return None, "unknown_realization"
    if (claim.realization == "immediate") != (ply == 1):
        return None, "timing_mismatch"
    return ply, None


def _replay_branch(
    facts: CommentFacts,
    branch_name: str,
    branch: CommentFacts | BestAlternative,
) -> dict:
    line = branch.display_line
    move_san = facts.san if branch_name == "main" else branch.san
    move_uci = facts.uci if branch_name == "main" else branch.uci
    branch_metadata = {
        "move": {"san": move_san, "uci": move_uci},
        "eval_white_pov_cp": branch.eval_cp,
        "verdict": branch.verdict,
    }
    unavailable_claims = []
    if branch_name == "alternative" and facts.display_line is None:
        unavailable_claims = [
            _claim_unavailable(claim, index, "played_branch_start_unavailable")
            for index, claim in enumerate(branch.claims)
        ]
        return {
            **branch_metadata,
            "status": "unavailable",
            "reason": "played_branch_start_unavailable",
            "line_labels": [],
            "support_positions": [],
            "claims": [],
            "unavailable_claims": unavailable_claims,
        }
    if line is None:
        unavailable_claims = [
            _claim_unavailable(claim, index, "missing_display_line")
            for index, claim in enumerate(branch.claims)
        ]
        return {
            **branch_metadata,
            "status": "unavailable",
            "reason": "missing_display_line",
            "line_labels": [],
            "support_positions": [],
            "claims": [],
            "unavailable_claims": unavailable_claims,
        }
    if not line.evidence_valid:
        unavailable_claims = [
            _claim_unavailable(claim, index, "line_marked_invalid")
            for index, claim in enumerate(branch.claims)
        ]
        return {
            **branch_metadata,
            "status": "invalid",
            "reason": "line_marked_invalid",
            "line_labels": [],
            "support_positions": [],
            "claims": [],
            "unavailable_claims": unavailable_claims,
        }

    try:
        board = chess.Board(line.start_fen)
    except (ValueError, TypeError):
        board = None
    if board is None:
        error = "invalid_start_fen"
    elif not board.is_valid():
        error = "invalid_start_position"
    else:
        error = None

    if error is None:
        if not line.line_uci or line.line_uci[0] != move_uci:
            error = "wrong_first_move"
        elif not line.line_san or line.line_san[0] != move_san:
            error = "wrong_first_san"
        elif ("White" if board.turn == chess.WHITE else "Black") != facts.mover:
            error = "mover_does_not_match_position"
        elif branch_name == "alternative" and facts.display_line is not None and (
            line.start_fen != facts.display_line.start_fen
        ):
            error = "alternative_starts_at_different_position"
        elif len(line.fens) != len(line.line_uci) or len(line.line_san) != len(
            line.line_uci
        ) or not line.line_uci:
            error = "incomplete_trace"

    labels: list[str] = []
    boards_after: list[chess.Board] = []
    if error is None and board is not None:
        for index, uci in enumerate(line.line_uci):
            try:
                move = chess.Move.from_uci(uci)
                san = board.san(move) if move in board.legal_moves else None
            except (ValueError, AssertionError):
                san = None
                move = None
            if move is None or san is None or san != line.line_san[index]:
                error = "illegal_or_mismatched_move"
                break
            labels.append(
                f"{board.fullmove_number}{'.' if board.turn == chess.WHITE else '...'}{san}"
            )
            board.push(move)
            boards_after.append(board.copy(stack=False))
            if board.fen() != line.fens[index]:
                error = "position_trace_mismatch"
                break
        if error is None and board.fen() != line.leaf_fen:
            error = "endpoint_mismatch"

    if error is not None or board is None:
        reason = error or "invalid_start_position"
        unavailable_claims = [
            _claim_unavailable(claim, index, reason)
            for index, claim in enumerate(branch.claims)
        ]
        return {
            **branch_metadata,
            "status": "invalid",
            "reason": reason,
            "line_labels": [],
            "support_positions": [],
            "claims": [],
            "unavailable_claims": unavailable_claims,
        }

    supported_claims = []
    support_plies: set[int] = set()
    for index, claim in enumerate(branch.claims):
        ply, reason = _claim_reason(claim, line)
        if reason is not None or ply is None:
            unavailable_claims.append(
                _claim_unavailable(claim, index, reason or "unsupported_claim")
            )
            continue
        support_plies.add(ply)
        supported_claims.append(
            {
                "claim_index": index,
                "evidence_status": "supported",
                "rule_id": claim.rule_id,
                "text": claim.text,
                "text_state": claim.text_state,
                "beneficiary": claim.beneficiary,
                "concession": claim.is_concession,
                "support_ply": ply,
                "node_anchor": labels[ply - 1],
                "support_fen": claim.support_fen,
                "baseline_fen": claim.baseline_fen,
                "realization": claim.realization,
                "features": _feature_evidence(claim),
            }
        )

    support_positions = [
        {
            "support_ply": ply,
            "node_anchor": labels[ply - 1],
            **(
                {"board_ref": "board_after_move"}
                if ply == 1
                else {"board": _board_summary(boards_after[ply - 1])}
            ),
        }
        for ply in sorted(support_plies)
    ]
    return {
        **branch_metadata,
        "status": "partially_available" if unavailable_claims else "available",
        "line_labels": labels,
        "board_before_move": _board_summary(chess.Board(line.start_fen)),
        "board_after_move": _board_summary(boards_after[0]),
        "support_positions": support_positions,
        "claims": supported_claims,
        "unavailable_claims": unavailable_claims,
    }


def build_evidence_context(facts: CommentFacts) -> dict:
    """Build a JSON-ready context from the branches actually shown to readers.

    Invalid branch lines are marked invalid and contribute no certified board
    positions. Claims with absent or inconsistent support anchors are excluded
    from ``claims`` and listed with a reason under ``unavailable_claims``.
    """
    main = _replay_branch(facts, "main", facts)
    alternative = (
        _replay_branch(facts, "alternative", facts.better_alternative)
        if facts.better_alternative is not None
        else {"status": "not_present", "reason": "no_alternative"}
    )
    branches = {"main": main, "alternative": alternative}
    present = [branch for branch in branches.values() if branch["status"] != "not_present"]
    if not any(branch["status"] in ("available", "partially_available") for branch in present):
        status = "unavailable"
    elif any(branch["status"] != "available" for branch in present):
        status = "partially_available"
    else:
        status = "available"
    return {
        "version": EVIDENCE_VERSION,
        "status": status,
        "mover": facts.mover,
        "played_san": facts.san,
        "played_uci": facts.uci,
        "eval_white_pov_cp": facts.eval_cp,
        "eval_mate": facts.eval_mate,
        "verdict": facts.verdict,
        "branches": branches,
    }
