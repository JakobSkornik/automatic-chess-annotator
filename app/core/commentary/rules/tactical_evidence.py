"""Concrete events proved by replaying this branch, without gain/force inference."""
from __future__ import annotations

import chess
from collections import Counter

from app.models.comment_facts import Claim, EnvisionedLine


CONCRETE_IDS = {"legal_capture", "absolute_pin", "exchange_sequence",
                "exchange_loses_material", "piece_left_en_prise", "fork", "discovered_attack",
                "discovered_check", "double_check", "mate_threat"}
_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}


def takes_attacker_safely(board: chess.Board, square: int, attacker_squares) -> bool:
    """Whether the piece on ``square`` can legally capture one of these attackers
    without losing material on that capture: the attacker is undefended or worth
    at least as much. Such a piece is not really threatened, pinned or targeted
    (a pawn on c3 "pinned" by a queen on d4 simply takes the queen)."""
    piece = board.piece_at(square)
    if piece is None:
        return False
    view = board.copy(stack=False)
    if view.turn != piece.color:
        view.ep_square = None
    view.turn = piece.color
    targets = set(attacker_squares)
    for move in view.generate_legal_captures():
        if move.from_square != square or move.to_square not in targets:
            continue
        taken = board.piece_at(move.to_square)
        defended = board.is_attacked_by(taken.color, move.to_square)
        if not defended or _VALUE[taken.piece_type] >= _VALUE[piece.piece_type]:
            return True
    return False


def absolute_pins(board: chess.Board, color: chess.Color) -> set[tuple[int, int, int]]:
    """Attacker, victim and king squares for absolute sliding-piece pins.

    A pinned piece that can safely capture its pinner (a pawn on c3 against a
    queen on d4) is not restrained, so that geometry is not reported as a pin."""
    king = board.king(not color)
    if king is None:
        return set()
    result = set()
    for victim in board.piece_map():
        piece = board.piece_at(victim)
        if piece.color == color or not board.is_pinned(not color, victim):
            continue
        for attacker in board.attackers(color, victim):
            slider = board.piece_type_at(attacker)
            if slider not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                continue
            if victim in chess.SquareSet(chess.between(attacker, king)) and not takes_attacker_safely(board, victim, [attacker]):
                result.add((attacker, victim, king))
    return result


def replay(line: EnvisionedLine) -> list[chess.Board]:
    """Reject the entire branch on illegal moves or inconsistent stored nodes."""
    if not line.evidence_valid or len(line.line_uci) != len(line.fens):
        return []
    try:
        board = chess.Board(line.start_fen)
        boards = [board.copy()]
        for uci, fen in zip(line.line_uci, line.fens):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                return []
            board.push(move)
            if board.fen() != chess.Board(fen).fen():
                return []
            boards.append(board.copy())
        if board.fen() != chess.Board(line.leaf_fen).fen():
            return []
        return boards
    except ValueError:
        return []


def tactical_claims(line: EnvisionedLine) -> list[Claim]:
    boards = replay(line)
    claims = []
    # Root action, reply, and immediate follow-through; deeper events remain
    # the variation's evidence, rather than being attributed to the root move.
    for ply in range(1, min(len(boards), 4)):
        before, after = boards[ply - 1:ply + 1]
        move = chess.Move.from_uci(line.line_uci[ply - 1])
        color = before.turn
        side = "White" if color else "Black"
        # Geometry proves an event, not its centipawn value or profitability.
        common = dict(beneficiary=side.lower(), delta_cp=0,
                      realization="immediate" if ply == 1 else "envisioned",
                      support_ply=ply, support_fen=after.fen(), baseline_fen=line.start_fen)
        if before.is_capture(move):
            square = move.to_square
            if before.is_en_passant(move):
                square += -8 if color else 8
            victim = before.piece_at(square)
            attacker = before.piece_at(move.from_square)
            text = (f"{side}'s {chess.piece_name(attacker.piece_type)} "
                    f"from {chess.square_name(move.from_square)} captures the "
                    f"{chess.piece_name(victim.piece_type)} on {chess.square_name(square)}.")
            claims.append(Claim(rule_id="legal_capture", text=text, text_state=text,
                                features_involved=["TACTICAL_CAPTURE"], **common))
        if ply == 1:
            # Why the move itself loses material (static exchange), at the move only.
            from .exchange import material_loss_claims
            from .move_tactics import move_tactic_claims
            root = {k: v for k, v in common.items() if k != "beneficiary"}
            claims.extend(material_loss_claims(before, move, after, root))
            claims.extend(move_tactic_claims(before, move, after, root))
        for attacker, victim, king in sorted(absolute_pins(after, color) - absolute_pins(before, color)):
            text = (f"{side}'s {chess.piece_name(after.piece_type_at(attacker))} "
                    f"on {chess.square_name(attacker)} pins the "
                    f"{chess.piece_name(after.piece_type_at(victim))} on {chess.square_name(victim)} "
                    f"to the king on {chess.square_name(king)}.")
            claims.append(Claim(rule_id="absolute_pin", text=text, text_state=text,
                                features_involved=["TACTICAL_ABSOLUTE_PIN"], **common))
    # Describe only the pieces actually removed in the displayed exchange.
    # This is deliberately not a statement about gain, force, or intention.
    end = 0
    target = None
    losses: dict[chess.Color, list[int]] = {chess.WHITE: [], chess.BLACK: []}
    for ply in range(1, min(len(boards), 4)):
        before = boards[ply - 1]
        move = chess.Move.from_uci(line.line_uci[ply - 1])
        if not before.is_capture(move) or before.is_en_passant(move):
            break
        if target is not None and move.to_square != target:
            break
        target = move.to_square
        victim = before.piece_at(target)
        losses[victim.color].append(victim.piece_type)
        end = ply
    # Do not summarize a three-ply prefix of a longer same-square exchange.
    continues = False
    if end and end < len(line.line_uci):
        following = chess.Move.from_uci(line.line_uci[end])
        continues = boards[end].is_capture(following) and following.to_square == target
    if end >= 2 and not continues:
        def pieces(types: list[int]) -> str:
            parts = []
            for piece_type, count in sorted(Counter(types).items()):
                name = chess.piece_name(piece_type)
                parts.append((f"an {name}" if name[0] in "aeiou" else f"a {name}")
                             if count == 1 else f"{count} {name}s")
            return " and ".join(parts)

        text = (f"In the displayed exchange on {chess.square_name(target)}, "
                f"White loses {pieces(losses[chess.WHITE])}, while "
                f"Black loses {pieces(losses[chess.BLACK])}.")
        summary = Claim(rule_id="exchange_sequence", text=text, text_state=text,
                        features_involved=["TACTICAL_CAPTURE", "TACTICAL_EXCHANGE"],
                        beneficiary="white" if boards[0].turn else "black",
                        delta_cp=0, realization="envisioned", support_ply=end,
                        support_fen=boards[end].fen(), baseline_fen=line.start_fen)
        claims = [summary] + [c for c in claims if not (
            c.rule_id == "legal_capture" and (c.support_ply or 0) <= end)]
    return claims
