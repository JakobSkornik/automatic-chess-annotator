"""Structural board facts, compared between the start position and a support node.

Each rule reports a state that is new at the node (absent at the start) and is
exact on the board: trapped pieces, the king's pawn cover, pawn storms, open
files at the king, pawn majorities, candidate passed pawns, holes and pawn
races (rule of the square). Definitions follow the Chess Programming Wiki
pages on evaluation, simplified to what can be stated as a plain fact.
"""
from __future__ import annotations

import chess

from app.models.comment_facts import Claim
from app.core.commentary.features.guid_features.evidence import evidence_for

from .context import _Ctx
from .exchange import VALUE, _captured_value, best_capture_gain

# Board facts are concrete; give them a weight that competes with positional swings.
BOARD_FACT_CP = 40
RACE_CP = 80
WINGS = {"queenside": range(0, 4), "kingside": range(4, 8)}


def _label(color: chess.Color) -> str:
    return "White" if color else "Black"


def _as(board: chess.Board, color: chess.Color) -> chess.Board:
    view = board.copy(stack=False)
    if view.turn != color:
        view.ep_square = None
    view.turn = color
    return view


def _rel_rank(square: int, color: chess.Color) -> int:
    rank = chess.square_rank(square)
    return rank if color else 7 - rank


def _claim(rule_id: str, beneficiary: chess.Color, text: str, feature: str, delta: int = BOARD_FACT_CP,
           *, board: chess.Board | None = None, squares=(), roles=None, entity: str | None = None,
           files=()) -> Claim:
    evidence = evidence_for(board, squares, roles=roles, files=files) if board is not None and squares else None
    return Claim(rule_id=rule_id, beneficiary=_label(beneficiary).lower(), text=text, text_state=text,
                 features_involved=[feature], delta_cp=delta, entity=entity,
                 **({"evidence": evidence} if evidence is not None else {}))


# ---------------------------------------------------------------- trapped pieces

def _still_lost(view: chess.Board, move: chess.Move, square: int) -> bool:
    """After ``move`` by the owner, the opponent still wins material on the piece from ``square``.

    What the rescue move itself captures counts against the loss: a rook that
    takes a rook and is then recaptured has been exchanged, not lost (fixed
    2 Nov 2026; before, every such trade counted as a trapped piece)."""
    after = view.copy(stack=False)
    taken = _captured_value(view, move)
    after.push(move)
    if after.is_check():
        return False  # a check gains a tempo; do not claim the piece is lost
    target = move.to_square if move.from_square == square else square
    gain, _ = best_capture_gain(after, target)
    return gain - taken >= VALUE[chess.PAWN]


def trapped_squares(board: chess.Board, color: chess.Color) -> set[int]:
    """Pieces of ``color`` that the opponent can win and that no move saves.

    Every legal move of the owner is tried, not only the piece's own moves: a
    piece that can be defended, blocked for or freed by a capture is not trapped."""
    trapped = set()
    opponent_view, own_view = _as(board, not color), _as(board, color)
    for square in chess.SquareSet(board.occupied_co[color]):
        piece_type = board.piece_type_at(square)
        if piece_type in (chess.PAWN, chess.KING):
            continue
        gain, _ = best_capture_gain(opponent_view, square)
        if gain < VALUE[chess.PAWN]:
            continue
        if all(_still_lost(own_view, move, square) for move in own_view.legal_moves):
            trapped.add(square)
    return trapped


def rule_trapped_piece(ctx: _Ctx) -> list[Claim]:
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out = []
    for color in (chess.WHITE, chess.BLACK):
        before = trapped_squares(ctx.start_board, color)
        for square in sorted(trapped_squares(ctx.leaf_board, color) - before):
            name = chess.piece_name(ctx.leaf_board.piece_type_at(square))
            attackers = list(ctx.leaf_board.attackers(not color, square))
            out.append(_claim("piece_trapped", not color,
                              f"{_label(color)}'s {name} on {chess.square_name(square)} is trapped: it can be won, "
                              f"and every square it can move to loses it as well.", "TRAPPED_PIECE",
                              board=ctx.leaf_board, squares=[square, *attackers],
                              roles={square: "target", **{a: "attacker" for a in attackers}},
                              entity=f"trapped:{chess.square_name(square)}"))
    return out


# ---------------------------------------------------------------- king shelter

def _castled_king(board: chess.Board, color: chess.Color) -> int | None:
    """The king on its back rank on a wing (files a-c or f-h): a castled position."""
    king = board.king(color)
    if king is None or _rel_rank(king, color) != 0 or chess.square_file(king) in (3, 4):
        return None
    return king


def _shield_files(king: int) -> list[int]:
    file = chess.square_file(king)
    return [f for f in (file - 1, file, file + 1) if 0 <= f <= 7]


def shield_count(board: chess.Board, color: chess.Color, king: int) -> int:
    """Own pawns directly in front of a castled king (its file and the two next to it, two ranks)."""
    rank = _rel_rank(king, color)
    return sum(1 for sq in board.pieces(chess.PAWN, color)
               if chess.square_file(sq) in _shield_files(king) and 1 <= _rel_rank(sq, color) - rank <= 2)


def storm_squares(board: chess.Board, color: chess.Color, king: int) -> set[int]:
    """Enemy pawns on the king's files within three ranks in front of the king."""
    rank = _rel_rank(king, color)
    return {sq for sq in board.pieces(chess.PAWN, not color)
            if chess.square_file(sq) in _shield_files(king) and 1 <= _rel_rank(sq, color) - rank <= 3}


def storm_advanced(start: chess.Board, node: chess.Board, color: chess.Color, king: int) -> bool:
    """An enemy pawn pushed forward on its own file into the storm zone (captures do not count)."""
    for square in storm_squares(node, color, king) - storm_squares(start, color, king):
        file = chess.square_file(square)
        behind = [sq for sq in start.pieces(chess.PAWN, not color)
                  if chess.square_file(sq) == file and _rel_rank(sq, color) > _rel_rank(square, color)]
        if behind and not node.piece_at(behind[0]):
            return True
    return False


def _open_king_files(board: chess.Board, color: chess.Color, king: int) -> set[int]:
    own = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
    return {f for f in _shield_files(king) if f not in own}


def rule_king_shelter(ctx: _Ctx) -> list[Claim]:
    """Pawn cover thinned, pawn storm, and a file opened at a castled king (enemy queen on board)."""
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out = []
    start, node = ctx.start_board, ctx.leaf_board
    for color in (chess.WHITE, chess.BLACK):
        king = _castled_king(node, color)
        if king is None or king != start.king(color) or not node.pieces(chess.QUEEN, not color):
            continue
        side, square = _label(color), chess.square_name(king)
        ring = [king, *chess.SquareSet(chess.BB_KING_ATTACKS[king])]
        shield_now = [sq for sq in node.pieces(chess.PAWN, color)
                      if chess.square_file(sq) in _shield_files(king)
                      and 1 <= _rel_rank(sq, color) - _rel_rank(king, color) <= 2]
        heavy = node.pieces(chess.ROOK, not color) | node.pieces(chess.QUEEN, not color)
        zone_now = sorted({a for sq in ring for a in node.attackers(not color, sq)
                           if node.piece_type_at(a) != chess.PAWN})
        zone_start = sorted({a for sq in ring for a in start.attackers(not color, sq)
                             if start.piece_type_at(a) != chess.PAWN})
        thinner = shield_count(node, color, king) < shield_count(start, color, king)
        opened = sorted(_open_king_files(node, color, king) - _open_king_files(start, color, king))
        heavy_on_open = [sq for sq in heavy if chess.square_file(sq) in opened]
        if (thinner or opened) and (heavy_on_open or len(zone_now) - len(zone_start) >= 2):
            # Combination rule (2 Nov 2026): shelter damage plus the pieces that
            # exploit it, stated as one attacking plan instead of two counts.
            file_names = " and ".join(f"{chess.FILE_NAMES[f]}-file" for f in opened)
            what = f"the open {file_names}" if opened else "the thinned pawn cover"
            who = "heavy pieces" if heavy_on_open else "pieces"
            out.append(_claim("king_attack_builds", not color,
                              f"With {what} next to {side}'s king on {square}, {_label(not color)}'s {who} "
                              f"bear down on the king.", "KING_OPEN_FILE" if opened else "PAWN_SHIELD", 60,
                              board=node, squares=[king, *shield_now, *heavy_on_open, *zone_now],
                              roles={king: "king", **{a: "attacker" for a in [*heavy_on_open, *zone_now]}},
                              entity=f"king_attack:{square}", files=[chess.FILE_NAMES[f] for f in opened]))
            continue
        if thinner:
            out.append(_claim("king_shelter_weakened", not color,
                              f"The pawn cover in front of {side}'s king on {square} is thinner.", "PAWN_SHIELD",
                              board=node, squares=[king, *shield_now], roles={king: "king"},
                              entity=f"shelter:{square}"))
        if storm_advanced(start, node, color, king):
            storm = storm_squares(node, color, king)
            out.append(_claim("pawn_storm", not color,
                              f"{_label(not color)}'s pawns advance toward {side}'s king on {square}.", "PAWN_STORM",
                              board=node, squares=[king, *storm],
                              roles={king: "king", **{p: "attacker" for p in storm}}, entity=f"storm:{square}"))
        for file in opened:
            if heavy:
                out.append(_claim("king_file_opened", not color,
                                  f"The {chess.FILE_NAMES[file]}-file next to {side}'s king on {square} has no "
                                  f"{side} pawn any more.", "KING_OPEN_FILE",
                                  board=node, squares=[king, *[sq for sq in heavy if chess.square_file(sq) == file]],
                                  roles={king: "king"}, entity=f"king_file:{chess.FILE_NAMES[file]}",
                                  files=[chess.FILE_NAMES[file]]))
    return out


# ---------------------------------------------------------------- pawn structure

def _no_pieces(board: chess.Board) -> bool:
    return not any(board.pieces(t, c) for t in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
                   for c in (chess.WHITE, chess.BLACK))


def majorities(board: chess.Board, color: chess.Color) -> set[str]:
    own = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)]
    other = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)]
    return {wing for wing, files in WINGS.items()
            if sum(f in files for f in own) > sum(f in files for f in other)}


def _is_passed(board: chess.Board, square: int, color: chess.Color) -> bool:
    file, rank = chess.square_file(square), _rel_rank(square, color)
    return not any(abs(chess.square_file(sq) - file) <= 1 and _rel_rank(sq, color) > rank
                   for sq in board.pieces(chess.PAWN, not color))


def candidate_passers(board: chess.Board, color: chess.Color) -> set[int]:
    """Not yet passed, no enemy pawn in front on its file, and at least as many own
    helpers on the neighbouring files as enemy pawns guarding its path."""
    result = set()
    for square in board.pieces(chess.PAWN, color):
        file, rank = chess.square_file(square), _rel_rank(square, color)
        if _is_passed(board, square, color):
            continue
        if any(chess.square_file(sq) == file and _rel_rank(sq, color) > rank for sq in board.pieces(chess.PAWN, not color)):
            continue
        sentries = sum(1 for sq in board.pieces(chess.PAWN, not color)
                       if abs(chess.square_file(sq) - file) == 1 and _rel_rank(sq, color) > rank)
        helpers = sum(1 for sq in board.pieces(chess.PAWN, color)
                      if abs(chess.square_file(sq) - file) == 1 and _rel_rank(sq, color) <= rank + 1)
        if helpers + 1 > sentries:
            result.add(square)
    return result


def holes(board: chess.Board, color: chess.Color) -> set[int]:
    """Central squares on ranks 3-4 (from ``color``'s side) that no ``color`` pawn can attack any
    more, and that an enemy knight or bishop already occupies or attacks (a usable outpost)."""
    result = set()
    minors = board.pieces(chess.KNIGHT, not color) | board.pieces(chess.BISHOP, not color)
    for file in range(2, 6):
        for rel in (2, 3):
            rank = rel if color else 7 - rel
            square = chess.square(file, rank)
            if board.piece_type_at(square) == chess.PAWN and board.color_at(square) == color:
                continue
            coverable = any(abs(chess.square_file(sq) - file) == 1 and _rel_rank(sq, color) < rel
                            for sq in board.pieces(chess.PAWN, color))
            used = square in minors or any(square in board.attacks(m) for m in minors)
            if not coverable and used:
                result.add(square)
    return result


def rule_pawn_plans(ctx: _Ctx) -> list[Claim]:
    """Majorities and candidate passers (endgames), holes (middlegames), as new states."""
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out = []
    start, node = ctx.start_board, ctx.leaf_board
    endgame = not (node.pieces(chess.QUEEN, chess.WHITE) or node.pieces(chess.QUEEN, chess.BLACK)) or ctx.phase == "end"
    for color in (chess.WHITE, chess.BLACK):
        side = _label(color)
        if endgame:
            for wing in sorted(majorities(node, color) - majorities(start, color)):
                wing_pawns = [sq for c in (color, not color) for sq in node.pieces(chess.PAWN, c)
                              if chess.square_file(sq) in WINGS[wing]]
                out.append(_claim("pawn_majority", color, f"{side} now has a pawn majority on the {wing}.",
                                  "PAWN_MAJORITY", 30, board=node, squares=wing_pawns, entity=f"majority:{wing}"))
            for square in sorted(candidate_passers(node, color) - candidate_passers(start, color)):
                file = chess.square_file(square)
                guards = [sq for sq in node.pieces(chess.PAWN, not color)
                          if abs(chess.square_file(sq) - file) <= 1 and _rel_rank(sq, color) > _rel_rank(square, color)]
                helpers = [sq for sq in node.pieces(chess.PAWN, color) if abs(chess.square_file(sq) - file) == 1]
                out.append(_claim("candidate_passer", color,
                                  f"{side}'s pawn on {chess.square_name(square)} can become a passed pawn: no enemy pawn "
                                  f"stands in front of it and {side}'s pawns outnumber the pawns guarding its path.",
                                  "CANDIDATE_PASSER", 30, board=node, squares=[square, *helpers, *guards],
                                  roles={square: "subject", **{g: "target" for g in guards}},
                                  entity=f"candidate:{chess.FILE_NAMES[file]}"))
        else:
            minors = node.pieces(chess.KNIGHT, not color) | node.pieces(chess.BISHOP, not color)
            for square in sorted(holes(node, color) - holes(start, color)):
                users = [m for m in minors if m == square or square in node.attacks(m)]
                occupant = node.piece_at(square)
                settled = (occupant is not None and occupant.color != color
                           and occupant.piece_type in (chess.KNIGHT, chess.BISHOP)
                           and start.piece_at(square) != occupant)
                if settled:
                    # Combination rule (2 Nov 2026): the hole and the piece that uses it.
                    out.append(_claim("outpost_occupied", not color,
                                      f"{_label(not color)}'s {chess.piece_name(occupant.piece_type)} settles on "
                                      f"{chess.square_name(square)}, a square no {side} pawn can drive it from.",
                                      "HOLES", 45, board=node, squares=[square], roles={square: "subject"},
                                      entity=f"hole:{chess.square_name(square)}"))
                    continue
                out.append(_claim("hole_created", not color,
                                  f"The square {chess.square_name(square)} becomes a hole in {side}'s position: "
                                  f"no {side} pawn can attack it any more.", "HOLES", 30,
                                  board=node, squares=[square, *users],
                                  roles={square: "target", **{u: "attacker" for u in users}},
                                  entity=f"hole:{chess.square_name(square)}"))
    return out


# ---------------------------------------------------------------- pawn races

def unstoppable_passers(board: chess.Board, color: chess.Color) -> set[int]:
    """Rule of the square, in pawn endings: passed pawns the enemy king cannot catch."""
    if not _no_pieces(board):
        return set()
    king = board.king(not color)
    result = set()
    for square in board.pieces(chess.PAWN, color):
        if not _is_passed(board, square, color):
            continue
        rank = _rel_rank(square, color)
        steps = 7 - max(rank, 2)  # a pawn on its start rank may advance two squares
        promotion = chess.square(chess.square_file(square), 7 if color else 0)
        own_path = [chess.square(chess.square_file(square), r if color else 7 - r) for r in range(rank + 1, 8)]
        if any(board.piece_at(sq) for sq in own_path):
            continue  # the path is blocked; the race is not a simple count
        # The defending king catches the pawn if it reaches the promotion square in time:
        # within the pawn's steps, or one step fewer when the pawn's side moves first.
        allowed = steps - 1 if board.turn == color else steps
        if chess.square_distance(king, promotion) > allowed:
            result.add(square)
    return result


def rule_pawn_race(ctx: _Ctx) -> list[Claim]:
    if ctx.start_board is None or ctx.leaf_board is None:
        return []
    out = []
    for color in (chess.WHITE, chess.BLACK):
        for square in sorted(unstoppable_passers(ctx.leaf_board, color) - unstoppable_passers(ctx.start_board, color)):
            king = ctx.leaf_board.king(not color)
            out.append(_claim("unstoppable_passer", color,
                              f"{_label(not color)}'s king cannot catch {_label(color)}'s passed pawn on "
                              f"{chess.square_name(square)} (rule of the square).", "PAWN_RACE", RACE_CP,
                              board=ctx.leaf_board, squares=[square, *([king] if king is not None else [])],
                              roles={square: "subject", **({king: "king"} if king is not None else {})},
                              entity=f"race:{chess.FILE_NAMES[chess.square_file(square)]}"))
    return out


BOARD_RULES = [rule_trapped_piece, rule_king_shelter, rule_pawn_plans, rule_pawn_race]
