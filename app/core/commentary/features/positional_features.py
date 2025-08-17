from __future__ import annotations

from typing import Dict, List, Set

import chess


FILES = [
    [chess.square(file_idx, rank) for rank in range(8)] for file_idx in range(8)
]


def _file_index(square: int) -> int:
    return chess.square_file(square)


def _is_light_square(square: int) -> bool:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return (file_idx + rank_idx) % 2 == 0


def _pawn_files(board: chess.Board, color: chess.Color) -> Dict[int, List[int]]:
    files: Dict[int, List[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        fi = _file_index(sq)
        files.setdefault(fi, []).append(sq)
    return files


def _is_passed_pawn(board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
    pawn_file = chess.square_file(pawn_sq)
    enemy = not color
    # squares ahead for enemy pawns relative to our direction
    ranks = range(chess.square_rank(pawn_sq) + 1, 8) if color == chess.WHITE else range(chess.square_rank(pawn_sq) - 1, -1, -1)
    for rank in ranks:
        for f in (pawn_file - 1, pawn_file, pawn_file + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, rank)
                if sq in board.pieces(chess.PAWN, enemy):
                    return False
    return True


def _open_and_semi_open_files(board: chess.Board) -> Dict[str, List[int]]:
    open_files: List[int] = []
    semi_open_white: List[int] = []
    semi_open_black: List[int] = []
    for f in range(8):
        has_white_pawn = any(sq in board.pieces(chess.PAWN, chess.WHITE) for sq in FILES[f])
        has_black_pawn = any(sq in board.pieces(chess.PAWN, chess.BLACK) for sq in FILES[f])
        if not has_white_pawn and not has_black_pawn:
            open_files.append(f)
        else:
            if not has_white_pawn and has_black_pawn:
                semi_open_white.append(f)
            if not has_black_pawn and has_white_pawn:
                semi_open_black.append(f)
    return {
        "open": open_files,
        "semiOpenWhite": semi_open_white,
        "semiOpenBlack": semi_open_black,
    }


def _side_squares(board: chess.Board, color: chess.Color) -> Set[int]:
    squares: Set[int] = set()
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        squares.update(board.pieces(pt, color))
    return squares


def _attacked_and_attacking(board: chess.Board, color: chess.Color) -> Dict[str, int]:
    enemy = not color
    own_sqs = _side_squares(board, color)
    enemy_sqs = _side_squares(board, enemy)

    # own pieces under attack
    attacked_pieces = sum(1 for sq in own_sqs if board.is_attacked_by(enemy, sq))

    # own pieces that attack any enemy piece
    attackers: Set[int] = set()
    for sq in enemy_sqs:
        for atk_sq in board.attackers(color, sq):
            attackers.add(atk_sq)
    return {
        "attackedPieces": attacked_pieces,
        "attackingPieces": len(attackers),
    }


def compute_hidden_features(board: chess.Board) -> Dict:
    """Compute a set of light-weight positional features for both sides.

    Returns a dict with keys: openFiles, white, black
    """
    features: Dict = {}

    # Open/semi-open files
    files_info = _open_and_semi_open_files(board)
    features["openFiles"] = files_info

    def _clear_between(a: int, b: int) -> bool:
        if a == b:
            return True
        fa, ra = chess.square_file(a), chess.square_rank(a)
        fb, rb = chess.square_file(b), chess.square_rank(b)
        df = (fb > fa) - (fb < fa)
        dr = (rb > ra) - (rb < ra)
        if df != 0 and dr != 0 and abs(fb - fa) != abs(rb - ra):
            return False
        f, r = fa + df, ra + dr
        while f != fb or r != rb:
            sq = chess.square(f, r)
            if board.piece_at(sq) is not None:
                return False
            f += df
            r += dr
        return True

    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        side: Dict = {}

        # Piece counts / pairs
        num_bishops = len(board.pieces(chess.BISHOP, color))
        num_knights = len(board.pieces(chess.KNIGHT, color))
        num_rooks = len(board.pieces(chess.ROOK, color))
        num_queens = len(board.pieces(chess.QUEEN, color))

        side["hasBishopPair"] = num_bishops >= 2
        side["hasKnightPair"] = num_knights >= 2
        side["hasRookPair"] = num_rooks >= 2
        side["hasQueen"] = num_queens >= 1

        # Bishop square colors counts
        bishops = list(board.pieces(chess.BISHOP, color))
        side["lightSquareBishops"] = sum(1 for b in bishops if _is_light_square(b))
        side["darkSquareBishops"] = sum(1 for b in bishops if not _is_light_square(b))

        # Castling rights (availability)
        side["canCastleKingSide"] = board.has_kingside_castling_rights(color)
        side["canCastleQueenSide"] = board.has_queenside_castling_rights(color)

        # Pawn structure: doubled, isolated, passed
        pawn_files = _pawn_files(board, color)
        side["doubledPawns"] = sum(1 for _, lst in pawn_files.items() if len(lst) > 1)

        isolated = 0
        passed_pawns = 0
        for f, lst in pawn_files.items():
            has_left = (f - 1) in pawn_files
            has_right = (f + 1) in pawn_files
            if not has_left and not has_right:
                isolated += len(lst)
            for p_sq in lst:
                if _is_passed_pawn(board, p_sq, color):
                    passed_pawns += 1
        side["isolatedPawns"] = isolated
        side["passedPawns"] = passed_pawns

        # Attacked/attacking counts
        side.update(_attacked_and_attacking(board, color))

        # Rook features: rooks on open/semi-open files and connected rooks
        rooks = list(board.pieces(chess.ROOK, color))
        if color == chess.WHITE:
            semi_open_for_side = set(files_info.get("semiOpenWhite", []))
        else:
            semi_open_for_side = set(files_info.get("semiOpenBlack", []))
        open_files_set = set(files_info.get("open", []))

        rooks_on_open = 0
        rooks_on_semi = 0
        for r_sq in rooks:
            f = chess.square_file(r_sq)
            if f in open_files_set:
                rooks_on_open += 1
            if f in semi_open_for_side:
                rooks_on_semi += 1
        side["rooksOnOpenFiles"] = rooks_on_open
        side["rooksOnSemiOpenFiles"] = rooks_on_semi

        connected = False
        if len(rooks) >= 2:
            for i in range(len(rooks)):
                for j in range(i + 1, len(rooks)):
                    a, b = rooks[i], rooks[j]
                    if chess.square_file(a) == chess.square_file(b) or chess.square_rank(a) == chess.square_rank(b):
                        if _clear_between(a, b):
                            connected = True
                            break
                if connected:
                    break
        side["connectedRooks"] = connected

        features[label] = side

    return features


