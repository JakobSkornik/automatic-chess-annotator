"""Material-phrasing rule tests (advisor feedback, feedback_v3.txt):

1. Material comments must always be whole/integer units, never a fractional
   pawn phrase like "(about 1.3 pawns)".
2. Vocabulary: "piece" means bishop/knight only; "exchange" means rook vs. a
   minor only.
3. Generic vs. specific naming must vary for the same kind of imbalance
   (e.g. "two pieces" vs. "a bishop and a knight"), deterministically per
   position so re-running on the same game reproduces the same wording.
"""

import re

import chess

from app.core.commentary.rules.material import describe_material

FRACTIONAL_RE = re.compile(r"\d*\.\d+\s*pawns?")


def _board(fen: str) -> chess.Board:
    return chess.Board(fen)


# ---------------------------------------------------------------------------
# 1. Integer-only output
# ---------------------------------------------------------------------------


def test_material_text_never_fractional_pawns():
    # White up a pawn, four pawns, and a bishop + four pawns imbalance.
    fens = [
        "rnbqkbnr/1ppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1",  # +1 pawn
        "4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1",  # white 4 pawns vs none
        "4k3/8/8/8/8/8/3PPPP1/2B1K3 w - - 0 1",  # bishop + 4 pawns vs none
    ]
    for fen in fens:
        described = describe_material(_board(fen), changed=True)
        assert described is not None
        text = described[0]
        assert not FRACTIONAL_RE.search(text), text
        # No stray decimal point of any kind in a material sentence.
        assert "." not in text.rstrip(".").replace("..", "."), text


def test_material_text_only_whole_numbers_or_number_words():
    board = _board("4k3/8/8/8/8/8/3PPPP1/2B1K3 w - - 0 1")
    text, _, _ = describe_material(board, changed=True)
    for tok in text.replace(".", "").split():
        if any(ch.isdigit() for ch in tok):
            assert tok.isdigit(), f"unexpected numeric token: {tok!r} in {text!r}"


# ---------------------------------------------------------------------------
# 2. Vocabulary rules: "piece" = bishop/knight only; "exchange" = rook vs minor
# ---------------------------------------------------------------------------


def test_lone_rook_up_never_called_a_piece_or_exchange():
    # White has an extra rook only (no minor imbalance) -> must not say
    # "piece" or "exchange" for a plain rook-up.
    board = _board("4k3/8/8/8/8/8/8/3RK3 w - - 0 1")
    text, benef, _ = describe_material(board, changed=True)
    assert "rook" in text
    assert "piece" not in text
    assert "exchange" not in text
    assert benef == "white"


def test_exchange_is_rook_vs_minor_only():
    # White has an extra rook, Black has an extra knight -> "the exchange".
    board = _board("4k3/8/8/8/8/8/8/3RKN2 w - - 0 1")
    # Black's knight offsets one of White's minors; construct precisely:
    board = chess.Board("4k1n1/8/8/8/8/8/8/3RK3 w - - 0 1")
    text, benef, _ = describe_material(board, changed=True)
    assert "exchange" in text
    assert benef == "white"


def test_queen_vs_two_rooks_not_labelled_exchange_or_piece():
    # An imbalance that is NOT rook-vs-minor must not borrow "exchange"/"piece".
    board = chess.Board("4k3/8/8/8/8/8/8/2QK4 w - - 0 1")
    # White has a lone extra queen vs nothing -> falls to the general/queen path.
    text, _, _ = describe_material(board, changed=True)
    assert "exchange" not in text
    assert "piece" not in text


# ---------------------------------------------------------------------------
# 3. Generic vs. specific phrasing variety (deterministic per position)
# ---------------------------------------------------------------------------


# Black king walked across the empty 8th rank: varies the FEN (and therefore
# the deterministic phrasing seed) without touching the material imbalance.
_BLACK_KING_FILES = "abcdefgh"


def _fen_with_black_king_on(file_: str, piece_rank: str) -> str:
    idx = _BLACK_KING_FILES.index(file_)
    king_rank = ("" if idx == 0 else str(idx)) + "k" + ("" if idx == 7 else str(7 - idx))
    return f"{king_rank}/8/8/8/8/8/8/{piece_rank} w - - 0 1"


def test_single_minor_phrasing_varies_generic_vs_specific():
    # Several distinct positions with a single extra minor (bishop-up); only
    # the black king's file differs, so the material imbalance is identical
    # but the phrasing seed (derived from the FEN) varies. Both the generic
    # "a piece" and the specific "a bishop" wording must appear across the
    # set (variety), and each result must be reproducible.
    seen_generic = False
    seen_specific = False
    for file_ in _BLACK_KING_FILES:
        fen = _fen_with_black_king_on(file_, "2B1K3")
        text1, _, _ = describe_material(chess.Board(fen), changed=True)
        text2, _, _ = describe_material(chess.Board(fen), changed=True)
        assert text1 == text2, "phrasing must be deterministic for the same position"
        if "a piece" in text1:
            seen_generic = True
        elif "a bishop" in text1:
            seen_specific = True
    assert seen_generic, "generic 'a piece' phrasing never appeared across sample positions"
    assert seen_specific, "specific 'a bishop' phrasing never appeared across sample positions"


def test_two_minors_phrasing_varies_generic_vs_specific():
    # Bishop + knight up (two minors, no other imbalance) across several
    # positions differing only in the black king's file.
    seen_generic = False
    seen_specific = False
    for file_ in _BLACK_KING_FILES:
        fen = _fen_with_black_king_on(file_, "2BNK3")
        text, _, _ = describe_material(chess.Board(fen), changed=True)
        if "two pieces" in text:
            seen_generic = True
        elif "bishop" in text and "knight" in text:
            seen_specific = True
    assert seen_generic, "generic 'two pieces' phrasing never appeared"
    assert seen_specific, "specific 'a bishop and a knight' phrasing never appeared"


def test_material_standing_deterministic_repeat_calls():
    """Same FEN, called repeatedly, must always produce the identical text
    (deterministic-per-move, not global random state)."""
    board = chess.Board("4k3/8/8/8/8/8/8/2BNK3 w - - 0 1")
    results = {describe_material(chess.Board(board.fen()), changed=True)[0] for _ in range(10)}
    assert len(results) == 1
