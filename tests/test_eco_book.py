"""ECO longest-prefix match and FEN fallback."""

import io

import chess
import chess.pgn

from app.core.commentary.openings.eco_book import (
    ECOBook,
    detect_opening,
    is_absent_opening_header,
    merge_opening_with_headers,
    parse_pgn_eco_tag,
)


def test_longest_prefix_finds_deeper_line():
    book = ECOBook()
    # Amar Opening: 1. Nh3 — should match at least this
    uci = ["g1h3"]
    info, n = book.match(uci)
    assert info is not None
    assert info.code == "A00"
    assert n == 1


def test_full_line_prefix_longer_than_single_move():
    book = ECOBook()
    uci = ["g1h3", "d7d5", "g2g3", "e7e5", "f2f4", "c8h3", "f1h3", "e5f4"]
    info, n = book.match(uci)
    assert info is not None
    assert info.code == "A00"
    assert n >= 8


def test_detect_opening_unknown_header_game():
    pgn = """[Event "Test"]
[White "A"]
[Black "B"]
[Opening "Unknown"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 *
"""
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    info, ply = detect_opening(game)
    assert info is not None
    assert info.code.startswith("C")
    assert ply >= 1


def test_parse_eco_valid():
    g = chess.pgn.Game()
    g.headers["ECO"] = "B12"
    assert parse_pgn_eco_tag(g.headers) == "B12"


def test_parse_eco_invalid():
    g = chess.pgn.Game()
    g.headers["ECO"] = "Z99"
    assert parse_pgn_eco_tag(g.headers) is None


def test_merge_prefers_header_name_when_family_matches():
    type("I", (), {"code": "B12", "name": "Caro-Kann Defense", "variation": None})()
    # merge expects OpeningInfo - use real OpeningInfo
    from app.core.commentary.openings.eco_book import OpeningInfo

    d = OpeningInfo("B12", "Caro-Kann Defense")
    m = merge_opening_with_headers(d, "Caro-Kann: Advance Variation", "B12")
    assert m.name == "Caro-Kann: Advance Variation"
    assert m.code == "B12"


def test_is_absent_opening():
    assert is_absent_opening_header("Unknown")
    assert is_absent_opening_header("")
    assert not is_absent_opening_header("Ruy Lopez")
