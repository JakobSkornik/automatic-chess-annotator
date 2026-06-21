"""Concrete, whole-unit material-imbalance phrasing (Guid: never fractional pawns)."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from .context import _benef, _side_label

# Material is described in concrete, whole-unit terms (Guid): never fractional
# pawns. piece = bishop/knight only; exchange = rook vs. a minor.
_MAT_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}
_MAT_VAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
_MAT_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
_NUMWORD = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five"}


def _imbalance(board: chess.Board) -> dict[int, int]:
    """White-minus-Black piece count per type."""
    return {
        pt: len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        for pt in _MAT_ORDER
    }


def _enumerate_extra(extra: dict[int, int]) -> str:
    """e.g. {ROOK:1, BISHOP:1} -> 'a rook and a bishop'; {KNIGHT:2} -> 'two knights'."""
    parts: list[str] = []
    for pt in _MAT_ORDER:
        c = extra.get(pt, 0)
        if c <= 0:
            continue
        name = _MAT_NAME[pt]
        parts.append(f"a {name}" if c == 1 else f"{_NUMWORD.get(c, str(c))} {name}s")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _pawns_phrase(n: int) -> str:
    return "a pawn" if n == 1 else f"{_NUMWORD.get(n, str(n))} pawns"


@dataclass
class _MatStanding:
    """The material imbalance from the leading side's ("subject") perspective."""

    label: str
    benef: str
    delta_cp: int
    subj: dict[int, int]
    opp: dict[int, int]
    nonpawn_subj: dict[int, int]
    nonpawn_opp: dict[int, int]
    subj_pawns: int
    opp_pawns: int
    changed: bool


def _material_standing(leaf_board: chess.Board, changed: bool) -> _MatStanding | None:
    imb = _imbalance(leaf_board)
    if all(v == 0 for v in imb.values()):
        return None
    white_extra = {pt: d for pt, d in imb.items() if d > 0}
    black_extra = {pt: -d for pt, d in imb.items() if d < 0}
    val = sum(d * _MAT_VAL[pt] for pt, d in imb.items())
    white_subject = (
        val > 0 if val != 0 else sum(white_extra.values()) >= sum(black_extra.values())
    )
    side = "WHITE" if white_subject else "BLACK"
    subj = white_extra if white_subject else black_extra
    opp = black_extra if white_subject else white_extra
    return _MatStanding(
        label=_side_label(side),
        benef=_benef(side),
        delta_cp=max(abs(val) * 100, 100),
        subj=subj,
        opp=opp,
        nonpawn_subj={pt: c for pt, c in subj.items() if pt != chess.PAWN},
        nonpawn_opp={pt: c for pt, c in opp.items() if pt != chess.PAWN},
        subj_pawns=subj.get(chess.PAWN, 0),
        opp_pawns=opp.get(chess.PAWN, 0),
        changed=changed,
    )


def _single_minor(d: dict[int, int]) -> str | None:
    if sum(d.values()) != 1:
        return None
    if d.get(chess.BISHOP) == 1:
        return "bishop"
    if d.get(chess.KNIGHT) == 1:
        return "knight"
    return None


def _phrase_pure_pawns(s: _MatStanding) -> tuple[str, str, int] | None:
    if s.nonpawn_subj or s.nonpawn_opp:
        return None
    text = (
        f"{s.label} has won {_pawns_phrase(s.subj_pawns)}."
        if s.changed
        else f"{s.label} is {_pawns_phrase(s.subj_pawns)} up."
    )
    return text, s.benef, max(s.subj_pawns * 100, 100)


def _phrase_exchange(s: _MatStanding) -> tuple[str, str, int] | None:
    is_exchange = (
        s.nonpawn_subj.get(chess.ROOK) == 1
        and len(s.nonpawn_subj) == 1
        and _single_minor(s.nonpawn_opp)
    )
    if not is_exchange:
        return None
    if s.opp_pawns:
        return (
            f"{s.label} has won an exchange for {_pawns_phrase(s.opp_pawns)}.",
            s.benef,
            s.delta_cp,
        )
    text = (
        f"{s.label} has won the exchange."
        if s.changed
        else f"{s.label} is up the exchange."
    )
    return text, s.benef, s.delta_cp


def _phrase_single_minor(s: _MatStanding) -> tuple[str, str, int] | None:
    minor = _single_minor(s.nonpawn_subj)
    if not minor or s.nonpawn_opp:
        return None
    if s.opp_pawns:
        return (
            f"{s.label} has won a {minor} for {_pawns_phrase(s.opp_pawns)}.",
            s.benef,
            s.delta_cp,
        )
    text = (
        f"{s.label} has won a {minor}." if s.changed else f"{s.label} is up a {minor}."
    )
    return text, s.benef, s.delta_cp


def _phrase_single_rook(s: _MatStanding) -> tuple[str, str, int] | None:
    is_lone_rook = (
        s.nonpawn_subj.get(chess.ROOK) == 1
        and len(s.nonpawn_subj) == 1
        and not s.nonpawn_opp
    )
    if not is_lone_rook:
        return None
    if s.opp_pawns:
        return (
            f"{s.label} has won a rook for {_pawns_phrase(s.opp_pawns)}.",
            s.benef,
            s.delta_cp,
        )
    text = f"{s.label} has won a rook." if s.changed else f"{s.label} is up a rook."
    return text, s.benef, s.delta_cp


def _phrase_general(s: _MatStanding) -> tuple[str, str, int]:
    """Catch-all: '<subj> for/against <opp>' (against a lone queen)."""
    subj_desc = _enumerate_extra(s.subj)
    opp_desc = _enumerate_extra(s.opp)
    if opp_desc:
        connector = "against" if set(s.nonpawn_opp) == {chess.QUEEN} else "for"
        return f"{s.label} has {subj_desc} {connector} {opp_desc}.", s.benef, s.delta_cp
    verb = "has won" if s.changed else "has"
    return f"{s.label} {verb} {subj_desc}.", s.benef, s.delta_cp


_MATERIAL_PHRASERS = (
    _phrase_pure_pawns,
    _phrase_exchange,
    _phrase_single_minor,
    _phrase_single_rook,
)


def describe_material(
    leaf_board: chess.Board, *, changed: bool
) -> tuple[str, str, int] | None:
    """(text, beneficiary, delta_cp) for the material standing, in concrete whole
    units. ``changed`` -> 'has won ...'; else the static 'is ... up' / 'has ...'."""
    standing = _material_standing(leaf_board, changed)
    if standing is None:
        return None
    for phraser in _MATERIAL_PHRASERS:
        result = phraser(standing)
        if result is not None:
            return result
    return _phrase_general(standing)
