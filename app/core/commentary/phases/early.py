"""Early-game (in-book) commenter.

Book plies carry no engine data by design. Comments here are deterministic
templates driven purely by ECO book metadata: one short line whenever the
matched opening *name or variation changes*, plus a closing line on the final
book ply. The richer "out of book, here are the plans" comment belongs to the
first non-book ply and is produced by the LLM path (opening_transition key
moment) — not here.

Evals are never mentioned: there are none.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.core.commentary.openings.eco_book import ECOBook, OpeningInfo
from app.models.chess_events import AnalyzedMoveData


def _label(info: OpeningInfo) -> str:
    if info.variation:
        return f"{info.name}, {info.variation} ({info.code})"
    return f"{info.name} ({info.code})"


# The ECO data mixes synonymous family names; normalize the worst offenders so a
# rename alone does not read as a new opening.
_FAMILY_ALIASES = {
    "spanish": "ruy lopez",
    "spanish game": "ruy lopez",
    "king's pawn game": "king's pawn",
    "queen's pawn game": "queen's pawn",
}


def _family(info: Optional[OpeningInfo]) -> str:
    """Normalized opening family: base name before any ':' qualifier."""
    if info is None or not info.name:
        return ""
    base = info.name.split(":", 1)[0].strip().lower()
    return _FAMILY_ALIASES.get(base, base)


class EarlyGameCommenter:
    """Deterministic opening comments for in-book plies."""

    def __init__(self, eco_book: Optional[ECOBook] = None) -> None:
        self._eco = eco_book or ECOBook()

    def comments_for_book_plies(
        self, analyzed_rows: List[AnalyzedMoveData]
    ) -> Dict[int, str]:
        """Map move_index -> opening comment, for in-book rows only."""
        out: Dict[int, str] = {}
        prev_family = ""
        uci_prefix: List[str] = []
        book_rows: List[Tuple[int, AnalyzedMoveData, Optional[OpeningInfo]]] = []

        for row in analyzed_rows:
            uci_prefix.append(row.uci)
            if (row.phase_raw or "") != "early":
                continue
            info, matched = self._eco.match(list(uci_prefix))
            if info is None or matched < len(uci_prefix):
                continue
            book_rows.append((row.index, row, info))

        for pos, (idx, row, info) in enumerate(book_rows):
            family = _family(info)
            is_first = pos == 0
            is_last = pos == len(book_rows) - 1
            changed = family != prev_family
            prev_family = family

            parts: List[str] = []
            if is_first:
                parts.append(f"The game opens as a {_label(info)}.")
            elif changed:
                parts.append(f"The game transposes into the {_label(info)}.")
            if is_last:
                if parts:
                    parts.append("This is the last book move.")
                else:
                    parts.append(
                        f"This is the last book move of the {_label(info)}; "
                        "both sides are on their own from here."
                    )
            if parts:
                out[idx] = " ".join(parts)
        return out


def attach_opening_comments(analyzed_rows: List[AnalyzedMoveData], eco_book: ECOBook) -> None:
    """Write opening comments into hiddenFeatures['_opening']['comment'] on book rows."""
    commenter = EarlyGameCommenter(eco_book)
    comments = commenter.comments_for_book_plies(analyzed_rows)
    for row in analyzed_rows:
        text = comments.get(row.index)
        if not text:
            continue
        analyzed_move = row.analyzed_move
        if analyzed_move is None or not isinstance(analyzed_move.hiddenFeatures, dict):
            continue
        analyzed_move.hiddenFeatures.setdefault("_opening", {})
        analyzed_move.hiddenFeatures["_opening"]["comment"] = text
