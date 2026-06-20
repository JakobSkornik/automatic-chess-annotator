"""Annotated PGN export from a finished ``GameJson``.

Renders comments, evals (``[%eval]``), NAGs, the displayed continuation as a
variation after the move, and the better alternative as a variation at the
same point — readable by Lichess/ChessBase/any PGN viewer. Interactive
``[type:content]`` tokens are flattened to plain text; ``[pv:...]`` tokens are
lifted out of the text and become the variations.
"""

from __future__ import annotations

import io
import logging
import re

import chess
import chess.pgn

from app.models.GameJson import GameJson, GameMove

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\[(\w+):([^\]]+)\]")

# NAG numbers: $1 !, $2 ?, $3 !!, $4 ??, $5 !?, $6 ?!
_CLASSIFICATION_NAGS = {
    "brilliant": 3,
    "great_move": 1,
}


def _flatten_tokens(text: str) -> tuple[str, list[list[str]]]:
    """Strip interactive tokens; return (plain_text, pv_lines_as_san_lists)."""
    pv_lines: list[list[str]] = []

    # "... after [pv:x y z] (+0.12, ...)" -> drop the connective with the token;
    # the line itself becomes a PGN variation instead.
    text = re.sub(r"\s*\bafter\s+(?=\[pv:)", " ", text or "")

    def repl(m: re.Match) -> str:
        ttype = m.group(1).lower()
        content = m.group(2).strip()
        if ttype == "pv":
            sans = [s for s in content.split() if s]
            if sans:
                pv_lines.append(sans)
            return ""
        if ttype == "file":
            return f"the {content}-file"
        # move / square / piece / eval: keep the raw content
        return content

    plain = _TOKEN_RE.sub(repl, text)
    plain = re.sub(r"\s+", " ", plain)
    plain = re.sub(r"\s+([.,;:!?])", r"\1", plain)
    return plain.strip(), pv_lines


# Positional-assessment NAGs (ChessBase glyphs) from the White-POV evaluation:
#   $10 =  ·  $14 ⩲  ·  $15 ⩱  ·  $16 ±  ·  $17 ∓  ·  $18 +−  ·  $19 −+
# Thresholds in centipawns (White POV); |cp| < 25 reads as "equal".
def _assessment_nag(cp: int | None, mate: int | None) -> int | None:
    if mate is not None:
        return 18 if mate > 0 else 19
    if cp is None:
        return None
    if cp >= 150:
        return 18
    if cp >= 75:
        return 16
    if cp >= 25:
        return 14
    if cp > -25:
        return 10
    if cp > -75:
        return 15
    if cp > -150:
        return 17
    return 19


def _eval_str(cp: int | None, mate: int | None, depth: int | None) -> str:
    """ChessBase-style ``[%eval x.xx,depth]`` (or ``#mate``) tag, or ""."""
    suffix = f",{depth}" if depth is not None else ""
    if mate is not None:
        return f"[%eval #{mate}{suffix}]"
    if cp is not None:
        return f"[%eval {cp / 100:.2f}{suffix}]"
    return ""


def _eval_tag(move: GameMove, depth: int | None) -> str:
    sc = move.score
    if sc is None:
        return ""
    return _eval_str(sc.cp, sc.mate, depth)


def _facts_eval(
    move: GameMove, default_depth: int | None
) -> tuple[int | None, int | None, int | None]:
    """(cp, mate, depth) for the played-move continuation's optimal-play leaf."""
    cf = move.comment_facts or {}
    cp = cf.get("eval_cp")
    mate = cf.get("eval_mate")
    depth = cf.get("depth", default_depth)
    if cp is None and mate is None and move.score is not None:
        return move.score.cp, move.score.mate, default_depth
    return cp, mate, depth


def _alt_eval(
    move: GameMove, first_san: str, default_depth: int | None
) -> tuple[int | None, int | None, int | None]:
    """(cp, mate, depth) for a better-alternative line, matched by its first move."""
    cf = move.comment_facts or {}
    alt = cf.get("better_alternative")
    if alt and alt.get("san") == first_san:
        return alt.get("eval_cp"), alt.get("eval_mate"), cf.get("depth", default_depth)
    return None, None, None


def _nag_for_move(move: GameMove) -> int | None:
    if move.classification in _CLASSIFICATION_NAGS:
        return _CLASSIFICATION_NAGS[move.classification]
    return None


def _try_add_line(
    start_board: chess.Board,
    node: chess.pgn.GameNode,
    sans: list[str],
    leaf_cp: int | None = None,
    leaf_mate: int | None = None,
    leaf_depth: int | None = None,
) -> None:
    """Attach a SAN line as a variation from ``node`` (position = start_board).

    When a leaf eval is supplied, tag the variation's final node with its
    ChessBase assessment NAG and ``[%eval]`` — the position evaluation shown at
    the end of the variation.
    """
    board = start_board.copy(stack=False)
    cursor = node
    advanced = False
    for i, san in enumerate(sans):
        try:
            mv = board.parse_san(san)
        except Exception:
            break
        if i == 0:
            if cursor.has_variation(mv):
                cursor = cursor.variation(mv)
                board.push(mv)
                advanced = True
                continue
            cursor = cursor.add_variation(mv)
        else:
            cursor = (
                cursor.add_variation(mv)
                if not cursor.has_variation(mv)
                else cursor.variation(mv)
            )
        board.push(mv)
        advanced = True
    if advanced and (leaf_cp is not None or leaf_mate is not None):
        nag = _assessment_nag(leaf_cp, leaf_mate)
        if nag:
            cursor.nags.add(nag)
        tag = _eval_str(leaf_cp, leaf_mate, leaf_depth)
        if tag:
            cursor.comment = (
                f"{cursor.comment} {tag}".strip() if cursor.comment else tag
            )


def game_json_to_pgn(gj: GameJson) -> str:
    game = chess.pgn.Game()
    md = gj.metadata
    game.headers["Event"] = md.eventId or "?"
    game.headers["White"] = md.white or "?"
    game.headers["Black"] = md.black or "?"
    game.headers["Result"] = md.result or "*"
    if md.whiteElo:
        game.headers["WhiteElo"] = str(md.whiteElo)
    if md.blackElo:
        game.headers["BlackElo"] = str(md.blackElo)
    if md.opening:
        game.headers["Opening"] = md.opening
    game.headers["Annotator"] = "automatic-chess-annotator"

    default_depth = gj.analysis_info.depth if gj.analysis_info else None

    board = chess.Board()
    node: chess.pgn.GameNode = game
    # Continuation lines from comments that so far match the actual game: they
    # only become variations at the ply where they diverge (otherwise they would
    # duplicate the mainline, e.g. "4. Nf3 (4. Nf3 e6 ...)"). Each entry carries
    # the line's leaf eval (cp, mate, depth) so its end can be annotated.
    LeafEval = tuple[int | None, int | None, int | None]
    pending_continuations: list[tuple[list[str], LeafEval]] = []
    for move in gj.moves:
        try:
            mv = chess.Move.from_uci(move.uci)
        except Exception:
            logger.warning("PGN export: bad uci %s — stopping", move.uci)
            break
        if mv not in board.legal_moves:
            logger.warning(
                "PGN export: illegal %s at %s — stopping", move.uci, board.fen()
            )
            break
        board_before = board.copy(stack=False)
        parent = node

        still_matching: list[tuple[list[str], LeafEval]] = []
        for cont, leaf in pending_continuations:
            if cont and cont[0] == move.san:
                rest = cont[1:]
                if rest:
                    still_matching.append((rest, leaf))
                # fully played out in the game -> nothing to add
            elif len(cont) >= 2:
                # Diverges here: a true alternative to this move.
                _try_add_line(board_before, parent, cont, *leaf)
        pending_continuations = still_matching

        node = node.add_main_variation(mv)
        board.push(mv)

        nag = _nag_for_move(move)
        if nag:
            node.nags.add(nag)
        # Positional-assessment glyph (=, ⩲, ±, +−, …) from this move's eval.
        if move.score is not None:
            assess = _assessment_nag(move.score.cp, move.score.mate)
            if assess:
                node.nags.add(assess)

        comment_bits: list[str] = []
        move_depth = (move.comment_facts or {}).get("depth", default_depth)
        ev = _eval_tag(move, move_depth)
        if ev:
            comment_bits.append(ev)

        move_comment = move.comment
        if move_comment:
            plain, pv_lines = _flatten_tokens(move_comment)
            if plain:
                comment_bits.append(plain)
            for sans in pv_lines:
                if not sans:
                    continue
                if sans[0] == move.san:
                    # The displayed continuation (starts with the played move):
                    # defer it — it becomes a variation only where it diverges
                    # from the game (avoids duplicating the next mainline move).
                    # Its leaf eval is the move's own optimal-play evaluation.
                    if len(sans) > 1:
                        pending_continuations.append(
                            (sans[1:], _facts_eval(move, default_depth))
                        )
                else:
                    # An alternative to the played move (e.g. "Better was ..."):
                    # a true variation at the same point, ending at its own eval.
                    _try_add_line(
                        board_before,
                        parent,
                        sans,
                        *_alt_eval(move, sans[0], default_depth),
                    )

        if comment_bits:
            node.comment = " ".join(comment_bits).strip()

    # Continuations still matching at the end of the game extend past it.
    for cont, leaf in pending_continuations:
        if len(cont) >= 2:
            _try_add_line(board, node, cont, *leaf)

    out = io.StringIO()
    exporter = chess.pgn.FileExporter(out)
    game.accept(exporter)
    return out.getvalue()
