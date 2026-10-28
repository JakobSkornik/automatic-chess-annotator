"""Engine analysis orchestration: PGN -> per-move engine pass -> event
extraction -> CommentFacts -> assembled :class:`GameJson`.

The heavy lifting lives in the ``analysis`` subpackage; this module wires the
stages together and exposes the public entry points (``run_engine_analysis_to_json``,
``assemble_game_json``, ``EnginePipelineState``). Selected helpers are re-exported
here so historical import paths keep working.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import chess
from chess.pgn import Game

from app.core.commentary.event_extractor import ChessEventExtractor
from app.core.commentary.features.guid_features import (
    compute_feature_vector,
    vector_to_plain,
)
from app.core.commentary.openings.eco_book import (
    ECOBook,
    detect_opening,
    is_absent_opening_header,
    merge_opening_with_headers,
)
from app.core.commentary.phase_classifier import PhaseClassifier
from app.core.engine.engine_connector import EngineConnector
from app.core.io.pgn_reader import PGNReader
from app.models.chess_events import (
    AnalyzedMoveData,
    GameAnalysisContext,
    MoveEvent,
)
from app.models.GameJson import (
    AnalysisInfo,
    GameJson,
    GameMetadata,
    GameMove,
)
from app.models.Move import Move
from app.models.PgnMetadata import PgnMetadata

from .analysis.board_utils import _board_before_mainline_move, _score_to_move_score
from .analysis.comment_assembly import (
    _build_move_debug,
    _comment_status,
    _build_variations,
    _resolve_comment_tokens,
    _resolve_move_comment,
)
from .analysis.constants import DEFAULT_ANALYSIS_DEPTH, DEFAULT_PV_COUNT
from .analysis.engine_pass import AnalysisRetriever
from .analysis.key_moments import (
    _annotation_symbol,
    _apply_back_to_back_key_moment_suppression,
    _comment_selection,
    _promote_key_moment,
)
from .analysis.serialization import _build_feature_series, _facts_to_json

logger = logging.getLogger(__name__)

# Progress-callback milestones (percent).
ENGINE_PASS_PROGRESS_PCT = 90.0
EVENT_EXTRACT_PROGRESS_PCT = 92.0
ANALYSIS_DONE_PROGRESS_PCT = 95.0

# Re-exported for historical import paths (callers / tests import these from
# ``app.core.engine.analysis_retriever``).
__all__ = [
    "AnalysisRetriever",
    "EnginePipelineState",
    "_annotation_symbol",
    "_apply_back_to_back_key_moment_suppression",
    "_facts_to_json",
    "_promote_key_moment",
    "_pv_line_for_ai_payload",
    "assemble_game_json",
    "reverse_order_engine_analysis_enabled",
    "run_engine_analysis_to_json",
]


@dataclass
class EnginePipelineState:
    """Mutable state shared between engine phase and LLM commentary phase."""

    game: Game
    retriever: AnalysisRetriever
    moves_list: list[Move]
    analyzed_rows: list[AnalyzedMoveData]
    move_events: list[MoveEvent]
    context: GameAnalysisContext
    metadata: GameMetadata
    # Set by GameAnnotationPipeline when the commentary sweep finished.
    llm_done: bool = False


def _build_game_move(
    state: EnginePipelineState,
    idx: int,
    row: AnalyzedMoveData,
    seen_transitions: dict[str, int] | None = None,
) -> GameMove:
    """Assemble one GameMove (comment + facts + variations + debug) for the UI."""
    game, moves_list, move_events = state.game, state.moves_list, state.move_events
    move_obj = moves_list[idx]
    analyzed_move = row.analyzed_move
    pvs = row.pvs if isinstance(row.pvs, list) else []

    move_event = move_events[idx] if idx < len(move_events) else None
    key_moment = (move_event.key_moment_type if move_event else None) or getattr(
        row, "key_moment_type", None
    )

    # Commentary side gate: when a side is selected, the other side's moves get
    # NO commentary apparatus at all (no comment, facts, dot, charts).
    side_sel = (state.metadata.comment_side or "both").lower()
    mover_is_white = chess.Board(row.fen_before).turn == chess.WHITE
    side_ok = side_sel == "both" or (side_sel == "white") == mover_is_white
    selection = _comment_selection(move_event) if move_event else {"allowed": False}
    prose_ok = side_ok and selection["allowed"]
    if move_event and move_event.comment_facts:
        move_event.comment_facts.selection_metadata["comment_selection"] = selection

    comment = _resolve_move_comment(
        row,
        analyzed_move,
        move_event,
        key_moment,
        side_ok=prose_ok,
        seen_transitions=seen_transitions,
    ).comment
    # Opening book labels are metadata rather than substantive move comments.
    if not side_ok or (not prose_ok and (row.phase_raw or "") != "early"):
        comment = None

    board_before = _board_before_mainline_move(game, idx)
    try:
        san_main = board_before.san(chess.Move.from_uci(move_obj.move))
    except Exception:
        san_main = move_obj.move

    facts = move_event.comment_facts if (move_event and side_ok) else None
    status = _comment_status(analyzed_move, move_event, comment, row.phase_raw or "",
                             eligible=bool(prose_ok and move_event and move_event.key_moment_type),
                             llm_done=state.llm_done)
    debug = _build_move_debug(analyzed_move, move_event, facts, comment, row.phase_raw or "")
    if debug is not None:
        debug.update(status)
    return GameMove(
        mn=board_before.fullmove_number,
        color="w" if board_before.turn == chess.WHITE else "b",
        san=san_main,
        uci=move_obj.move,
        fen=move_obj.position,
        phase=str(row.phase_raw or analyzed_move.phase or "mid"),
        # Mate(0) loses the winner when serialized as an integer. A finished
        # game has a result, rather than an ongoing position evaluation.
        score=None if chess.Board(move_obj.position).is_game_over() else _score_to_move_score(analyzed_move.score),
        variations=_build_variations(game, idx, pvs),
        comment=comment,
        **status,
        classification=key_moment,
        annotation=_annotation_symbol(key_moment) if side_ok else None,
        is_key_moment=bool(move_event and move_event.key_moment_type and prose_ok),
        final_comment=bool(
            comment
            and move_event
            and move_event.key_moment_type
            and prose_ok
            and not move_event.brief_commentary
        ),
        resolved_tokens=_resolve_comment_tokens(comment, move_event),
        comment_facts=_facts_to_json(facts) if facts is not None else None,
        debug=debug,
    )


def assemble_game_json(state: EnginePipelineState) -> GameJson:
    """Build GameJson from pipeline state (engine + optional LLM fields)."""
    seen_transitions: dict[str, int] = {}
    game_moves = [
        _build_game_move(state, idx, row, seen_transitions)
        for idx, row in enumerate(state.analyzed_rows)
    ]
    return GameJson(
        metadata=state.metadata,
        moves=game_moves,
        feature_series=_build_feature_series(state.analyzed_rows),
        commentary_complete=state.llm_done,
        analysis_info=AnalysisInfo(
            engine="Stockfish",
            depth=DEFAULT_ANALYSIS_DEPTH,
            multipv=DEFAULT_PV_COUNT,
            timestamp=time.time(),
        ),
    )


# Structural facts describe a lasting state (a lost bishop pair, a weak back
# rank), so they are stated once and muted over a wide window when best play
# keeps recreating them — unlike one-off event claims.
_STRUCTURAL_RULE_IDS = frozenset(
    {
        "bishop_pair_eliminated",
        "back_rank_weakness",
        "bad_bishop_created",
        "bad_bishop_solved",
        "strong_knight_established",
        "strong_knight_lost",
        "rook_reaches_seventh",
        "rooks_connected",
        "passed_pawn_created",
        "outside_passer",
        # A standing material edge (e.g. "White is a rook up") describes a
        # lasting state, not a one-off event: without the wide window it would
        # resurface on every quiet move in a winning endgame.
        "material_standing",
    }
)


def _parse_single_game(pgn_string: str) -> Game:
    PGNReader.validate_single_game(pgn_string)
    game = PGNReader().read_game_from_string(pgn_string)
    if not game:
        raise ValueError("Invalid PGN")
    return game


def _resolve_opening_metadata(
    game: Game, headers: PgnMetadata, eco_book: ECOBook
) -> tuple[str | None, str | None]:
    """Merge the detected opening with the PGN header opening/ECO."""
    detected_opening, _det_ply = detect_opening(game, eco_book)
    header_opening = (
        "" if is_absent_opening_header(headers.opening) else headers.opening.strip()
    )
    merged = merge_opening_with_headers(detected_opening, header_opening, headers.eco)
    name = merged.name if merged else (header_opening or None)
    return name, (merged.code if merged else headers.eco)


def _build_metadata(
    headers: PgnMetadata,
    game: Game,
    opening_name: str | None,
    opening_eco: str | None,
    metadata_id: str | None,
) -> GameMetadata:
    return GameMetadata(
        id=metadata_id or str(uuid.uuid4()),
        white=headers.whiteName,
        black=headers.blackName,
        result=headers.result,
        initial_fen=game.board().fen(),
        date=game.headers.get("Date"),
        eventId=headers.event,
        site=headers.site,
        round=headers.round,
        whiteElo=headers.whiteElo,
        blackElo=headers.blackElo,
        opening=opening_name,
        opening_eco=opening_eco,
    )


def _attach_guid_vector(analyzed_move: Move, board_after: chess.Board) -> None:
    """Attach the engine-free Guid feature vector (feeds charts + rule engine)."""
    try:
        guid_vec = compute_feature_vector(board_after)
        if isinstance(analyzed_move.hiddenFeatures, dict):
            analyzed_move.hiddenFeatures["_guid"] = vector_to_plain(guid_vec)
    except Exception as e:
        logger.error("Guid feature vector failed at ply %s: %s", analyzed_move.depth, e)


def _attach_score_meta(analyzed_move: Move, previous_move_obj: Move | None) -> None:
    """Record prev/now eval and their delta under hiddenFeatures._ai."""
    try:
        if analyzed_move.hiddenFeatures is None:
            analyzed_move.hiddenFeatures = {}
        if isinstance(analyzed_move.hiddenFeatures, dict):
            ai_meta = analyzed_move.hiddenFeatures.setdefault("_ai", {})
            ai_meta["prevScore"] = (
                previous_move_obj.score if previous_move_obj else None
            )
            ai_meta["scoreNow"] = analyzed_move.score
            if ai_meta["prevScore"] is not None and ai_meta["scoreNow"] is not None:
                ai_meta["scoreDelta"] = ai_meta["scoreNow"] - ai_meta["prevScore"]
    except Exception:
        pass


def _build_analyzed_row(
    idx: int,
    move_obj: Move,
    san_main: str,
    fen_before: str,
    analyzed_move: Move,
    pvs: list,
) -> AnalyzedMoveData:
    engine_meta = {}
    if isinstance(analyzed_move.hiddenFeatures, dict):
        engine_meta = analyzed_move.hiddenFeatures.get("_engine") or {}
    return AnalyzedMoveData(
        index=idx,
        ply=move_obj.depth,
        san=san_main,
        uci=move_obj.move,
        fen_before=fen_before,
        fen_after=move_obj.position,
        score_cp=analyzed_move.score,
        phase_raw=str(analyzed_move.phase or "mid"),
        pvs=list(pvs) if pvs else [],
        hidden_features=analyzed_move.hiddenFeatures or {},
        captured_by_white=analyzed_move.capturedByWhite or {},
        captured_by_black=analyzed_move.capturedByBlack or {},
        analyzed_move=analyzed_move,
        eval_at_depth=dict(engine_meta.get("eval_at_depth") or {}),
        pv1_change_count=int(engine_meta.get("pv1_change_count", 0)),
    )


def _seed_book_exit_eval(
    retriever: AnalysisRetriever,
    previous_move_obj: Move | None,
    analyzed_rows: list[AnalyzedMoveData],
    fen_before: str,
) -> None:
    """First out-of-book move: backfill the prior move's eval so eval-swing
    detection has a baseline for this novelty."""
    if previous_move_obj is None or previous_move_obj.score is not None:
        return
    seed = retriever.evaluate_position(fen_before, depth=DEFAULT_ANALYSIS_DEPTH)
    if seed is None:
        return
    previous_move_obj.score = seed
    if analyzed_rows:
        analyzed_rows[-1] = analyzed_rows[-1].model_copy(update={"score_cp": seed})


def reverse_order_engine_analysis_enabled() -> bool:
    """Whether the engine pass sends positions to Stockfish last-move-first.

    Analyzing a game backward lets the (never explicitly cleared) transposition
    hash table already hold entries from the actual continuation once analysis
    reaches an earlier position, since that continuation is exactly the line
    just searched. Set ``REVERSE_ORDER_ENGINE_ANALYSIS=0`` to fall back to the
    old forward (first-move-first) order for comparison/debugging."""
    return os.environ.get("REVERSE_ORDER_ENGINE_ANALYSIS", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _engine_analysis_order(total_moves: int) -> range:
    """Ply indices in the order positions are sent to the engine."""
    if reverse_order_engine_analysis_enabled():
        return range(total_moves - 1, -1, -1)
    return range(total_moves)


def _requires_tactical_analysis(board_before: chess.Board, board_after: chess.Board) -> bool:
    """Book membership is metadata, not proof that a move is safe."""
    if board_after.is_game_over() or board_before.is_check() or board_after.is_check():
        return True
    for source in (board_before, board_after):
        board = source.copy(stack=False)
        for move in list(board.legal_moves):
            if not board.gives_check(move):
                continue
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return True
    return False


async def _run_engine_pass(
    game: Game,
    retriever: AnalysisRetriever,
    moves_list: list[Move],
    progress_callback: Callable[[float, str], Awaitable[None]],
) -> list[AnalyzedMoveData]:
    """Analyze every mainline ply (book or engine) into AnalyzedMoveData rows.

    The engine is fed positions in ``_engine_analysis_order`` (last-move-first
    by default, on the single persistent engine instance, hash never cleared
    between positions — see ``reverse_order_engine_analysis_enabled``). The
    chronological bookkeeping below (book-exit eval seeding, prev/now score
    deltas) always runs in a second, forward-order pass over the results so
    the ORDER positions were sent to the engine never changes which move an
    evaluation/PV ends up attached to.
    """
    total_moves = len(moves_list)
    phase_classifier = PhaseClassifier(retriever._eco_book)
    raw_by_idx: list[dict[str, Any] | None] = [None] * total_moves
    order = _engine_analysis_order(total_moves)

    for step, idx in enumerate(order):
        move_obj = moves_list[idx]
        progress = (step / max(total_moves, 1)) * ENGINE_PASS_PROGRESS_PCT
        board_before = _board_before_mainline_move(game, idx)
        fen_before = board_before.fen()
        try:
            san_main = board_before.san(chess.Move.from_uci(move_obj.move))
        except Exception:
            san_main = move_obj.move
        board_after = chess.Board(move_obj.position)
        uci_prefix = retriever._uci_prefix_for_depth(move_obj.depth)
        move_obj.phase = phase_classifier.classify(board_after, uci_prefix)
        if move_obj.phase == "early" and _requires_tactical_analysis(board_before, board_after):
            move_obj.phase = "mid"

        if move_obj.phase == "early":
            await progress_callback(progress, f"Book: move {idx + 1}/{total_moves}")
            analyzed_move, pvs = retriever.analyze_book_move(move_obj)
        else:
            await progress_callback(progress, f"Engine: move {idx + 1}/{total_moves}")
            analyzed_move, pvs = retriever.analyze_move(
                move_obj, stage=DEFAULT_ANALYSIS_DEPTH
            )

        raw_by_idx[idx] = {
            "move_obj": move_obj,
            "san_main": san_main,
            "fen_before": fen_before,
            "board_after": board_after,
            "analyzed_move": analyzed_move,
            "pvs": pvs,
        }

    analyzed_rows: list[AnalyzedMoveData] = []
    previous_move_obj: Move | None = None
    for idx in range(total_moves):
        raw = raw_by_idx[idx]
        assert raw is not None  # every idx was visited exactly once above
        move_obj = raw["move_obj"]
        if move_obj.phase != "early":
            _seed_book_exit_eval(
                retriever, previous_move_obj, analyzed_rows, raw["fen_before"]
            )
        analyzed_move = raw["analyzed_move"]
        _attach_guid_vector(analyzed_move, raw["board_after"])
        _attach_score_meta(analyzed_move, previous_move_obj)
        analyzed_rows.append(
            _build_analyzed_row(
                idx,
                move_obj,
                raw["san_main"],
                raw["fen_before"],
                analyzed_move,
                raw["pvs"],
            )
        )
        previous_move_obj = analyzed_move
    return analyzed_rows


def _dedup_claims(
    facts: Any,
    ply: int,
    last_claim_ply: dict[tuple[Any, ...], int],
    claim_window: int,
    structural_window: int,
) -> Any:
    """Mute only repeated, evidence-identical persistent positional claims.

    Concrete events must survive: a forecast capture and its later execution,
    or two different captures by the same side, are separate explanations.
    ``claim_window`` remains accepted for historical callers; event claims no
    longer share a coarse rule/side repetition budget.
    """
    kept = []
    muted = list(facts.muted_claims)
    for c in facts.claims:
        if c.rule_id not in _STRUCTURAL_RULE_IDS:
            kept.append(c)
            continue
        # Position counters are irrelevant, but piece placement, turn, castling
        # and en-passant rights all belong to the evidence. Realization keeps a
        # previously forecast state from suppressing the move that achieves it.
        baseline_fen = c.baseline_fen or (
            facts.display_line.start_fen if facts.display_line else ""
        )
        support_fen = c.support_fen or (
            facts.display_line.leaf_fen if facts.display_line else ""
        )
        key = (
            c.rule_id,
            c.beneficiary,
            c.realization,
            c.text,
            tuple(c.features_involved),
            c.delta_cp,
            c.flag_note,
            c.is_concession,
            " ".join(baseline_fen.split()[:4]),
            " ".join(support_fen.split()[:4]),
        )
        prev = last_claim_ply.get(key)
        if prev is not None and (ply - prev) <= structural_window:
            muted.append(c.text)
            continue
        last_claim_ply[key] = ply
        kept.append(c)
    return facts.model_copy(update={"claims": kept, "muted_claims": muted})


def _attach_comment_facts(
    move_events: list[MoveEvent],
    analyzed_rows: list[AnalyzedMoveData],
    engine_connector: EngineConnector | None = None,
) -> None:
    """Fire the rule engine per move, dedup persistent claims, promote key moments."""
    from app.core.commentary.rules import build_comment_facts

    prober = _leaf_probe_prober(engine_connector) if engine_connector else None
    claim_window = int(os.environ.get("CLAIM_DEDUP_WINDOW_PLIES", "6"))
    structural_window = int(os.environ.get("STRUCTURAL_DEDUP_WINDOW_PLIES", "24"))
    last_claim_ply: dict[tuple[Any, ...], int] = {}
    for mi, move_event in enumerate(move_events):
        if not (0 <= move_event.move_index < len(analyzed_rows)):
            continue
        row = analyzed_rows[move_event.move_index]
        try:
            # The prober is threaded into fact-building itself (not applied
            # after the fact) so a probe-trimmed leaf is what claims and the
            # feature diff are actually computed from -- a trim applied only
            # to the already-built facts.display_line would leave claims and
            # feature_diff computed against the untrimmed, possibly
            # horizon-effect-contaminated leaf.
            facts = build_comment_facts(
                row, move_event, depth=DEFAULT_ANALYSIS_DEPTH, prober=prober
            )
        except Exception as e:
            logger.warning("comment facts failed at ply %s: %s", move_event.ply, e)
            continue
        if facts is None:
            continue
        # Selection must see all evidence, before editorial repetition filtering.
        new_km = _promote_key_moment(move_event, facts, facts.claims)
        selection = _comment_selection(move_event, facts)
        metadata = dict(facts.selection_metadata)
        metadata["comment_selection"] = selection
        if mi > 0:
            previous = move_events[mi - 1]
            try:
                before = chess.Board(previous.fen_before)
                previous_move = chess.Move.from_uci(previous.uci)
                if previous_move in before.legal_moves:
                    san = before.san(previous_move)
                    before.push(previous_move)
                    if before.fen() == previous.fen_after == move_event.fen_before:
                        metadata["previous_move"] = {
                            "fen_before": previous.fen_before,
                            "fen_after": previous.fen_after,
                            "uci": previous.uci, "san": san,
                        }
            except ValueError:
                pass
        facts = facts.model_copy(update={"selection_metadata": metadata})
        if not selection["allowed"]:
            new_km = None
        if new_km:
            facts = _dedup_claims(
                facts, move_event.ply, last_claim_ply, claim_window, structural_window
            )
        update: dict[str, Any] = {"comment_facts": facts}
        if new_km != move_event.key_moment_type:
            update["key_moment_type"] = new_km
        move_events[mi] = move_event.model_copy(update=update)


def _build_analysis_context(
    headers: PgnMetadata,
    move_events: list[MoveEvent],
    default_opening_name: str | None,
    default_opening_eco: str | None,
) -> GameAnalysisContext:
    opening_name = default_opening_name
    opening_eco = default_opening_eco
    for move_event in move_events:
        opening_name = move_event.opening_name or opening_name
        opening_eco = move_event.opening_eco or opening_eco
    return GameAnalysisContext(
        metadata={
            "white": headers.whiteName,
            "black": headers.blackName,
            "result": headers.result,
            "whiteElo": headers.whiteElo,
            "blackElo": headers.blackElo,
        },
        move_events=move_events,
        critical_moments=[e for e in move_events if e.is_critical],
        opening_name=opening_name,
        opening_eco=opening_eco,
    )


def _board_turn_is_white(fen: str) -> bool:
    try:
        return chess.Board(fen).turn == chess.WHITE
    except Exception:
        return True


def _leaf_probe_prober(engine_connector: EngineConnector):
    """Cheap depth-limited eval prober for envisioned-line sanity checks.

    Returns None (probe skipped) when the env knob disables it or the engine
    fails, so the trim is a no-op — the pipeline never hard-depends on it."""
    import os

    depth = int(os.environ.get("LEAF_PROBE_DEPTH", "6"))
    if depth <= 0:
        return None

    def prober(fen: str) -> int | None:
        try:
            return engine_connector.evaluate_position(fen, depth=depth)
        except Exception:
            return None

    return prober


async def run_engine_analysis_to_json(
    pgn_string: str,
    engine_connector: EngineConnector,
    progress_callback: Callable[[float, str], Awaitable[None]],
    *,
    metadata_id: str | None = None,
) -> tuple[GameJson, EnginePipelineState]:
    """Passes 1-3: engine analysis + event extraction. Heuristic comments only."""
    game = _parse_single_game(pgn_string)
    eco_book = ECOBook()
    retriever = AnalysisRetriever(engine_connector, game, eco_book)
    headers = retriever.get_pgn_headers()
    opening_name, opening_eco = _resolve_opening_metadata(game, headers, eco_book)
    metadata = _build_metadata(headers, game, opening_name, opening_eco, metadata_id)

    moves_list = retriever.get_move_list()
    analyzed_rows = await _run_engine_pass(
        game, retriever, moves_list, progress_callback
    )

    # Deterministic opening comments for in-book plies (no engine, no LLM).
    from app.core.commentary.phases.early import attach_opening_comments

    attach_opening_comments(analyzed_rows, eco_book)

    await progress_callback(EVENT_EXTRACT_PROGRESS_PCT, "Extracting move events...")
    extractor = ChessEventExtractor(
        eco_book=eco_book, key_moment_detector=retriever.key_moment_detector
    )
    move_events = extractor.extract_events(game, analyzed_rows)

    # Targeted refutation scan (Guid: refute moves that look good at low
    # depth) over key-moment positions only. Optional; failures are logged
    # and never block the pipeline.
    try:
        from app.core.commentary.features.refutation_scan import RefutationScanner

        scanner = RefutationScanner(engine_connector)
        for mi, me in enumerate(move_events):
            if not me.key_moment_type or not me.fen_before:
                continue
            deep_after = me.eval_after_cp
            if deep_after is None:
                continue
            sign = 1 if _board_turn_is_white(me.fen_before) else -1
            mover_pov_deep = sign * int(deep_after)
            found = scanner.scan(me.fen_before, mover_pov_deep)
            if found:
                move_events[mi] = me.model_copy(
                    update={"refutations": [f.as_dict() for f in found]}
                )
    except Exception as e:
        logger.warning("refutation scan skipped: %s", e)

    _attach_comment_facts(move_events, analyzed_rows, engine_connector)
    context = _build_analysis_context(headers, move_events, opening_name, opening_eco)

    await progress_callback(ANALYSIS_DONE_PROGRESS_PCT, "Engine analysis complete.")
    state = EnginePipelineState(
        game=game,
        retriever=retriever,
        moves_list=moves_list,
        analyzed_rows=analyzed_rows,
        move_events=move_events,
        context=context,
        metadata=metadata,
    )
    return assemble_game_json(state), state


def rebuild_comment_facts(state: EnginePipelineState, engine_connector: EngineConnector) -> None:
    """Re-run event extraction and fact building on saved engine rows (no new engine search).

    Lets a study apply changed rules to identical engine evidence. Refutation
    scan results are kept from the saved events; the leaf probe uses the engine.
    """
    eco_book = ECOBook()
    detector = AnalysisRetriever(engine_connector, state.game, eco_book).key_moment_detector
    refutations = {event.ply: event.refutations for event in state.move_events}
    move_events = ChessEventExtractor(eco_book=eco_book, key_moment_detector=detector).extract_events(
        state.game, state.analyzed_rows)
    move_events = [event.model_copy(update={"refutations": refutations.get(event.ply) or []})
                   for event in move_events]
    _attach_comment_facts(move_events, state.analyzed_rows, engine_connector)
    state.move_events = move_events
    state.context = state.context.model_copy(
        update={"move_events": move_events, "critical_moments": [e for e in move_events if e.is_critical]})


def _pv_line_for_ai_payload(row: AnalyzedMoveData | None) -> list[dict[str, str]]:
    """SAN + FEN after each ply of PV1 for interactive commentary hover boards."""
    if not row or not row.pvs or not row.pvs[0]:
        return []
    out: list[dict[str, str]] = []
    board = chess.Board(row.fen_before)
    for pm in row.pvs[0]:
        uci = getattr(pm, "move", None)
        if not uci:
            break
        try:
            chm = chess.Move.from_uci(str(uci))
            san = board.san(chm)
            board.push(chm)
            out.append({"san": san, "fen": board.fen()})
        except Exception:
            break
    return out
