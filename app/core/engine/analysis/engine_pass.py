"""The engine analysis pass: ``AnalysisRetriever`` drives Stockfish over the
mainline, producing per-move evals, PVs, hidden features and book metadata."""

from __future__ import annotations

import logging

import chess
from chess.pgn import Game

from app.core.commentary.features.positional_features import compute_hidden_features
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.openings.eco_book import ECOBook, parse_pgn_eco_tag
from app.core.engine.engine_connector import EngineConnector
from app.models.Move import AnalysisStage, Move
from app.models.PgnMetadata import PgnMetadata

from .constants import ANALYSIS_STAGES, DEFAULT_PV_COUNT, MATE_SCORE
from .pass_utils import _compute_features_delta, _cp_and_pv1

logger = logging.getLogger(__name__)

# Depths re-searched on the after-move position to gauge search instability.
INSTABILITY_DEPTHS = (8, 12, 16)


class AnalysisRetriever:
    def __init__(
        self,
        engine_connector: EngineConnector,
        game: Game,
        eco_book: ECOBook | None = None,
    ):
        self.analysis_stages = ANALYSIS_STAGES
        self.engine_connector = engine_connector
        self.game = game
        self._eco_book = eco_book or ECOBook()
        self.id_counter = 0
        self.analyzed_game: list[Move] = self.get_move_list()
        self.key_moment_detector = KeyMomentDetector()

    def _uci_prefix_for_depth(self, depth: int) -> list[str]:
        """First ``depth`` mainline half-moves as UCI (same order as ``Move.depth``)."""
        out: list[str] = []
        board = self.game.board()
        for i, gm in enumerate(self.game.mainline_moves()):
            if i >= depth:
                break
            out.append(board.uci(gm))
            board.push(gm)
        return out

    def get_pgn_headers(self) -> PgnMetadata:
        """Extract game metadata from PGN headers."""
        headers = self.game.headers

        return PgnMetadata(
            whiteName=headers.get("White", ""),
            blackName=headers.get("Black", ""),
            whiteElo=(
                int(headers.get("WhiteElo", 0))
                if headers.get("WhiteElo", "").isdigit()
                else None
            ),
            blackElo=(
                int(headers.get("BlackElo", 0))
                if headers.get("BlackElo", "").isdigit()
                else None
            ),
            event=headers.get("Event", ""),
            site=headers.get("Site", ""),
            round=headers.get("Round", ""),
            opening=headers.get("Opening", ""),
            eco=parse_pgn_eco_tag(headers),
            result=headers.get("Result", ""),
        )

    def get_game_id(self) -> int:
        """
        Returns a unique ID for the game.
        This is used to identify the game in the database.
        """
        self.id_counter += 1
        return self.id_counter

    def get_move_list(self) -> list[Move]:
        """
        Returns a list of moves in the game.
        If PGN parsing failed during init, this will return an empty list.
        """
        moves = []
        board = self.game.board()

        depth = 1
        for chess_move in self.game.mainline_moves():
            board_before_move = board.copy()
            san_representation = board.uci(chess_move)
            board.push(chess_move)
            fen_after_move = board.fen()

            move_obj = Move(
                id=self.get_game_id(),
                position=fen_after_move,
                move=san_representation,
                context="mainline",
                isAnalyzed=False,
                depth=depth,
                piece=self._get_piece_for_move(
                    board_before_move=board_before_move, san_move=san_representation
                ),
            )
            moves.append(move_obj)
            depth += 1

        return moves

    def analyze_book_move(self, main_move_obj: Move) -> tuple[Move, list[list[Move]]]:
        """Opening-book ply: static features and metadata only — no engine calls at all."""
        board_after_move = chess.Board(main_move_obj.position)
        uci_prefix = self._uci_prefix_for_depth(main_move_obj.depth)
        _info_book, matched_ply = self._eco_book.match(uci_prefix)

        main_move_obj.phase = "early"
        main_move_obj.score = None
        try:
            after_features = compute_hidden_features(
                board_after_move, in_opening_book_phase=True
            )
        except Exception as e:
            logger.error(
                f"Hidden features (book) failed at depth {main_move_obj.depth} "
                f"FEN={board_after_move.fen()}: {e}"
            )
            after_features = {"error": str(e)}
        if isinstance(after_features, dict):
            after_features.setdefault("_engine", {})
            after_features["_engine"]["opening_book_hit"] = True
            after_features["_engine"]["opening_matched_ply"] = int(matched_ply)
        main_move_obj.hiddenFeatures = after_features
        (
            main_move_obj.capturedByWhite,
            main_move_obj.capturedByBlack,
        ) = self._get_all_captured_pieces(board_after_move)
        main_move_obj.isAnalyzed = True
        main_move_obj.analysisStage = AnalysisStage.FINAL
        main_move_obj.analysisVersion = 1
        return main_move_obj, []

    def evaluate_position(self, fen: str, *, depth: int) -> int | None:
        """One-off White-POV cp eval of a position (used to seed the book-exit baseline)."""
        return self.engine_connector.evaluate_position(fen, depth=depth)

    def analyze_move(
        self, main_move_obj: Move, stage: float
    ) -> tuple[Move, list[list[Move]]]:
        """Eval + features on the after-move position; PVs from the position before."""
        board_after = chess.Board(main_move_obj.position)
        after_primary = self._primary_after(board_after, stage)
        main_move_obj.score = (
            after_primary.get("score").white().score(mate_score=MATE_SCORE)
        )
        in_book, matched_ply = self._book_status(main_move_obj.depth)
        # Phase is decided before the engine runs; this path is out-of-book only.
        if not main_move_obj.phase:
            main_move_obj.phase = "mid"

        eval_at_depth, pv1_change_count = self._search_instability(
            board_after, after_primary, stage
        )
        after_features = self._after_features(
            board_after,
            main_move_obj.depth,
            in_book,
            matched_ply,
            eval_at_depth,
            pv1_change_count,
            after_primary,
        )
        (
            main_move_obj.capturedByWhite,
            main_move_obj.capturedByBlack,
        ) = self._get_all_captured_pieces(board_after)
        main_move_obj.isAnalyzed = True
        main_move_obj.analysisStage = (
            AnalysisStage.FINAL
            if stage == max(self.analysis_stages)
            else AnalysisStage.DEEP
        )
        main_move_obj.analysisVersion = 1

        board_before = self._board_before_move(main_move_obj.depth)
        before_features = self._before_features(board_before, main_move_obj.depth)
        self._attach_features(main_move_obj, after_features, before_features)
        pvs = self._compute_pvs(board_before, stage, main_move_obj.depth)
        return main_move_obj, pvs

    def _primary_after(self, board_after: chess.Board, stage: float) -> dict:
        """Engine eval of the position after the move (multipv=1 primary line)."""
        results = self.engine_connector.analyse(board_after, depth=stage, multiPv=1)
        return results[0] if isinstance(results, list) else results

    def _book_status(self, depth: int) -> tuple[bool, int]:
        """Whether this ply is still in the opening book, and the matched ply count."""
        uci_prefix = self._uci_prefix_for_depth(depth)
        info_book, matched_ply = self._eco_book.match(uci_prefix)
        in_book = (
            len(uci_prefix) > 0
            and info_book is not None
            and matched_ply >= len(uci_prefix)
        )
        return in_book, matched_ply

    def _search_instability(
        self, board_after: chess.Board, after_primary: dict, stage: float
    ) -> tuple[dict[int, int], int]:
        """Eval-by-depth and PV1 change count, from ONE streamed search to the
        final depth (snapshots at the shallower depths) instead of a re-search
        per depth. Falls back to the per-depth re-searches when the engine or
        protocol does not cooperate, so the output contract is unchanged."""
        stage_i = int(stage)
        depths = sorted({d for d in INSTABILITY_DEPTHS if d != stage_i} | {stage_i})
        if depths[-1] != stage_i:
            depths.append(stage_i)
        final, snaps = self.engine_connector.analyse_with_depth_snapshots(
            board_after, stage_i, snapshot_depths=tuple(depths)
        )
        eval_at_depth: dict[int, int] = {}
        pv1_ucis: list[str | None] = []
        for d in depths:
            if d == stage_i:
                cp, uci = _cp_and_pv1(final)
            else:
                cp, uci = snaps.get(d, (None, None))
            if cp is None and d != stage_i and uci is None:
                # Snapshot missed (streaming unavailable): explicit shallow search.
                try:
                    inf = self.engine_connector.analyse(
                        board_after, depth=d, multiPv=1
                    )
                except Exception:
                    inf = None
                cp, uci = _cp_and_pv1(inf)
            if cp is not None:
                eval_at_depth[d] = cp
            pv1_ucis.append(uci)
        pv1_change_count = sum(
            1
            for i in range(len(pv1_ucis) - 1)
            if pv1_ucis[i] and pv1_ucis[i + 1] and pv1_ucis[i] != pv1_ucis[i + 1]
        )
        return eval_at_depth, pv1_change_count

    def _after_features(
        self,
        board_after: chess.Board,
        depth: int,
        in_book: bool,
        matched_ply: int,
        eval_at_depth: dict[int, int],
        pv1_change_count: int,
        after_primary: dict,
    ) -> dict:
        """Hidden features on the after-move position, decorated with engine meta."""
        try:
            features = compute_hidden_features(
                board_after, in_opening_book_phase=in_book
            )
        except Exception as e:
            logger.error(
                "Hidden features (after) failed at depth %s FEN=%s: %s",
                depth,
                board_after.fen(),
                e,
            )
            features = {"error": str(e)}
        if isinstance(features, dict):
            engine_meta = features.setdefault("_engine", {})
            engine_meta["eval_at_depth"] = eval_at_depth
            engine_meta["pv1_change_count"] = pv1_change_count
            engine_meta["opening_book_hit"] = bool(in_book)
            engine_meta["opening_matched_ply"] = int(matched_ply)
            # Engine continuation from the after-move position = the played move's
            # "envisioned" line (no extra search needed).
            try:
                engine_meta["after_pv_uci"] = [
                    m.uci() for m in (after_primary.get("pv") or [])
                ]
            except Exception:
                engine_meta["after_pv_uci"] = []
        return features

    def _board_before_move(self, depth: int) -> chess.Board:
        """Reconstruct the board just before the mainline move at ``depth``."""
        board = self.game.board()
        try:
            target_depth = max(0, int(depth) - 1)
        except Exception:
            target_depth = 0
        for idx, game_move in enumerate(self.game.mainline_moves()):
            if idx >= target_depth:
                break
            board.push(game_move)
        return board

    def _before_features(self, board_before: chess.Board, depth: int) -> dict:
        """Hidden features on the before-move position (for the strategic delta)."""
        prefix_before = self._uci_prefix_for_depth(max(0, depth - 1))
        info_book, matched_ply = self._eco_book.match(prefix_before)
        in_book_before = (
            len(prefix_before) > 0
            and info_book is not None
            and matched_ply >= len(prefix_before)
        )
        try:
            return compute_hidden_features(
                board_before, in_opening_book_phase=in_book_before
            )
        except Exception as e:
            logger.error(
                "Hidden features (before) failed at depth %s FEN=%s: %s",
                depth,
                board_before.fen(),
                e,
            )
            return {"error": str(e)}

    def _attach_features(
        self, main_move_obj: Move, after_features: dict, before_features: dict
    ) -> None:
        """Keep after-features at top level; nest before-features + delta under _ai."""
        try:
            positional_delta = _compute_features_delta(before_features, after_features)
        except Exception:
            positional_delta = {}
        main_move_obj.hiddenFeatures = after_features
        try:
            if isinstance(main_move_obj.hiddenFeatures, dict):
                ai_meta = main_move_obj.hiddenFeatures.setdefault("_ai", {})
                ai_meta["before"] = before_features
                ai_meta["delta"] = positional_delta
        except Exception:
            pass

    def _build_pv_steps(
        self, board_before: chess.Board, engine_pv_moves: list, pv_idx: int, depth: int
    ) -> list[Move]:
        steps: list[Move] = []
        board = chess.Board(board_before.fen())
        for pv_move_idx, pv_chess_move in enumerate(engine_pv_moves):
            uci = pv_chess_move.uci()
            before_pv = board.copy()
            board.push(pv_chess_move)
            steps.append(
                Move(
                    id=self.get_game_id(),
                    position=board.fen(),
                    move=uci,
                    context=f"pv_{pv_idx}_step_{pv_move_idx}",
                    isAnalyzed=False,
                    piece=self._get_piece_for_move(
                        board_before_move=before_pv, san_move=uci
                    ),
                    depth=depth + pv_move_idx + 1,
                )
            )
        return steps

    def _compute_pvs(
        self, board_before: chess.Board, stage: float, depth: int
    ) -> list[list[Move]]:
        """Engine PVs from the before-move position as lists of step Moves."""
        pv_results = self.engine_connector.analyse(
            board_before, depth=stage, multiPv=DEFAULT_PV_COUNT
        )
        all_pvs: list[list[Move]] = []
        for pv_idx, pv_data in enumerate(pv_results):
            score = None
            if pv_data.get("score"):
                score = pv_data["score"].white().score(mate_score=MATE_SCORE)
            engine_pv = pv_data.get("pv")
            if not engine_pv:
                if score is not None:
                    all_pvs.append([])
                continue
            steps = self._build_pv_steps(board_before, engine_pv, pv_idx, depth)
            if score is not None and steps:
                steps[0].score = score
            all_pvs.append(steps)
        return all_pvs

    def _get_piece_for_move(
        self, board_before_move: chess.Board, san_move: str
    ) -> str | None:
        """Helper to determine the piece (e.g., wP, bN) that made a move.

        Tries SAN first, then falls back to UCI to be robust with input formats.
        """
        move_obj = None
        try:
            move_obj = board_before_move.parse_san(san_move)
        except Exception:
            try:
                move_obj = chess.Move.from_uci(san_move)
                if move_obj not in board_before_move.legal_moves:
                    move_obj = None
            except Exception:
                move_obj = None

        if move_obj is None:
            return None

        piece = board_before_move.piece_at(move_obj.from_square)
        if piece:
            color_char = "w" if piece.color == chess.WHITE else "b"
            return f"{color_char}{piece.symbol().upper()}"
        return None

    def _get_all_captured_pieces(
        self, board: chess.Board
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Counts the missing original pieces for both players.
        The first dictionary returned contains Black pieces captured by White.
        The second dictionary returned contains White pieces captured by Black.
        Keys are piece symbols (p, n, b, r, q), values are counts.
        """
        initial_piece_counts = {
            chess.PAWN: 8,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.ROOK: 2,
            chess.QUEEN: 1,
        }

        # Standard algebraic notation for pieces (lowercase)
        piece_to_symbol = {
            chess.PAWN: "p",
            chess.KNIGHT: "n",
            chess.BISHOP: "b",
            chess.ROOK: "r",
            chess.QUEEN: "q",
        }

        # Stores Black's pieces that White has captured
        black_pieces_captured_by_white: dict[str, int] = {}
        # Stores White's pieces that Black has captured
        white_pieces_captured_by_black: dict[str, int] = {}

        for piece_type, initial_count in initial_piece_counts.items():
            symbol = piece_to_symbol[piece_type]

            # Count White's current pieces of this type
            current_white_pieces_on_board = len(board.pieces(piece_type, chess.WHITE))
            # Difference is the number of White pieces of this type captured by Black
            num_white_captured = initial_count - current_white_pieces_on_board
            if num_white_captured > 0:
                white_pieces_captured_by_black[symbol] = num_white_captured

            # Count Black's current pieces of this type
            current_black_pieces_on_board = len(board.pieces(piece_type, chess.BLACK))
            # Difference is the number of Black pieces of this type captured by White
            num_black_captured = initial_count - current_black_pieces_on_board
            if num_black_captured > 0:
                black_pieces_captured_by_white[symbol] = num_black_captured

        # The function is expected to return (pieces_captured_by_white, pieces_captured_by_black)
        return black_pieces_captured_by_white, white_pieces_captured_by_black
