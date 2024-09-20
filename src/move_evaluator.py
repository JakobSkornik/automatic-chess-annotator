import chess.engine
import chess
from enum import Enum
from tqdm import tqdm

from src.classes.AltMoveScore import AltMoveScore
from src.classes.MovePoints import MovePoints
from src.classes.Score import Score
from src.classes.Wdl import Wdl


# Tag enum
class Tags(Enum):
    BRILLIANT_MOVE_TAG = "!!"
    GOOD_MOVE_TAG = "!"
    MISTAKE_TAG = "?"
    BLUNDER_TAG = "??"


class MoveEvaluator:
    def __init__(self, engine_path="stockfish", shallow_depth=5, deep_depth=20):
        """
        Initializes the MoveEvaluator with a chess engine, evaluation depths, and a time limit.

        :param engine_path: Path to the chess engine executable. Default is "stockfish".
        :param shallow_depth: Depth for short-term evaluation.
        :param deep_depth: Depth for long-term evaluation.
        :param time_limit: Time limit for evaluating each move in seconds.
        """
        self.engine_path = engine_path
        self.shallow_depth = shallow_depth
        self.deep_depth = deep_depth
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def evaluate_game(self, game, debug=False, prog_bar=True):
        """
        Evaluates each move in a game and returns a list of move evaluations.

        :param game: A chess.pgn.Game object representing the game to be evaluated.
        :return: A list of dictionaries, each containing move information and evaluation.
        """
        board = game.board()
        move_evaluations = []

        prev_terms = None

        # Loop through moves and display progress bar
        n_moves = len(list(game.mainline_moves()))
        for move in tqdm(game.mainline_moves(), total=n_moves, disable=not prog_bar):
            board.push(move)

            score = self.get_score(board, prev_terms)
            clsf = self.classify_move(score, board, move)

            move_evaluations.append(
                {
                    "move": move,
                    "evaluation": clsf,
                    "color": board.turn,
                    "fen": board.fen(),
                }
            )

            prev_terms = {
                "short_term_info": score.short_term_info,
                "long_term_info": score.long_term_info,
            }

            if debug:
                print(clsf, move, "white" if not board.turn else "black")

        return move_evaluations

    def get_score(self, board, prev_terms):
        """
        Gets the evaluation score for a move at different depths.

        :param board: The current state of the board.
        :param prev_term_info: The engine's previous analysis info.
        """

        stm = board.turn

        # Short-term evaluation at shallow depth
        st_analysis = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.shallow_depth),
        )
        st_score = st_analysis["score"]
        st_wdl = self.get_wdl_from_score(st_score, stm)

        # Long-term evaluation at deep depth
        lt_analysis = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.deep_depth),
        )
        lt_score = lt_analysis["score"]
        lt_wdl = self.get_wdl_from_score(lt_score, stm)

        pst = prev_terms["short_term_info"] if prev_terms else None
        plt = prev_terms["long_term_info"] if prev_terms else None

        score = Score(st_wdl, lt_wdl, pst, plt)
        return score

    def get_wdl_from_score(self, score, stm):
        """
        Adapted from:
          - https://support.chess.com/en/articles/8572705-how-are-moves-classified-what-is-a-blunder-or-brilliant-etc

        :param score: The evaluation score.
        :param stm: The side to move.
        :return : A Wdl object.
        """

        wdl = score.wdl()
        wins, draws, losses = wdl[0], wdl[1], wdl[2]

        score = wins + draws / 2
        total = wins + draws + losses

        score_rate = score / total
        win_rate = wins / total
        loss_rate = losses / total

        # Win chance
        bwc = win_rate if stm == chess.WHITE else loss_rate
        wwc = win_rate if stm == chess.BLACK else loss_rate

        # Score rate
        bsr = score_rate if stm == chess.WHITE else 1 - score_rate
        wsr = score_rate if stm == chess.BLACK else 1 - score_rate

        wdl = Wdl(wwc, wsr, bwc, bsr)
        return wdl

    def classify_move(self, score, board, move):
        """
        Classifies a move based on short-term and long-term evaluation scores and other factors.

        :param score: The evaluation score at different depths after the move.
        :param board: The current state of the board after the move.
        :param move: The move to be classified.
        :return: A string classifying the move as '!!', '!', '?', '??', or None.
        """

        scores = self._process_score(board, score)
        if scores is None:
            return None, None

        # Obtain pv scores
        plti = score.prev_lterm_info
        pv_scores = self.get_pv_scores(board, plti)

        # Blunder (??) (blunder can happen on non-quiescent moves)
        blunder, t, e = self.blunder_check(scores, pv_scores)
        if blunder:
            return t, scores, e

        # Quiescence check: only classify quiescent moves
        if not self._is_position_quiescent(board, move):
            return "Not Quiescent", scores, None

        # Brilliant move (!!)
        brilliant, t, e = self.brilliant_move_check(scores, pv_scores, move)
        if brilliant:
            return t, scores, e

        # Good move (!)
        good, t, e = self.good_move_check(scores, pv_scores, move)
        if good:
            return t, scores, e

        # Mistake (?)
        mistake, t, e = self.mistake_check(scores, pv_scores)
        if mistake:
            return t, scores, e

        return None, scores, None

    def get_pv_scores(self, board, prev_term):
        """
        Calculates the evaluation scores for the principal variations at a given board.

        :param board: The current state of the board.
        :param prev_term: The engine's previous analysis info.
        :return: A list of tuples containing the move and its evaluation score.
        """

        stm = board.turn
        pvs = []

        if not prev_term:
            return (None, None)

        prev_board = board.copy()
        prev_board.pop()

        for move in prev_board.legal_moves:
            pv_board = prev_board.copy()
            pv_board.push(move)

            analysis = self.engine.analyse(
                pv_board, chess.engine.Limit(depth=10, time=0.1)
            )
            score = analysis["score"]
            prev_score = prev_term

            # Obtain wdl for current POV from previous board state
            wdl = self.get_wdl_from_score(score, stm)
            if stm == chess.WHITE:
                diff = wdl.white_score_rate - prev_score.white_score_rate
            else:
                diff = wdl.black_score_rate - prev_score.black_score_rate

            alt_move_score = AltMoveScore(move, diff)
            pvs.append(alt_move_score)

        # sort pvs by score desc
        pvs.sort(key=lambda x: x.diff, reverse=True)
        return pvs

    def _process_score(self, board, score):
        """
        Calculates the difference in evaluation scores between the current
        and previous moves.

        :param board: The current state of the board.
        :param score: The evaluation score at different depths after the move.
        :return: The difference in short-term and long-term evaluation scores.
        """

        stm = board.turn

        # Expected points
        if stm == chess.WHITE:
            st_exp = score.short_term_info.white_score_rate
            lt_exp = score.long_term_info.white_score_rate
            pst_exp = (
                score.prev_sterm_info.white_score_rate if score.prev_sterm_info else 0.5
            )
            plt_exp = (
                score.prev_lterm_info.white_score_rate if score.prev_lterm_info else 0.5
            )
        else:
            st_exp = score.short_term_info.black_score_rate
            lt_exp = score.long_term_info.black_score_rate
            pst_exp = (
                score.prev_sterm_info.black_score_rate if score.prev_sterm_info else 0.5
            )
            plt_exp = (
                score.prev_lterm_info.black_score_rate if score.prev_lterm_info else 0.5
            )

        std = st_exp - pst_exp
        ltd = lt_exp - plt_exp

        wcw = score.long_term_info.white_win_chance
        wcb = score.long_term_info.black_win_chance

        mp = MovePoints(std, ltd, st_exp, lt_exp, pst_exp, plt_exp, wcw, wcb, stm)

        return mp

    def brilliant_move_check(self, scores, pv_scores, move):
        """
        Checks if the move is a brilliant move based on the
        evaluation scores and principal variations.

        :param scores: Short-term and long-term evaluation scores.
        :param pv_scores: The evaluation scores for the principal variations.
        :param move: The move to be evaluated.
        :return: True if the move is a brilliant move, False otherwise along
                 with the tag and the explanation.
        """

        # Diff limits
        LIMITS = [0.1, 1]
        BRILLIANT_LIMITS = [0.2, 1]

        # Expected points limits
        VERY_LOSING_LIMITS = [0, 0.3]
        VERY_WINNING_LIMITS = [0.9, 1]
        LONG_DELTA_FACTOR = 0.7

        std = scores.short_term_diff
        ltd = scores.long_term_diff
        lt_exp = scores.long_term_ep
        plt_exp = scores.prev_long_term_ep

        """ Scenario 1:
        
        The move is evaluated highly and has long-term score 
        much higher than the short-term score. 
        """

        if (
            self.is_in(ltd, BRILLIANT_LIMITS)
            and std <= LONG_DELTA_FACTOR * ltd
            and not self.is_in(lt_exp, VERY_LOSING_LIMITS)
        ):
            explanation = f"The move is evaluated highly and has long-term score ({ltd:0.2f}) much higher than the short-term score ({std:0.2f})."
            return True, Tags.BRILLIANT_MOVE_TAG.value, explanation

        """ Scenario 2:

        The move is the best move among the possible variations 
        and is rated much higher than the second best move.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        best_move_score = pv_scores[0].diff
        second_best_move_score = pv_scores[1].diff

        if (
            best_move_score > 0
            and second_best_move_score <= 0
            and self.is_in(ltd, LIMITS)
            and not self.is_in(lt_exp, VERY_LOSING_LIMITS)
            and not self.is_in(plt_exp, VERY_WINNING_LIMITS)
        ):
            explanation = f"The only positive move among possible variations."
            return True, Tags.GOOD_MOVE_TAG.value, explanation

        return False, None, None

    def good_move_check(self, scores, pv_scores, move):
        """
        Checks if the move is a good move based on the evaluation
        scores and principal variations.

        :param scores: Short-term and long-term evaluation scores.
        :param pv_scores: The evaluation scores for the principal variations.
        :return: True if the move is a good move, False otherwise along
                 with the tag and the explanation.
        """

        # Diff limits
        LIMITS = [0.02, 0.1]

        # Expected points limits
        LOSING_LIMITS = [0, 0.48]
        EQUAL_LIMITS = [0.48, 0.52]
        WINNING_LIMITS = [0.52, 1]

        ltd = scores.long_term_diff
        std = scores.short_term_diff
        st_exp = scores.short_term_ep
        lt_exp = scores.long_term_ep
        plt_exp = scores.prev_long_term_ep
        pst_exp = scores.prev_short_term_ep

        """ Scenario 1:
        Turning a losing position into a winning one.
        """

        if (
            self.is_in(std, LIMITS)
            and self.is_in(pst_exp, LOSING_LIMITS)
            and self.is_in(st_exp, WINNING_LIMITS)
        ):
            explanation = f"Turning a losing position into a winning one ({pst_exp:0.2f} -> {st_exp:0.2f})."
            return True, Tags.GOOD_MOVE_TAG.value, explanation

        """ Scenario 2:
        Turning a losing position into an equal one.
        """

        if (
            self.is_in(std, LIMITS)
            and self.is_in(pst_exp, LOSING_LIMITS)
            and self.is_in(st_exp, EQUAL_LIMITS)
        ):
            explanation = f"Turning a losing position into an equal one ({pst_exp:0.2f} -> {st_exp:0.2f})."
            return True, Tags.GOOD_MOVE_TAG.value, explanation

        """ Scenario 3:
        Turning a drawn position into a winning one.
        """

        if (
            self.is_in(std, LIMITS)
            and self.is_in(pst_exp, EQUAL_LIMITS)
            and self.is_in(st_exp, WINNING_LIMITS)
        ):
            explanation = f"Turning a drawn position into a winning one ({pst_exp:0.2f} -> {st_exp:0.2f})."
            return True, Tags.GOOD_MOVE_TAG.value, explanation

        return False, None, None

    def mistake_check(self, scores, pv_scores):
        """
        Checks if the move is a mistake based on the evaluation
        scores and principal variations.

        :param scores: Short-term and long-term evaluation scores.
        :param pv_scores: The evaluation scores for the principal variations.
        :return: True if the move is a mistake, False otherwise along
                 with the tag and the explanation.
        """

        # Diff limits
        MISTAKE_LIMITS = [-0.2, -0.1]
        POSITIVE_LIMITS = [0.0, 1]

        ltd = scores.long_term_diff
        lt_exp = scores.long_term_ep
        plt_exp = scores.prev_long_term_ep

        """ Scenario 1:

        Missed opportunity to make a good move.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        best_move_score = pv_scores[0].diff
        if self.is_in(ltd, MISTAKE_LIMITS) and self.is_in(
            best_move_score, POSITIVE_LIMITS
        ):
            explanation = f"Missed opportunity ({ltd:0.2f}) to make a good move {pv_scores[0].move} ({best_move_score:0.2f})."
            return True, Tags.MISTAKE_TAG.value, explanation

        """ Scenario 2:
        
        A bad move leads to a slightly worse position.
        """
        if self.is_in(ltd, MISTAKE_LIMITS):
            explanation = (
                f"Leads to a slightly worse position ({plt_exp:0.2f} -> {lt_exp:0.2f})."
            )
            return True, Tags.MISTAKE_TAG.value, explanation

        return False, None, None

    def blunder_check(self, scores, pv_scores):
        """
        Checks if the move is a blunder based on the evaluation scores.

        :param scores: Short-term and long-term evaluation scores.
        :return: True if the move is a blunder, False otherwise along
                 with the tag and the explanation.
        """

        # Diff limits
        BLUNDER_LIMITS = [-1, -0.2]
        POSITIVE_LIMITS = [0.0, 1]

        ltd = scores.long_term_diff

        """ Scenario 1:

        Missed opportunity to make a neutral or good move and massively
        tanking your score.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        best_move_score = pv_scores[0].diff
        if self.is_in(ltd, BLUNDER_LIMITS) and self.is_in(
            best_move_score, POSITIVE_LIMITS
        ):
            explanation = f"Missed opportunity ({ltd:0.2f}) to make a neutral or good move {pv_scores[0].move} ({best_move_score:0.2f}) and heavily tanking your score."
            return True, Tags.BLUNDER_TAG.value, explanation

        """ Scenario 2:
        Move that leads to a significantly worse position.
        """

        if self.is_in(ltd, BLUNDER_LIMITS):
            explanation = f"Leads to a significantly worse position ({ltd:0.2f}) with a clear better option available {pv_scores[0].move} ({best_move_score:0.2f})."
            return True, Tags.BLUNDER_TAG.value, explanation

        return False, None, None

    def _verify_pv_scores(self, pv_scores):
        """
        :param pv_scores: The evaluation scores for the principal variations.
        :return: True if the principal variations are not available, False otherwise.
        """

        return len(pv_scores) < 2 or not pv_scores[0] or not pv_scores[1]

    def _is_difficult_move(self, move, prev_short_term, prev_long_term):
        """
        Determines if the move is difficult based on the engine's principal variations at different depths.

        :param board: The current state of the board.
        :param short_term_info: The engine's short-term analysis info.
        :param long_term_info: The engine's long-term analysis info.
        :return: True if the move is considered difficult, False otherwise.
        """

        if not prev_short_term or not prev_long_term:
            return False

        # Compare PVs at different depths to assess difficulty
        short_term_best_move = prev_short_term["pv"][0]
        long_term_best_move = prev_long_term["pv"][0]

        # If the move is not favored at shallow depth but is the best at deep depth, it is difficult
        if move != short_term_best_move and move == long_term_best_move:
            return True

        return False

    def _is_obvious_mistake(self, move, prev_term_info):
        """
        Determines if the mistake was an obvious one, which might indicate a blunder instead.

        :param move: The move to be evaluated.
        :param prev_term_info: The engine's previous analysis info.
        :return: True if the mistake is obvious, False otherwise.
        """

        # Check if better alternatives was obvious at previous depth
        return len(prev_term_info["pv"]) > 1 and prev_term_info["pv"][0] != move

    def _is_position_quiescent(self, board, move):
        """
        Determines if the position is quiescent, i.e., no checks, captures, or threats.

        :param board: The current state of the board.
        :param move: The move to be evaluated.
        :return: True if the position is quiescent, False otherwise.
        """

        prev_board = board.copy()
        prev_board.pop()

        if board.is_check():
            return False

        if prev_board.is_capture(move):
            return False

        return True

    def is_in(self, value: float, range_tuple: tuple) -> bool:
        """
        Determines if a value is within a given range.

        :param value: The value to be checked.
        :param range_tuple: A tuple representing the range.
        :return: True if the value is within the range, False otherwise.
        """
        lower_bound, upper_bound = sorted(range_tuple)  # Ensure the tuple is sorted
        return lower_bound <= value <= upper_bound

    def close_engine(self):
        """
        Closes the chess engine process.
        """
        self.engine.close()
