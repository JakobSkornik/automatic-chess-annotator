import chess.engine
import chess
from enum import Enum
from tqdm import tqdm


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
            eval = self.get_score(board, prev_terms)

            # Classify the move based on both shallow and deep evaluations
            clsf = self.classify_move(eval, board, move)

            move_evaluations.append(
                {
                    "move": move,
                    "evaluation": clsf,
                    "color": board.turn,
                    "fen": board.fen(),
                }
            )

            prev_terms = {
                "short_term_info": eval["short_term_info"],
                "long_term_info": eval["long_term_info"],
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

        # Short-term evaluation at shallow depth
        short_term_info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.shallow_depth),
        )

        # Long-term evaluation at deep depth
        long_term_info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.deep_depth),
        )

        score = {
            "short_term_info": short_term_info,
            "long_term_info": long_term_info,
            "prev_lterm_info": prev_terms["long_term_info"] if prev_terms else None,
            "prev_sterm_info": prev_terms["short_term_info"] if prev_terms else None,
        }

        return score

    def classify_move(self, evaluation, board, move):
        """
        Classifies a move based on short-term and long-term evaluation scores and other factors.

        :param evaluation: The evaluation score at different depths after the move.
        :param board: The current state of the board after the move.
        :param move: The move to be classified.
        :return: A string classifying the move as '!!', '!', '?', '??', or None.
        """
        plti = evaluation["prev_lterm_info"]

        scores = self._parse_eval(board, evaluation)
        if scores is None:
            return None, None

        # Obtain pv scores
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
        good, t, e = self.good_move_check(scores, pv_scores)
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
        MATE_SCORE = 10000
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
            score = analysis["score"].white().score(mate_score=MATE_SCORE)
            prev_score = prev_term["score"].white().score(mate_score=MATE_SCORE)

            diff = score - prev_score
            if board.turn == chess.WHITE:
                diff = -diff

            pvs.append((move, diff if score and prev_score else 0))

        # sort pvs by score desc
        pvs.sort(key=lambda x: x[1], reverse=True)
        print(pvs[0][1])
        return pvs

    def _parse_eval(self, board, evaluation):
        """
        Calculates the difference in evaluation scores between the current and previous moves.

        :param board: The current state of the board.
        :param evaluation: The evaluation score at different depths after the move.
        :return: The difference in short-term and long-term evaluation scores.
        """

        MATE_SCORE = 10000

        sti = evaluation["short_term_info"]
        lti = evaluation["long_term_info"]

        psti = evaluation["prev_sterm_info"]
        plti = evaluation["prev_lterm_info"]

        sts = sti["score"].white().score(mate_score=MATE_SCORE)
        lts = lti["score"].white().score(mate_score=MATE_SCORE)

        psts = (psti["score"].white().score(mate_score=MATE_SCORE)) if psti else 0
        plts = (plti["score"].white().score(mate_score=MATE_SCORE)) if plti else 0

        if plts is None or psts is None:
            return None

        std = sts - psts
        ltd = lts - plts

        lwdl = lti["score"].wdl().white().expectation()
        pwdl = psti["score"].wdl().white().expectation() if psti else 0
        win_chance_diff = lwdl - pwdl
        win_chance = lwdl

        # Technically we check if previous to make a move is white or black
        if board.turn == chess.WHITE:
            std = -std
            ltd = -ltd
            win_chance_diff = -win_chance_diff

        return std, ltd, sts, lts, psts, plts, board.turn, win_chance_diff, win_chance

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

        BRILLIANT_MOVE = 150
        LONG_DELTA_FACTOR = 0.8
        BRILLIANT_MOVE_FACTOR = 0.7

        std = scores[0]
        ltd = scores[1]
        wcd = scores[7]

        if wcd < 0.1:
            return False, None, None

        """ Scenario 1:
        
        The move is evaluated highly and has long-term score 
        much higher than the short-term score. 
        """

        if ltd >= BRILLIANT_MOVE and std <= LONG_DELTA_FACTOR * ltd:
            explanation = f"The move is evaluated highly and has long-term score ({ltd}) much higher than the short-term score ({std})."
            return True, Tags.BRILLIANT_MOVE_TAG.value, explanation

        """ Scenario 2:

        The move is the best move among the possible variations 
        and is rated much higher than the second best move.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        if (
            len(pv_scores) >= 2
            and move == pv_scores[0][0]
            and pv_scores[1][1] / (pv_scores[0][1] + 1e-7) >= BRILLIANT_MOVE_FACTOR
        ):
            explanation = f"The move is the best move among the possible variations ({pv_scores[0][1]}) and is rated much higher than the second best move {pv_scores[1][0]} ({pv_scores[1][1]})."
            return True, Tags.BRILLIANT_MOVE_TAG.value, explanation

        return False, None, None

    def good_move_check(self, scores, pv_scores):
        """
        Checks if the move is a good move based on the evaluation
        scores and principal variations.

        :param scores: Short-term and long-term evaluation scores.
        :param pv_scores: The evaluation scores for the principal variations.
        :return: True if the move is a good move, False otherwise along
                 with the tag and the explanation.
        """

        ALLOWED_DELTA = 0.7
        GOOD_MOVE = 50

        std = scores[0]
        ltd = scores[1]
        wcd = scores[7]

        if wcd < 0.1:
            return False, None, None

        """ Scenario 1:
        A good move leads to a better position and is not significantly worse 
        than the best move.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        best_move_score = pv_scores[0][1]
        if (
            std > GOOD_MOVE
            and ltd > GOOD_MOVE
            and ltd >= ALLOWED_DELTA * best_move_score
        ):
            explanation = f"Leads to a noticeably better position ({ltd}) and is not significantly worse than the best move ({best_move_score})."
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

        GOOD_MOVE = 50
        MISTAKE = -50
        BLUNDER = -150
        MISS_FACTOR = 0.6

        std = scores[0]
        ltd = scores[1]
        wcd = scores[7]

        """ Scenario 1:

        Missed opportunity to make a good move.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        best_move_score = pv_scores[0][1]
        if ltd <= MISS_FACTOR * best_move_score and best_move_score > GOOD_MOVE:
            explanation = f"Missed opportunity ({ltd}) to make a good move {pv_scores[0][0]} ({best_move_score})."
            return True, Tags.MISTAKE_TAG.value, explanation

        """ Scenario 2:
        
        A bad move leads to a slightly worse position.
        """
        if ltd < MISTAKE and ltd > BLUNDER:
            explanation = f"Leads to a slightly worse position ({ltd})."
            return True, Tags.MISTAKE_TAG.value, explanation

        return False, None, None

    def blunder_check(self, scores, pv_scores):
        """
        Checks if the move is a blunder based on the evaluation scores.

        :param scores: Short-term and long-term evaluation scores.
        :return: True if the move is a blunder, False otherwise along
                 with the tag and the explanation.
        """

        BLUNDER = -150

        std = scores[0]
        ltd = scores[1]

        """ Scenario 1:
        Move that leads to a significantly worse position.
        """

        if self._verify_pv_scores(pv_scores):
            return False, None, None

        best_move_score = pv_scores[0][1]

        if ltd <= BLUNDER and best_move_score > 0:
            explanation = f"Leads to a significantly worse position ({ltd}) with a clear better option available {pv_scores[0][0]} ({best_move_score})."
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

    def close_engine(self):
        """
        Closes the chess engine process.
        """
        self.engine.close()
