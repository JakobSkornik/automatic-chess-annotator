import chess
from chess.engine import SimpleEngine
from tabulate import tabulate
from openai import OpenAI

from chevy.features import KingSafety, PawnStructure, BoardFeatures

from config import CHAT_GPT_KEY, CHAT_GPT_PROMPT
from src.features.comments import BASE_COMMENTS
from src.features.chevy import (
    CHEVY_NUM_FEATURES,
    CHEVY_BOARD_FEATURES,
    CHEVY_KING_FEATURES,
    CHEVY_PAWN_FEATURES,
    CHEVY_NORM_FACTOR,
)
from src.utils.pgn_manager import PgnManager
from src.utils.visualizer import Visualizer


class Simulator:
    """Simulator class to run the simulation"""

    def __init__(self, pgn_path, game_number=1, engine_path=r"stockfish.exe"):
        self.engine_path = engine_path
        self.engine = SimpleEngine.popen_uci(self.engine_path)
        self.client = OpenAI(api_key=CHAT_GPT_KEY)

        self.pgn_manager = PgnManager(self.engine)
        self.game = self.pgn_manager.load_nth_pgn(pgn_path, game_number)

        self.king_features = CHEVY_KING_FEATURES
        self.pawn_features = CHEVY_PAWN_FEATURES
        self.board_features = CHEVY_BOARD_FEATURES
        self.chevy_normalization = CHEVY_NORM_FACTOR

        self.visualizer = Visualizer()

    def get_nth_move(
        self, n, display_board=False, print_evaluation=False, get_chatgpt_comment=False
    ):
        """Retrieve the n-th move and optionally display the board."""

        if n < 1:
            print("Invalid move number. Defaulting to move 1.")
            n = 1

        self.board = self.game.board()
        self.color = "WHITE" if n % 2 == 1 else "BLACK"
        self.move_number = n

        for i, move in enumerate(self.game.mainline_moves()):
            self.move = move
            self.board.push(move)

            if i == n - 1:
                print(f"Move #{self.move_number} for {self.color}: {move}")

                if display_board:
                    self.visualizer.display_board(self.board, move, display_arrows=True)

                if print_evaluation:
                    eval = self._print_move_evaluation()

                    if get_chatgpt_comment:
                        self._get_chatgpt_comment(eval)

                break

        return move

    def simulate_game(self, display_board=False, print_evaluation=False, n_moves=None):
        """Simulate the entire game and optionally display the board after each move."""

        board = self.game.board()
        for i, _ in enumerate(self.game.mainline_moves()):
            if n_moves and i == n_moves:
                break

            self.get_nth_move(i + 1, display_board, print_evaluation)

    def _print_move_evaluation(self):
        """Evaluate and print the engine's evaluation of the current board state."""

        info = self.engine.analyse(self.board, chess.engine.Limit(time=0.5))
        score = info["score"].relative
        print(f"""Evaluation for {self.color}: {score}""")

        return self._print_features()

    def _print_features(self):
        """Print the all positional features for the current board state."""

        after = self.board
        before = self.board.copy()
        before.pop()

        before_features = self._get_features(before)
        after_features = self._get_features(after)

        table = []
        for i in range(CHEVY_NUM_FEATURES):
            white_diff = after_features[0][i][1] - before_features[0][i][1]
            black_diff = after_features[1][i][1] - before_features[1][i][1]
            feature = before_features[0][i][0]
            table.append(
                (
                    feature,
                    None,
                    before_features[0][i][1],
                    after_features[0][i][1],
                    white_diff,
                    None,
                    before_features[1][i][1],
                    after_features[1][i][1],
                    black_diff,
                )
            )

        headers = [
            "Feature",
            "WHITE",
            "Before",
            "After",
            "Diff",
            "BLACK",
            "Before",
            "After",
            "Diff",
        ]
        features_table = tabulate(table, headers=headers, tablefmt="outline")
        print(features_table)
        print("\n========================================\n")

        return table

    def _get_chatgpt_comment(self, table):
        column = 4 if self.color == "WHITE" else 8
        nonzero_features = [[row[0], row[column]] for row in table if row[column] != 0]
        comments = []
        for feature in nonzero_features:
            change = "positive" if feature[1] > 0 else "negative"
            comments.append(BASE_COMMENTS[feature[0]][change])
            print(f"{BASE_COMMENTS[feature[0]][change]}\n")

        messages = [
            {
                "role": "system",
                "content": CHAT_GPT_PROMPT
                + " Opening: "
                + self.game.headers["Opening"],
            }
        ]

        msg = (
            f"Move [{self.move}] number {self.move_number} for {self.color} Comments: "
            + " ".join(comments)
        )
        messages.append({"role": "user", "content": msg})

        chat = self.client.chat.completions.create(model="gpt-4", messages=messages)
        reply = chat.choices[0].message.content
        print(f"ChatGPT comment: {reply}\n")

    def _get_features(self, board):
        king_features_white = KingSafety(board, color=chess.WHITE)
        pawn_features_white = PawnStructure(board, color=chess.WHITE)
        board_features_white = BoardFeatures(board, color=chess.WHITE)

        king_features_black = KingSafety(board, color=chess.BLACK)
        pawn_features_black = PawnStructure(board, color=chess.BLACK)
        board_features_black = BoardFeatures(board, color=chess.BLACK)

        feature_values_white = []
        feature_values_black = []
        for feature in self.king_features:
            white_value = getattr(king_features_white, feature)
            feature_values_white.append((feature, white_value))

            black_value = getattr(king_features_black, feature)
            feature_values_black.append((feature, black_value))

        for feature in self.pawn_features:
            white_value = getattr(pawn_features_white, feature)
            feature_values_white.append((feature, white_value))

            black_value = getattr(pawn_features_black, feature)
            feature_values_black.append((feature, black_value))

        for feature in self.board_features:
            white_value = getattr(board_features_white, feature)
            feature_values_white.append((feature, white_value))

            black_value = getattr(board_features_black, feature)
            feature_values_black.append((feature, black_value))

        numerical_features_white = []
        for feature in feature_values_white:
            f = feature[0]

            if isinstance(feature[1], int):
                val = int(feature[1]) / self.chevy_normalization[f]
                numerical_features_white.append((f, val))
            elif isinstance(feature[1], list):
                array_sum = 0
                for _, value in enumerate(feature[1]):
                    array_sum += value
                val = array_sum / self.chevy_normalization[f]
                numerical_features_white.append((f, val))

        numerical_features_black = []
        for feature in feature_values_black:
            f = feature[0]

            if isinstance(feature[1], int):
                val = int(feature[1]) / self.chevy_normalization[f]
                numerical_features_black.append((f, val))
            elif isinstance(feature[1], list):
                array_sum = 0
                for _, value in enumerate(feature[1]):
                    array_sum += value
                val = array_sum / self.chevy_normalization[f]
                numerical_features_black.append((f, val))

        return [numerical_features_white, numerical_features_black]

    def __del__(self):
        """Close the engine properly on deletion of the object."""

        self.engine.quit()
