import chess
import chess.engine
import chess.pgn
import chess.svg
from chevy.features import KingSafety, PawnStructure, BoardFeatures
from IPython.display import display, SVG

from src.features.chevy import (
    CHEVY_BOARD_FEATURES,
    CHEVY_KING_FEATURES,
    CHEVY_PAWN_FEATURES,
)

MAX_POSITIONAL_FEAUTRES = 5


class ChessGame:
    """Class to represent a chess game and interact with it programmatically."""

    def __init__(
        self,
        pgn_path,
        game_number=1,
        engine_path=r"stockfish.exe",
    ):
        self.engine_path = engine_path
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self._load_game(pgn_path, game_number)

        self.king_features = CHEVY_KING_FEATURES
        self.pawn_features = CHEVY_PAWN_FEATURES
        self.board_features = CHEVY_BOARD_FEATURES

    def simulate_game(self, display_board=False, print_evaluation=False, n_moves=None):
        """Simulate the entire game and optionally display the board after each move."""

        self.board = self.game.board()
        for i, _ in enumerate(self.game.mainline_moves()):
            if n_moves and i == n_moves:
                break

            self.get_nth_move(i + 1, display_board, print_evaluation)

    def get_nth_move(self, n, display_board=False, print_evaluation=False):
        """Retrieve the n-th move and optionally display the board."""

        self.board = self.game.board()
        last_move = None
        color = chess.WHITE if n % 2 == 1 else chess.BLACK

        for i, move in enumerate(self.game.mainline_moves()):
            self.board.push(move)
            if i == n - 1:
                last_move = move

                if display_board:
                    self._display_board(last_move)

                if print_evaluation:
                    print(f"Move {n}: {move.uci()}")
                    self._print_move_evaluation(color)

                break

        return last_move

    def _load_game(self, pgn_path, n=1):
        """Load the n-th game from a PGN file."""

        with open(pgn_path) as pgn_file:
            current_game = None
            for _ in range(n):
                current_game = chess.pgn.read_game(pgn_file)
                if current_game is None:
                    raise ValueError("No such game number exists in the PGN file.")
            self.game = current_game
            self.board = self.game.board()

            print(
                f"Loading game {n}.\nRound: {current_game.headers['Round']}\nWhite: {current_game.headers['White']}\nBlack: {current_game.headers['Black']}"
            )

    def _print_move_evaluation(self, color):
        """Evaluate and print the engine's evaluation of the current board state."""

        info = self.engine.analyse(self.board, chess.engine.Limit(time=0.1))
        score = info["score"].relative
        print(f"""Evaluation for {"WHITE" if color else "BLACK"}: {score}""")

        self._print_features(color)

    def _print_features(self, color):
        """Print the all positional features for the current board state."""

        king_features = KingSafety(self.board, color=color)
        pawn_features = PawnStructure(self.board, color=color)
        board_features = BoardFeatures(self.board, color=color)

        feature_values = []
        for feature in self.king_features:
            value = getattr(king_features, feature)
            feature_values.append((feature, value))

        for feature in self.pawn_features:
            value = getattr(pawn_features, feature)
            feature_values.append((feature, value))

        for feature in self.board_features:
            value = getattr(board_features, feature)
            feature_values.append((feature, value))

        numerical_features = []
        for feature in feature_values:
            if isinstance(feature[1], int):
                numerical_features.append((feature[0], int(feature[1])))
            elif isinstance(feature[1], list):
                array_sum = 0
                for i, value in enumerate(feature[1]):
                    array_sum += value
                numerical_features.append((f"{feature[0]}", array_sum))

        print("\nPositional features:")
        for feature, value in numerical_features:
            print(f"\t{feature}: {value}")

    def _display_board(self, last_move=None):
        """Display the current board state with the last move highlighted."""

        if last_move:
            display(
                SVG(
                    chess.svg.board(
                        board=self.board,
                        size=350,
                        arrows=[(last_move.from_square, last_move.to_square)],
                        colors={"arrow": "#f00"},
                    )
                )
            )
        else:
            display(SVG(chess.svg.board(board=self.board, size=350)))

    def __del__(self):
        """Close the engine properly on deletion of the object."""

        self.engine.quit()
