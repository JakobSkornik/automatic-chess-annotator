import cairosvg
import chess
import io
import matplotlib.pyplot as plt
from chess import Move
from chess.pgn import Game
from IPython.display import display, SVG
from PIL import Image


class Visualizer:
    """Visualizer class to visualize the moves and boards."""

    def visualize_board(self, game: Game, move: Move):
        board = game.board()
        before = game.board().copy()
        before.pop()

        before_svg = chess.svg.board(
            board=before,
            size=350,
            colors={"square lastmove": "#ffce9e", "square": "#d18b47"},
        )
        after_svg = chess.svg.board(
            board=board,
            size=350,
            arrows=[(move.from_square, move.to_square)],
            colors={"square lastmove": "#ffce9e", "square": "#d18b47"},
        )
        before_bytes = cairosvg.svg2png(bytestring=before_svg.encode("utf-8"))
        after_bytes = cairosvg.svg2png(bytestring=after_svg.encode("utf-8"))

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(Image.open(io.BytesIO(before_bytes)))
        ax[0].axis("off")
        ax[0].set_title("Before")

        ax[1].imshow(Image.open(io.BytesIO(after_bytes)))
        ax[1].axis("off")
        ax[1].set_title("After")

        plt.show()

    def display_board(
        self, board, last_move: Move = None, display_arrows: bool = False
    ):
        """Display the current board state with the last move highlighted."""

        if last_move:
            arrows = []
            if display_arrows:
                arrows = [(last_move.from_square, last_move.to_square)]

            display(
                SVG(
                    chess.svg.board(
                        board=board,
                        size=350,
                        arrows=arrows,
                        colors={"arrow": "#f00"},
                    )
                )
            )
        else:
            display(SVG(chess.svg.board(board=board, size=350)))
