import cairosvg
import chess
import io
import matplotlib.pyplot as plt
from chess import Move
from chess.pgn import Game
from IPython.display import display, SVG
from PIL import Image


def display_board(fen: str, last_move: Move = None):
    """Display the current board state with the last move highlighted."""
    board = chess.Board(fen)

    if last_move:
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
