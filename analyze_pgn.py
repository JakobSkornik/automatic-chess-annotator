import os

from src.pgn_reader import PGNReader
from src.move_evaluator import MoveEvaluator

# from src.annotator import Annotator
# from src.output_writer import OutputWriter


def main():
    pgn_file = "data/1.pgn"

    # Step 1: Read PGN file
    reader = PGNReader(pgn_file)
    game = reader.read_game()

    # Step 2: Evaluate moves
    evaluator = MoveEvaluator()
    evaluated_moves = evaluator.evaluate_game(game)

    # # Step 3: Annotate game
    # annotator = Annotator(evaluated_moves)
    # annotated_game = annotator.annotate_game()

    # # Step 4: Write output
    # writer = OutputWriter()
    # writer.write_annotated_game(annotated_game, "output/annotated_game.pgn")

    stop()


def stop():
    os._exit(0)


if __name__ == "__main__":
    main()
