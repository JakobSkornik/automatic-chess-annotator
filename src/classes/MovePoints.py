import chess


class MovePoints:

    # Measured in expected points [0, 1]
    short_term_diff = 0
    long_term_diff = 0
    short_term_ep = 0
    long_term_ep = 0
    prev_short_term_ep = 0
    prev_long_term_ep = 0
    win_chance_white = 0
    win_chance_black = 0
    stm = chess.WHITE

    def __init__(self, std, ltd, ste, lte, pste, plte, wcw, wcb, stm) -> None:
        self.short_term_diff = std
        self.long_term_diff = ltd
        self.short_term_ep = ste
        self.long_term_ep = lte
        self.prev_short_term_ep = pste
        self.prev_long_term_ep = plte
        self.win_chance_white = wcw
        self.win_chance_black = wcb
        self.stm = stm
