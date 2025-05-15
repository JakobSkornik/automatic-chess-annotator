class Wdl:
    white_win_chance = 0
    white_score_rate = 0
    black_win_chance = 0
    black_score_rate = 0

    def __init__(self, wwc, wsr, bwc, bsr) -> None:
        self.white_win_chance = wwc
        self.white_score_rate = wsr
        self.black_win_chance = bwc
        self.black_score_rate = bsr
