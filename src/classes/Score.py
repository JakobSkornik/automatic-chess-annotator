class Score:
    short_term_info = None
    long_term_info = None
    prev_sterm_info = None
    prev_lterm_info = None

    def __init__(self, st_wdl, lt_wdl, pst, plt) -> None:
        self.short_term_info = st_wdl
        self.long_term_info = lt_wdl
        self.prev_sterm_info = pst
        self.prev_lterm_info = plt
