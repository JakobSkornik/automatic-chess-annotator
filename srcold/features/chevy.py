CHEVY_NUM_FEATURES = 17

CHEVY_KING_FEATURES = [
    "king_mobility",
    "king_centrality",
    "king_attackers_looking_at_ring_1",
    "king_defenders_at_ring_1",
    "checked",
    "castling_rights",
]

CHEVY_PAWN_FEATURES = [
    "passed_pawns",
    "isolated_pawns",
    "blocked_pawns",
    "central_pawns",
]

CHEVY_BOARD_FEATURES = [
    "bishop_pair",
    "fianchetto_queen",
    "fianchetto_king",
    "queens_mobility",
    "open_files_rooks_count",
    "connected_rooks",
    "connectivity",
]

CHEVY_NORM_FACTOR = {
    "king_mobility": 8,
    "king_centrality": 3,
    "king_attackers_looking_at_ring_1": 8,
    "king_defenders_at_ring_1": 6,
    "checked": 1,
    "castling_rights": 1,
    "passed_pawns": 6,
    "isolated_pawns": 7,
    "blocked_pawns": 8,
    "central_pawns": 3,
    "bishop_pair": 1,
    "fianchetto_queen": 1,
    "fianchetto_king": 1,
    "queens_mobility": 80,
    "open_files_rooks_count": 3,
    "connected_rooks": 1,
    "connectivity": 36,
}
