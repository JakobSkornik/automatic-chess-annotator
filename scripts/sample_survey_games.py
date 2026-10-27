"""Reservoir-sample one game per rating band from a bounded public PGN stream.

The population is the first --scan-games records, not the complete monthly dump.
Only standard, completed games of 30–100 plies with two known ratings qualify.
Human annotations are stripped before the annotator receives the selected games.
"""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import chess.pgn
import requests
import zstandard

BANDS = [(0, 800), *[(n, n + 200) for n in range(800, 2400, 200)], (2400, 4001)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scan-games', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    if (args.output / 'selection.json').exists():
        raise SystemExit('Selection already exists; choose a new output directory.')
    rng = random.Random(args.seed)
    counts = [0] * 10
    selected = {}
    with requests.get(args.url, stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with zstandard.ZstdDecompressor().stream_reader(response.raw) as reader:
            stream = io.TextIOWrapper(reader, encoding='utf-8')
            for record in range(args.scan_games):
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                if (record + 1) % 1000 == 0:
                    print(f'Scanned {record + 1}; eligible counts: {counts}', flush=True)
                h = game.headers
                if game.errors or h.get('Variant', 'Standard') != 'Standard':
                    continue
                if h.get('Result') not in ('1-0', '0-1', '1/2-1/2') or any(h.get(k) == 'BOT' for k in ('WhiteTitle', 'BlackTitle')):
                    continue
                try:
                    white, black = int(h['WhiteElo']), int(h['BlackElo'])
                except (KeyError, ValueError):
                    continue
                if not 100 <= min(white, black) <= max(white, black) <= 4000:
                    continue
                plies = sum(1 for _ in game.mainline_moves())
                if not 30 <= plies <= 100 or game.board().fen() != chess.STARTING_FEN:
                    continue
                mean = (white + black) / 2
                band = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= mean < hi)
                counts[band] += 1
                if rng.randrange(counts[band]) == 0:
                    pgn = game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
                    selected[band] = {'pgn': pgn, 'average_rating': mean, 'source': h.get('Site'), 'plies': plies}
    if len(selected) != 10:
        raise SystemExit(f'Not all bands found: {counts}; increase --scan-games.')
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, (lo, hi) in enumerate(BANDS):
        entry = selected[i]
        pgn = entry.pop('pgn')
        game_id = f'game-{i+1:02d}'
        (args.output / f'{game_id}.pgn').write_text(pgn, encoding='utf-8')
        entries.append({'id': game_id, 'bracket': 'Under 800' if i == 0 else '2400+' if i == 9 else f'{lo}–{hi-1}',
                        'min_rating': lo, 'max_rating_exclusive': hi, 'eligible_count': counts[i],
                        'pgn_sha256': hashlib.sha256(pgn.encode()).hexdigest(), **entry})
    manifest = {'source_url': args.url, 'source_license': 'CC0', 'rating_system': 'Lichess ratings (Glicko-2), not FIDE Elo',
                'sampling': 'Seeded reservoir sample per average-player-rating band within the first scanned records; standard completed games, 30–100 plies, no titled bots, known ratings.',
                'seed': args.seed, 'records_scanned': record + 1, 'games': entries}
    (args.output / 'selection.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved ten games to {args.output}', flush=True)


if __name__ == '__main__':
    main()
