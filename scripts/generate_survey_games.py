"""Pre-generate a frozen survey set using the normal application pipeline.

Resumes completed games. Keeps the original PGNs, generation provenance and
unsanitized outputs private; publishing copies only participant-facing fields.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from app.core.queue_manager import queue_manager
from app.core.worker import _process_job
from app.core.output_paths import find_game_path
from app.core.commentary.llm_policy import resolve_model
from app.core.engine.engine_connector import _shutdown_global_engine


async def generate(folder: Path):
    manifest = json.loads((folder / 'selection.json').read_text(encoding='utf-8'))
    provider = os.environ.get('LLM_DEFAULT_PROVIDER', 'openai')
    model = resolve_model(provider, 'composer')
    settings = {'provider': provider, 'model': model, 'commentary_level': 'intermediate', 'comment_side': 'both'}
    previous_path = folder / 'generation.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    previous_settings = previous.get('generation', {})
    if previous_settings and any(previous_settings.get(k) != v for k, v in settings.items()):
        raise RuntimeError('Generation settings changed. Use the original settings or a new preparation folder.')
    manifest['generation'] = {**settings, 'started_at': previous_settings.get('started_at', datetime.now(timezone.utc).isoformat())}
    previous_games = {g['id']: g for g in previous.get('games', [])}
    generated_new = False
    for game in manifest['games']:
        target = folder / f"{game['id']}.json"
        pgn = (folder / f"{game['id']}.pgn").read_text(encoding='utf-8')
        if hashlib.sha256(pgn.encode()).hexdigest() != game['pgn_sha256']:
            raise RuntimeError('Selected PGN changed')
        source_game = chess.pgn.read_game(io.StringIO(pgn))
        if source_game is None or source_game.errors:
            raise RuntimeError('Selected PGN is invalid')
        if target.exists():
            data = json.loads(target.read_text(encoding='utf-8'))
            if not data.get('commentary_complete'):
                raise RuntimeError(f'Incomplete saved game: {target}')
            recorded_hash = previous_games.get(game['id'], {}).get('artifact_sha256')
            if recorded_hash and hashlib.sha256(target.read_bytes()).hexdigest() != recorded_hash:
                raise RuntimeError(f'Saved analysis changed: {target}')
            print(f"Reusing {game['id']}", flush=True)
        else:
            generated_new = True
            print(f"Generating {game['id']} ({game['bracket']}, {game['plies']} plies) using {provider}/{model}", flush=True)
            job_id = await queue_manager.add_job(pgn, llm_provider=provider, commentary_level='intermediate', comment_side='both')
            await _process_job(job_id, queue_manager.get_job_data(job_id))
            source = find_game_path(job_id)
            data = json.loads(source.read_text(encoding='utf-8'))
            if not data.get('commentary_complete'):
                raise RuntimeError('Pipeline did not complete commentary')
            target.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
        if [m['uci'] for m in data['moves']] != [m.uci() for m in source_game.mainline_moves()]:
            raise RuntimeError(f'Saved analysis does not match the selected game: {target}')
        counts = Counter(m.get('commentary_status') or 'none' for m in data['moves'])
        game['commentary_status_counts'] = dict(counts)
        game['generated_comments'] = sum(bool(m.get('comment')) for m in data['moves'])
        game['model_comments'] = sum(m.get('comment_kind') == 'llm' for m in data['moves'])
        game['artifact_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        (folder / 'generation.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Saved {game['id']}: {game['model_comments']} model comments; statuses={dict(counts)}", flush=True)
        if game['model_comments'] == 0:
            raise RuntimeError('No model-generated comments; refusing to silently publish a fallback-only game.')
    manifest['generation']['completed_at'] = (
        previous_settings.get('completed_at') if not generated_new else None
    ) or datetime.now(timezone.utc).isoformat()
    (folder / 'generation.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"All {len(manifest['games'])} selected games have completed generation.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    logging.basicConfig(filename=args.folder / 'generation.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    try:
        asyncio.run(generate(args.folder))
    finally:
        # python-chess owns a background thread; close it before interpreter exit.
        _shutdown_global_engine()
