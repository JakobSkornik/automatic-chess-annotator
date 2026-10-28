"""Generate one complete model/game/level variant from verified private engine evidence."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
from app.research.survey_grid import grid_cells, source_record, run_settings
from app.research.grid_budget import SpendLedger, BudgetExhausted
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.pipeline.game_pipeline import GameAnnotationPipeline
from app.core.engine.analysis_retriever import assemble_game_json
from app.core.engine.engine_connector import _shutdown_global_engine


async def generate(args):
    definition = json.loads((args.folder/'grid.json').read_text(encoding='utf-8'))
    if not definition.get('engine_preparation_completed_at'):
        raise ValueError('Engine preparation is incomplete')
    if source_record() != definition['architecture']:
        raise ValueError('Analysis architecture changed; prepare a new grid')
    settings = definition['settings']
    if run_settings(settings['reasoning_effort']) != settings:
        raise ValueError('Runtime configuration differs from the frozen grid')
    cell = next((c for c in grid_cells(definition) if c['id']==args.cell), None)
    if cell is None:
        raise ValueError('Unknown grid cell')
    directory = args.folder/'variants'/cell['id']
    directory.mkdir(parents=True, exist_ok=True)
    record_path, game_path = directory/'record.json', directory/'game.json'
    previous_attempts = []
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        # Earlier unfinished attempts stay on record; their traces remain in private-traces.
        previous_attempts = record.get('previous_attempts', []) + [
            {k: record.get(k) for k in ('job_id', 'status', 'error_type', 'started_at', 'finished_at')}]
        if record['status']=='complete':
            if hashlib.sha256(game_path.read_bytes()).hexdigest()!=record['artifact_sha256']:
                raise ValueError('Completed variant changed')
            print('Reusing verified '+cell['id'],flush=True)
            return
    entry = next(g for g in definition['games'] if g['id']==cell['game_id'])
    cache = args.folder/'engine'/f"{cell['game_id']}.private.pkl"
    raw = cache.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['engine_sha256']:
        raise ValueError('Engine cache changed')
    # Only the locally prepared, hash-verified cache is accepted here.
    state = pickle.loads(raw)
    state.metadata = state.metadata.model_copy(update={'id':str(uuid.uuid4())})
    os.environ[f"COMMENTARY_{cell['provider'].upper()}_MODEL"] = cell['model']
    os.environ['LOG_LLM_TO_FILE']='1'
    os.environ['LOG_LLM_DIR']=str(directory/'private-traces')
    ledger = SpendLedger(args.folder/'spend.private.sqlite3', args.budget_usd)
    ledger.install_guard(cell['id'])
    service = AdvancedCommentService(provider_key=cell['provider'])
    client = getattr(service._provider, '_client', None)  # Cursor agents have no HTTP client.
    if client is not None:
        client.max_retries = 0
    record = {**cell, 'status':'running', 'started_at':datetime.now(timezone.utc).isoformat(),
              'source_sha256':definition['architecture']['sha256'], 'engine_sha256':entry['engine_sha256'],
              'settings':settings, 'job_id':state.metadata.id,
              **({'previous_attempts':previous_attempts} if previous_attempts else {})}
    record_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    try:
        await GameAnnotationPipeline().run_llm_phases(state,service,llm_effort=settings['reasoning_effort'],
                                                     commentary_level=cell['level'],comment_side='both',
                                                     stop_at_move_index=entry['comparison_position']['move_index']
                                                     if definition.get('stop_at_comparison') else None)
        game=assemble_game_json(state).model_dump(mode='json')
        if not game['commentary_complete']:
            raise RuntimeError('Commentary did not complete')
        trace = directory/'private-traces'/f'{state.metadata.id}.jsonl'
        calls=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not calls or any(c['model']!=cell['model'] for c in calls):
            raise RuntimeError('Model provenance missing or inconsistent')
        if any(not c.get('ok') for c in calls):
            raise RuntimeError('Provider calls failed; retain traces and repair before publishing')
        if source_record()['sha256']!=record['source_sha256']:
            raise RuntimeError('Analysis source changed during generation')
        target=game['moves'][entry['comparison_position']['move_index']]
        if not target.get('comment'):
            raise RuntimeError('Selected comparison position has no comment')
        game_path.write_text(json.dumps(game,ensure_ascii=False),encoding='utf-8')
        record.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat(),
                      artifact_sha256=hashlib.sha256(game_path.read_bytes()).hexdigest(),
                      model_calls=len(calls),model_comments=sum(m.get('comment_kind')=='llm' for m in game['moves']),
                      commentary_status_counts=dict(Counter(m.get('commentary_status') or 'none' for m in game['moves'])),
                      comparison_comment_status=target.get('commentary_status'),
                      response_models=sorted({a.get('response_model') for c in calls for a in
                          ((c.get('session_context') or {}).get('request') or {}).get('attempts',[]) if a.get('response_model')}))
        print('Completed '+cell['id'],flush=True)
    except BaseException as exc:
        record.update(status='budget_exhausted' if isinstance(exc,BudgetExhausted) else 'failed',
                      error_type=type(exc).__name__,finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        if client is not None:
            await client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--cell',required=True)
    parser.add_argument('--budget-usd',type=float,required=True)
    args=parser.parse_args()
    log_dir=args.folder/'variants'/args.cell
    # Validate the identifier before using it as a directory name.
    definition=json.loads((args.folder/'grid.json').read_text(encoding='utf-8'))
    if args.cell not in {c['id'] for c in grid_cells(definition)}:
        parser.error('Unknown cell')
    log_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=log_dir/'generation.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',encoding='utf-8')
    try:
        asyncio.run(generate(args))
    except BudgetExhausted:
        print('Paused: API spending headroom exhausted.',flush=True)
        raise SystemExit(75)
    finally:
        _shutdown_global_engine()
