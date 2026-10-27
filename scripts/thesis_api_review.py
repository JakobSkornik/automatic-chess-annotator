"""Run real HTTP API routes and worker in an isolated review output directory."""
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
if os.environ.get('RAG_BM25_PATH'):
    os.environ['RAG_BM25_PATH'] = str((ROOT / os.environ['RAG_BM25_PATH']).resolve())
OUT = ROOT / 'data' / 'experiments' / 'thesis_review_20260905'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUT)

import httpx
from fastapi import FastAPI
from app.routes.jobs import router
from app.core.worker import analysis_worker
from app.core.engine.engine_connector import get_global_engine_connector

CASES = {
    'pawn_endgame': '[Event "Pawn endgame review"]\n[SetUp "1"]\n[FEN "8/4k3/8/4K3/4P3/8/8/8 w - - 0 1"]\n[Result "*"]\n\n1. Kd5 Kd7 2. e5 Ke7 3. e6 Ke8 4. Kd6 Kd8 5. e7+ Ke8 6. Ke6 *',
    'opera': '[Event "Opera review"]\n[White "Morphy"]\n[Black "Duke and Count"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0',
    'scholar': '[Event "Short tactical review"]\n[Result "1-0"]\n\n1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0',
    'quiet_opening': '[Event "Quiet positional review"]\n[Result "*"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 8. Bd3 Bb7 9. O-O Nbd7 10. Rc1 c5 *',
}

async def main():
    logging.basicConfig(level=logging.WARNING)
    app = FastAPI()
    app.include_router(router)
    worker = asyncio.create_task(analysis_worker())
    summary_path = OUT / 'run_summary.json'
    results = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else []
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://review') as client:
            for name, pgn in CASES.items():
                if len(sys.argv) > 1 and name not in sys.argv[1:]:
                    continue
                started = time.monotonic()
                r = await client.post('/jobs/submit', json={'pgn_string': pgn, 'llm_provider': 'openai', 'commentary_level': 'intermediate'})
                r.raise_for_status()
                job = r.json()
                job_id = job.get('job_id') or job.get('id')
                print(json.dumps({'case': name, 'submitted': job}), flush=True)
                while time.monotonic() - started < 1200:
                    status = (await client.get(f'/jobs/{job_id}/status')).json()
                    if status['status'] in ('completed', 'failed'):
                        break
                    await asyncio.sleep(1)
                game = await client.get(f'/jobs/{job_id}/game')
                record = {'case': name, 'status': status, 'seconds': round(time.monotonic()-started, 2), 'game_http_status': game.status_code}
                if game.status_code == 200:
                    (OUT / f'{name}.json').write_text(json.dumps(game.json(), indent=2), encoding='utf-8')
                    export = await client.get(f'/jobs/{job_id}/pgn')
                    (OUT / f'{name}.pgn').write_text(export.text, encoding='utf-8')
                results.append(record)
                (OUT / 'run_summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                print(json.dumps(record), flush=True)
    finally:
        worker.cancel()
        await worker
        get_global_engine_connector().close()

if __name__ == '__main__':
    asyncio.run(main())
