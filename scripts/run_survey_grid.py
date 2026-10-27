"""Run the frozen full-game grid with bounded concurrency and a shared API spend cap."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.research.survey_grid import grid_cells, source_record
from app.research.grid_budget import SpendLedger


async def run(args):
    definition=json.loads((args.folder/'grid.json').read_text(encoding='utf-8'))
    if source_record()!=definition['architecture']:
        raise ValueError('Analysis architecture changed; prepare a new grid')
    ledger=SpendLedger(args.folder/'spend.private.sqlite3',args.budget_usd)
    cells=grid_cells(definition)
    if args.cells:
        if not set(args.cells)<={c['id'] for c in cells}:
            raise ValueError('Unknown requested cell')
        cells=[c for c in cells if c['id'] in args.cells]
    pending=list(cells)
    active={}
    retries={}
    failed=False
    def next_cell():
        # With --per-provider, each provider's rate limit is shared by at most N
        # of its own cells, so different providers can run side by side.
        for cell in pending:
            running=sum(c['provider']==cell['provider'] for _,c in active.values())
            if args.per_provider is None or running<args.per_provider:
                pending.remove(cell)
                return cell
        return None
    async def launch(cell):
        process=await asyncio.create_subprocess_exec(sys.executable,str(Path(__file__).with_name('generate_grid_cell.py')),
            str(args.folder),'--cell',cell['id'],'--budget-usd',str(args.budget_usd))
        active[asyncio.create_task(process.wait())]=(process,cell)
        print('Running '+cell['id'],flush=True)
    try:
        while True:
            while not failed and len(active)<args.workers:
                cell=next_cell()
                if cell is None: break
                await launch(cell)
            if not active: break
            done,_=await asyncio.wait(active,return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                _,cell=active.pop(task)
                if task.result()!=0:
                    ledger.mark_uncertain(cell['id'])
                    retries[cell['id']]=retries.get(cell['id'],0)+1
                    if retries[cell['id']]<=args.retry_failed:
                        # Regenerate the whole variant later with a fresh conversation.
                        pending.append(cell)
                        print(f"Requeued {cell['id']} after exit {task.result()} (retry {retries[cell['id']]}/{args.retry_failed})",flush=True)
                        continue
                    failed=True
                    print(f"Stopped scheduling after {cell['id']} exited {task.result()}; completed variants remain saved.",flush=True)
    finally:
        for task,(process,cell) in active.items():
            if process.returncode is None:
                process.terminate()
                await process.wait()
            ledger.mark_uncertain(cell['id'])
    print(json.dumps(ledger.summary(),indent=2),flush=True)
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--budget-usd',required=True,type=float)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,9))
    parser.add_argument('--cells',nargs='+',help='Optional calibration subset from the frozen grid')
    parser.add_argument('--retry-failed',type=int,default=0,choices=range(0,4),
                        help='Regenerate a failed variant from scratch up to N times before stopping')
    parser.add_argument('--per-provider',type=int,choices=range(1,9),
                        help='Also cap concurrent cells per provider (e.g. --workers 2 --per-provider 1)')
    args=parser.parse_args()
    lock=args.folder/'generation.lock'
    try:
        with lock.open('x',encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit('Another grid run owns generation.lock. Check that process before removing a stale lock.')
    try:
        asyncio.run(run(args))
    finally:
        lock.unlink(missing_ok=True)
