"""Conservative task-local API spend reservations shared by grid workers.

This tracks an upper bound, not an invoice. Cache reads are discounted only
where their rates are recorded; uncertain failed requests retain reservations.
"""
from __future__ import annotations
import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

# USD per million tokens, verified against provider docs on 2026-10-24.
# Tuple: input, output, cached input. Cursor-hosted models are billed by Cursor and not priced here.
PRICES = {
    'gpt-6-astra': (10, 50, 1), 'gpt-6.1-sol': (2, 10, .1), 'gpt-6-luna': (.1, .5, .01),
}


class BudgetExhausted(BaseException):
    """Abort the conversation instead of silently substituting budget fallbacks."""


def charge_bound(model, usage):
    inp, out, cached_rate = PRICES[model]
    tokens = usage['input_tokens']
    cached = min(tokens, usage.get('cached_tokens', 0))
    long = tokens > 272000
    return ((tokens-cached)*inp*1.25*(2 if long else 1)
            + cached*cached_rate*(2 if long else 1)
            + usage['output_tokens']*out*(1.5 if long else 1))/1e6


def reservation_bound(model, max_output_tokens):
    inp, out, _ = PRICES[model]
    # Reserve the model's whole possible input window at the long-context rate
    # before sending. This deliberately leaves headroom near the ceiling.
    return (1050000*inp*2.5 + max_output_tokens*out*1.5)/1e6


class SpendLedger:
    def __init__(self, path: Path, limit_usd: float):
        if limit_usd <= 0:
            raise ValueError('A positive API spending limit is required')
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY, limit_usd REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, model TEXT NOT NULL, cell TEXT NOT NULL,
                reserved REAL NOT NULL, charged_bound REAL, state TEXT NOT NULL, usage TEXT, created_at REAL NOT NULL);''')
            db.execute('INSERT INTO budget VALUES (1,?) ON CONFLICT(id) DO NOTHING', (limit_usd,))
            saved = db.execute('SELECT limit_usd FROM budget WHERE id=1').fetchone()[0]
            if abs(saved-limit_usd) > .000001:
                raise ValueError('Existing ledger limit differs; explicitly update the ledger budget before resuming')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def reserve(self, model, cell, maximum):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            limit = db.execute('SELECT limit_usd FROM budget WHERE id=1').fetchone()[0]
            used = db.execute('SELECT COALESCE(SUM(COALESCE(charged_bound,reserved)),0) FROM calls').fetchone()[0]
            if used+maximum > limit:
                pending = db.execute("SELECT COUNT(*) FROM calls WHERE state='pending'").fetchone()[0]
                if pending:
                    return None
                raise BudgetExhausted('Insufficient remaining budget to reserve the next request safely')
            request_id = str(uuid.uuid4())
            db.execute('INSERT INTO calls VALUES (?,?,?,?,NULL,?,NULL,?)', (request_id, model, cell, maximum, 'pending', time.time()))
            return request_id

    def settle(self, request_id, model, usage, error=False):
        known = 'input_tokens' in usage and 'output_tokens' in usage
        charge = charge_bound(model, usage) if known else None
        with self.connect() as db:
            db.execute('UPDATE calls SET charged_bound=?,state=?,usage=? WHERE id=?',
                       (charge, 'uncertain' if error or not known else 'complete', json.dumps(usage), request_id))

    def summary(self):
        with self.connect() as db:
            return {'limit_usd': db.execute('SELECT limit_usd FROM budget WHERE id=1').fetchone()[0],
                    'committed_upper_bound_usd': db.execute('SELECT COALESCE(SUM(COALESCE(charged_bound,reserved)),0) FROM calls').fetchone()[0],
                    'requests': db.execute('SELECT COUNT(*) FROM calls').fetchone()[0],
                    'pending': db.execute("SELECT COUNT(*) FROM calls WHERE state='pending'").fetchone()[0],
                    'uncertain': db.execute("SELECT COUNT(*) FROM calls WHERE state='uncertain'").fetchone()[0]}

    def mark_uncertain(self, cell):
        with self.connect() as db:
            db.execute("UPDATE calls SET state='uncertain' WHERE cell=? AND state='pending'", (cell,))

    def install_guard(self, cell):
        from app.core.commentary import llm_runtime, llm_providers, commentary_session
        original = llm_runtime.call_with_truncation_retry

        async def guarded_retry(create, kwargs, *, provider, usage_total, trace):
            async def guarded_create(**request):
                model = request['model']
                cap = request.get('max_tokens', request.get('max_output_tokens'))
                if not isinstance(cap, int):
                    raise ValueError('Grid calls require an explicit output cap')
                maximum = reservation_bound(model, cap)
                request_id = self.reserve(model, cell, maximum)
                while request_id is None:
                    await asyncio.sleep(2)
                    request_id = self.reserve(model, cell, maximum)
                try:
                    result = await create(**request)
                except BaseException:
                    self.settle(request_id, model, {}, error=True)
                    raise
                self.settle(request_id, model, llm_runtime.usage_breakdown(result))
                return result
            return await original(guarded_create, kwargs, provider=provider, usage_total=usage_total, trace=trace)
        llm_providers.call_with_truncation_retry = guarded_retry
        commentary_session.call_with_truncation_retry = guarded_retry
