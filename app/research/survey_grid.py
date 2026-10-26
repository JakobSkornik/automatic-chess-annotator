"""Fixed factors and provenance for the commentary comparison study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MODELS = [
    {'id': 'gpt-6-astra', 'provider': 'openai', 'label': 'GPT-6 Astra'},
    {'id': 'gpt-6.1-sol', 'provider': 'openai', 'label': 'GPT-6.1 Sol'},
    {'id': 'gpt-6-luna', 'provider': 'openai', 'label': 'GPT-6 Luna'},
    {'id': 'claude-fable-5-1', 'provider': 'anthropic', 'label': 'Claude Fable 5.1'},
    {'id': 'claude-opus-5-5', 'provider': 'anthropic', 'label': 'Claude Opus 5.5'},
    {'id': 'claude-sonnet-5-5', 'provider': 'anthropic', 'label': 'Claude Sonnet 5.5'},
    {'id': 'claude-haiku-5-5', 'provider': 'anthropic', 'label': 'Claude Haiku 5.5'},
]
LEVELS = ['beginner', 'intermediate', 'expert']
ROOT = Path(__file__).resolve().parents[2]


def source_record():
    """Fingerprint live analysis code, including uncommitted architecture changes."""
    paths = sorted([*ROOT.joinpath('app/core').rglob('*.py'), *ROOT.joinpath('app/models').rglob('*.py')])
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'files': files}


def run_settings(effort='medium'):
    return {
        'reasoning_effort': effort,
        'render_mode': os.getenv('COMMENTARY_RENDER_MODE', 'grounded_rewrite'),
        'context_profile': os.getenv('COMMENTARY_CONTEXT_PROFILE', 'board_grounded'),
        'humanization_level': os.getenv('HUMANIZATION_LEVEL', '2'),
        'semantic_review': os.getenv('COMMENTARY_LLM_REVIEW_ENABLED', '0'),
        'output_token_cap': os.getenv('COMMENTARY_MAX_OUTPUT_TOKENS', '').strip() or None,
        'truncation_retries': os.getenv('COMMENTARY_TRUNCATION_RETRIES', '1'),
        'comment_side': 'both',
        'language': 'English',
    }


def variant_id(game_id, model_id, level):
    return f'{game_id}__{model_id}__{level}'


def grid_cells(definition):
    return [
        {'id': variant_id(g['id'], m['id'], level), 'game_id': g['id'], 'model': m['id'],
         'provider': m['provider'], 'level': level}
        for g in definition['games'] for m in definition['models'] for level in definition['levels']
    ]
