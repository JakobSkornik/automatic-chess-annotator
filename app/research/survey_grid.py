"""Fixed factors and provenance for the commentary comparison study."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# 'provider' is the model-policy key passed to make_llm_provider: Grok and Composer
# are both Cursor-hosted and share the Cursor agent transport.
MODELS = [
    {'id': 'gpt-6-luna', 'provider': 'openai', 'label': 'GPT-6 Luna'},
    {'id': 'grok-4.6', 'provider': 'grok', 'label': 'Grok 4.6'},
    {'id': 'composer-2.5', 'provider': 'cursor', 'label': 'Composer 2.5'},
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


def game_levels(definition, game):
    """A game with a fixed ``level`` is written at that level only; otherwise at every level."""
    return [game['level']] if game.get('level') else definition['levels']


# How a version of the version study is written (the 'writer' of an entry in 'models').
WRITERS = ('full', 'lines_only', 'template')


def versions(model):
    """The three versions of the version study, all on one language model: P the full system,
    T the template without a model, B the model without the statements of the rules."""
    base = {'provider': model['provider'], 'llm': model['id']}
    return [{'id': 'P', 'label': 'Full system', 'writer': 'full', **base},
            {'id': 'T', 'label': 'Template without model', 'writer': 'template', **base},
            {'id': 'B', 'label': 'Model without claims', 'writer': 'lines_only', **base}]


def grid_cells(definition):
    """One cell per game, model (or version) and level. ``llm`` is the language model that
    writes the cell and ``writer`` how; for a model comparison they are the model itself and
    the full system."""
    return [
        {'id': variant_id(g['id'], m['id'], level), 'game_id': g['id'], 'model': m['id'],
         'provider': m['provider'], 'level': level,
         **({'llm': m['llm'], 'writer': m['writer']} if m.get('writer') else {})}
        for g in definition['games'] for m in definition['models'] for level in game_levels(definition, g)
    ]
