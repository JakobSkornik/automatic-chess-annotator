"""Blinded model comparisons with balanced level assignment and best/worst reasons."""
from __future__ import annotations

import json
import random
import secrets
import uuid
from contextlib import contextmanager
from typing import Literal

from pydantic import Field

from app import survey
from app.models.GameJson import GameJson
from app.research.survey_grid import LEVELS, grid_cells


class ComparisonReview(survey.StrictModel):
    game_id: str = Field(pattern=r'^game-\d{2}$')
    decision: Literal['ranked', 'similar', 'cannot_judge', 'skipped']
    best: str | None = None
    worst: str | None = None
    best_reason: str = Field(default='', max_length=1500)
    worst_reason: str = Field(default='', max_length=1500)
    feedback: str = Field(default='', max_length=1500)


class ComparisonSave(survey.StrictModel):
    version: int = Field(ge=0)
    reviews: list[ComparisonReview] = Field(max_length=10)
    feedback: str = Field(default='', max_length=4000)
    completed: bool = False


@contextmanager
def database():
    with survey.database() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS comparison_variants (
                survey_id TEXT NOT NULL REFERENCES surveys(id), variant_id TEXT NOT NULL,
                game_id TEXT NOT NULL, model TEXT NOT NULL, level TEXT NOT NULL,
                game TEXT NOT NULL, provenance TEXT NOT NULL,
                PRIMARY KEY (survey_id,variant_id)
            );
            CREATE TABLE IF NOT EXISTS comparison_responses (
                id TEXT PRIMARY KEY, survey_id TEXT NOT NULL REFERENCES surveys(id),
                token_hash TEXT UNIQUE NOT NULL, rating TEXT NOT NULL, assignment TEXT NOT NULL,
                reviews TEXT NOT NULL DEFAULT '[]', feedback TEXT NOT NULL DEFAULT '',
                version INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, completed_at TEXT
            );
        ''')
        yield db


def _study(db, token, *, admin=False):
    row = survey._survey(db, token, admin=admin)
    if json.loads(row['manifest']).get('kind') != 'comparison_grid':
        raise survey.SurveyError('Comparison study not found.', 404)
    return row


def freeze_grid(manifest, variants):
    games = manifest.get('games', [])
    models = manifest.get('models', [])
    if (manifest.get('kind') != 'comparison_grid' or manifest.get('questionnaire_version') != 2
            or len(games) != 10 or [g['bracket'] for g in games] != survey.BRACKETS
            or len({g['id'] for g in games}) != 10 or manifest.get('levels') != LEVELS
            or not 2 <= len(models) <= 7 or len({m['id'] for m in models}) != len(models)):
        raise ValueError('Invalid comparison grid definition')
    expected = {c['id']: c for c in grid_cells(manifest)}
    if set(expected) != set(variants):
        raise ValueError('Every model/game/level cell must be present before publication')
    entries = {g['id']: g for g in games}
    moves_by_game = {}
    frozen = {}
    for key, cell in expected.items():
        variant = variants[key]
        if any(variant[k] != cell[k] for k in ('game_id', 'model', 'level')):
            raise ValueError('Variant factors do not match the grid')
        game = GameJson.model_validate(variant['game']).model_dump(mode='json')
        entry = entries[cell['game_id']]
        if not game['commentary_complete'] or game['metadata']['commentary_level'] != cell['level']:
            raise ValueError('Incomplete or incorrectly labelled variant')
        ratings = [game['metadata'].get(k) for k in ('whiteElo', 'blackElo')]
        if None in ratings or not entry['min_rating'] <= sum(ratings) / 2 < entry['max_rating_exclusive']:
            raise ValueError('Variant rating band mismatch')
        mainline = [(m['uci'], m['fen']) for m in game['moves']]
        if moves_by_game.setdefault(cell['game_id'], mainline) != mainline:
            raise ValueError('Variants must describe the same game positions')
        position = entry['comparison_position']
        index = position['move_index']
        if not 0 <= index < len(game['moves']) or game['moves'][index]['fen'] != position['fen_after']:
            raise ValueError('Comparison position mismatch')
        if not str(game['moves'][index].get('comment') or '').strip():
            raise ValueError('Comparison position has no comment')
        for move in game['moves']:
            move.pop('debug', None)
        game['metadata']['id'] = cell['game_id']
        frozen[key] = {k: cell[k] for k in ('game_id', 'model', 'level')}
        frozen[key].update(game=game, provenance=variant.get('provenance', {}))
    return frozen


def publish_grid(manifest, variants):
    manifest = {**manifest, 'kind': 'comparison_grid', 'questionnaire_version': 2,
                'title': manifest.get('title', 'Chess commentary comparison')}
    frozen = freeze_grid(manifest, variants)
    public, admin = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    definition = {'id': str(uuid.uuid4()), 'public_hash': survey._digest(public),
                  'admin_hash': survey._digest(admin), 'manifest': manifest, 'games': {},
                  'created_at': survey._now(), 'variants': frozen}
    install_definition(definition)
    return {'id': definition['id'], 'participant_token': public, 'results_token': admin}


def exported_variants(db, survey_id):
    return {r['variant_id']: {'game_id': r['game_id'], 'model': r['model'], 'level': r['level'],
                             'game': json.loads(r['game']), 'provenance': json.loads(r['provenance'])}
            for r in db.execute('SELECT * FROM comparison_variants WHERE survey_id=?', (survey_id,))}


def install_definition(definition):
    frozen = freeze_grid(definition['manifest'], definition['variants'])
    if definition['games'] != {}:
        raise ValueError('Comparison game data belongs in variants')
    with database() as db, db:
        db.execute('BEGIN IMMEDIATE')
        existing = db.execute('SELECT * FROM surveys WHERE id=?', (definition['id'],)).fetchone()
        if existing:
            if (any(existing[k] != definition[k] for k in ('public_hash', 'admin_hash', 'created_at'))
                    or json.loads(existing['manifest']) != definition['manifest']
                    or exported_variants(db, definition['id']) != frozen):
                raise ValueError('Existing frozen comparison study differs from bundle')
            return 0
        db.execute('INSERT INTO surveys VALUES (?,?,?,?,?,?)', (
            definition['id'], definition['public_hash'], definition['admin_hash'],
            json.dumps(definition['manifest'], ensure_ascii=False), '{}', definition['created_at']))
        db.executemany('INSERT INTO comparison_variants VALUES (?,?,?,?,?,?,?)', [
            (definition['id'], key, v['game_id'], v['model'], v['level'],
             json.dumps(v['game'], ensure_ascii=False), json.dumps(v['provenance'], ensure_ascii=False))
            for key, v in frozen.items()])
    return 1


def public_manifest(token):
    with database() as db:
        manifest = json.loads(_study(db, token)['manifest'])
    return {'title': 'Chess commentary comparison', 'round_count': 10,
            'candidate_count': len(manifest['models']), 'rating_brackets': survey.BRACKETS,
            'levels': LEVELS, 'review_scope': manifest.get('review_scope', 'position')}


def _response(row):
    assignment = json.loads(row['assignment'])
    return {'id': row['id'], 'rating': row['rating'], 'version': row['version'],
            'completed': row['completed_at'] is not None, 'reviews': json.loads(row['reviews']),
            'feedback': row['feedback'], 'rounds': [
                {'game_id': r['game_id'], 'level': r['level'],
                 'candidates': [{'id': c['id'], 'label': chr(65 + i)} for i, c in enumerate(r['candidates'])]}
                for r in assignment]}


def start_response(token, request: survey.StartRequest):
    if request.rating not in survey.BRACKETS + ['Unrated / prefer not to say']:
        raise survey.SurveyError('Choose a listed rating band.')
    response_token = secrets.token_urlsafe(32)
    response_id, now = str(uuid.uuid4()), survey._now()
    rng = random.SystemRandom()
    with database() as db, db:
        db.execute('BEGIN IMMEDIATE')
        study = _study(db, token)
        manifest = json.loads(study['manifest'])
        # Every consecutive three sessions cover each game's three levels once.
        slot = db.execute('SELECT COUNT(*) FROM comparison_responses WHERE survey_id=?', (study['id'],)).fetchone()[0] % 3
        assignment = []
        cells = grid_cells(manifest)
        for i, game in enumerate(manifest['games']):
            level = LEVELS[(i + slot) % 3]
            candidates = [{'id': secrets.token_urlsafe(18), 'variant_id': c['id']}
                          for c in cells if c['game_id'] == game['id'] and c['level'] == level]
            rng.shuffle(candidates)
            assignment.append({'game_id': game['id'], 'level': level, 'candidates': candidates})
        rng.shuffle(assignment)
        db.execute('''INSERT INTO comparison_responses
            (id,survey_id,token_hash,rating,assignment,created_at,updated_at) VALUES (?,?,?,?,?,?,?)''',
                   (response_id, study['id'], survey._digest(response_token), request.rating, json.dumps(assignment), now, now))
        row = db.execute('SELECT * FROM comparison_responses WHERE id=?', (response_id,)).fetchone()
        return {**_response(row), 'token': response_token}


def _participant(db, study, token):
    row = db.execute('SELECT * FROM comparison_responses WHERE survey_id=? AND token_hash=?',
                     (study['id'], survey._digest(token))).fetchone()
    if not row:
        raise survey.SurveyError('Response not found.', 404)
    return row


def get_response(token, response_token):
    with database() as db:
        return _response(_participant(db, _study(db, token), response_token))


def get_round(token, response_token, game_id):
    with database() as db:
        study = _study(db, token)
        row = _participant(db, study, response_token)
        assigned = next((r for r in json.loads(row['assignment']) if r['game_id'] == game_id), None)
        if not assigned:
            raise survey.SurveyError('Round not found.', 404)
        manifest = json.loads(study['manifest'])
        entry = next(g for g in manifest['games'] if g['id'] == game_id)
        candidates = []
        for i, candidate in enumerate(assigned['candidates']):
            variant = db.execute('SELECT game FROM comparison_variants WHERE survey_id=? AND variant_id=?',
                                 (study['id'], candidate['variant_id'])).fetchone()
            game = json.loads(variant['game'])
            candidates.append({'id': candidate['id'], 'label': chr(65 + i),
                               'comment': game['moves'][entry['comparison_position']['move_index']]['comment']})
        return {'game_id': game_id, 'level': assigned['level'], 'bracket': entry['bracket'],
                'position': entry['comparison_position'], 'candidates': candidates}


def get_candidate_game(token, response_token, candidate_id):
    with database() as db:
        study = _study(db, token)
        row = _participant(db, study, response_token)
        candidate = next((c for r in json.loads(row['assignment']) for c in r['candidates'] if c['id'] == candidate_id), None)
        if not candidate:
            raise survey.SurveyError('Candidate not found.', 404)
        variant = db.execute('SELECT game FROM comparison_variants WHERE survey_id=? AND variant_id=?',
                             (study['id'], candidate['variant_id'])).fetchone()
        return json.loads(variant['game'])


def save_response(token, response_token, request: ComparisonSave):
    with database() as db, db:
        db.execute('BEGIN IMMEDIATE')
        row = _participant(db, _study(db, token), response_token)
        assignments = {r['game_id']: r for r in json.loads(row['assignment'])}
        ids = [r.game_id for r in request.reviews]
        if len(ids) != len(set(ids)) or not set(ids) <= set(assignments):
            raise survey.SurveyError('Unknown or duplicate comparison round.')
        if request.completed and set(ids) != set(assignments):
            raise survey.SurveyError('Complete or skip all ten rounds before submitting.')
        for review in request.reviews:
            candidates = {c['id'] for c in assignments[review.game_id]['candidates']}
            if review.decision == 'ranked':
                if review.best not in candidates or review.worst not in candidates or review.best == review.worst:
                    raise survey.SurveyError('Choose different best and worst comments from this round.')
                if not review.best_reason.strip() or not review.worst_reason.strip():
                    raise survey.SurveyError('Explain both your best and worst choices.')
            elif review.best is not None or review.worst is not None:
                raise survey.SurveyError('Unranked rounds cannot contain best/worst choices.')
        serialized = json.dumps([r.model_dump() for r in request.reviews])
        if row['completed_at']:
            if request.completed and row['reviews'] == serialized and row['feedback'] == request.feedback:
                return _response(row)
            raise survey.SurveyError('This response has already been submitted.', 409)
        if request.version != row['version']:
            raise survey.SurveyError('Response changed in another tab. Reload saved progress.', 409)
        now = survey._now()
        db.execute('UPDATE comparison_responses SET reviews=?,feedback=?,version=version+1,updated_at=?,completed_at=? WHERE id=?',
                   (serialized, request.feedback, now, now if request.completed else None, row['id']))
        return _response(db.execute('SELECT * FROM comparison_responses WHERE id=?', (row['id'],)).fetchone())


def results(token):
    with database() as db:
        study = _study(db, token, admin=True)
        manifest = json.loads(study['manifest'])
        rows = db.execute('SELECT * FROM comparison_responses WHERE survey_id=? ORDER BY created_at', (study['id'],)).fetchall()
    cells = {c['id']: {**c, 'ranked': 0, 'best': 0, 'worst': 0, 'similar': 0, 'cannot_judge': 0, 'skipped': 0}
             for c in grid_cells(manifest)}
    completed = [r for r in rows if r['completed_at']]
    reasons, exports = [], []
    for row in completed:
        assignment = {r['game_id']: r for r in json.loads(row['assignment'])}
        order = list(assignment)
        for review in json.loads(row['reviews']):
            assigned = assignment[review['game_id']]
            for position, candidate in enumerate(assigned['candidates']):
                cell = cells[candidate['variant_id']]
                cell[review['decision']] += 1
                best, worst = candidate['id'] == review['best'], candidate['id'] == review['worst']
                cell['best'] += best
                cell['worst'] += worst
                reason = review['best_reason'] if best else review['worst_reason'] if worst else ''
                if reason:
                    reasons.append({'model': cell['model'], 'game_id': cell['game_id'], 'level': cell['level'],
                                    'choice': 'best' if best else 'worst', 'text': reason})
                exports.append({'response_id': row['id'], 'participant_rating': row['rating'],
                                'game_id': cell['game_id'], 'model': cell['model'], 'level': cell['level'],
                                'round_order': order.index(cell['game_id']) + 1, 'candidate_order': position + 1,
                                'decision': review['decision'], 'best': best, 'worst': worst, 'reason': reason,
                                'round_feedback': review['feedback'], 'overall_feedback': row['feedback'],
                                'submitted_at': row['completed_at']})
    for cell in cells.values():
        cell['best_rate'] = cell['best'] / cell['ranked'] if cell['ranked'] else None
        cell['worst_rate'] = cell['worst'] / cell['ranked'] if cell['ranked'] else None
    return {'title': manifest['title'], 'started': len(rows), 'completed': len(completed),
            'in_progress': len(rows) - len(completed), 'models': manifest['models'], 'levels': LEVELS,
            'games': [{'id': g['id'], 'bracket': g['bracket']} for g in manifest['games']],
            'cells': list(cells.values()), 'reasons': reasons, 'responses': exports,
            'feedback': [r['feedback'] for r in completed if r['feedback']], 'provenance': manifest}
