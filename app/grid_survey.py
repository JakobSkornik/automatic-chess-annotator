"""Blinded model comparisons with balanced level assignment and 1-5 ratings per comment."""
from __future__ import annotations

import json
import random
import secrets
import uuid
from contextlib import contextmanager
from typing import Literal

from pydantic import Field, StrictInt

from app import survey
from app.models.GameJson import GameJson
from app.research.survey_grid import LEVELS, grid_cells


# Each comment is rated on every criterion, 1 = strongly disagree ... 5 = strongly agree.
CRITERIA = [
    {'id': 'understandable', 'label': 'Understandable', 'statement': 'I can follow this explanation.'},
    {'id': 'correct', 'label': 'Correct', 'statement': 'The chess statements seem right to me.'},
    {'id': 'useful', 'label': 'Useful', 'statement': 'It helps me understand the move.'},
]
SCALE = {'min': 1, 'max': 5, 'low': 'Strongly disagree', 'high': 'Strongly agree'}
QUESTIONNAIRE_VERSION = 3


class ComparisonReview(survey.StrictModel):
    game_id: str = Field(pattern=r'^game-\d{2}$')
    decision: Literal['rated', 'skipped']
    # candidate id -> criterion id -> score
    scores: dict[str, dict[str, survey.Score]] = Field(default_factory=dict)
    feedback: str = Field(default='', max_length=1500)


RATING_SYSTEMS = ['FIDE', 'Lichess', 'Chess.com', 'National federation', 'Estimate (unrated)']


class GridStartRequest(survey.StrictModel):
    consent: Literal[True]
    elo: StrictInt = Field(ge=100, le=3500)
    rating_system: Literal[tuple(RATING_SYSTEMS)]


def rating_band(elo):
    """Map a participant rating onto the game bands so older summaries keep working."""
    if elo < 800:
        return survey.BRACKETS[0]
    if elo >= 2400:
        return survey.BRACKETS[-1]
    return survey.BRACKETS[1 + (elo - 800) // 200]


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
                updated_at TEXT NOT NULL, completed_at TEXT,
                elo INTEGER, rating_system TEXT
            );
        ''')
        columns = {r[1] for r in db.execute('PRAGMA table_info(comparison_responses)')}
        for column, kind in (('elo', 'INTEGER'), ('rating_system', 'TEXT')):
            if column not in columns:
                db.execute(f'ALTER TABLE comparison_responses ADD COLUMN {column} {kind}')
        yield db


def _study(db, token, *, admin=False):
    row = survey._survey(db, token, admin=admin)
    if json.loads(row['manifest']).get('kind') != 'comparison_grid':
        raise survey.SurveyError('Comparison study not found.', 404)
    return row


def freeze_grid(manifest, variants):
    games = manifest.get('games', [])
    models = manifest.get('models', [])
    # Version 2 (best/worst choices) studies stay importable; new studies use version 3.
    if (manifest.get('kind') != 'comparison_grid' or manifest.get('questionnaire_version') not in (2, 3)
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
    manifest = {**manifest, 'kind': 'comparison_grid', 'questionnaire_version': QUESTIONNAIRE_VERSION,
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
            'rating_systems': RATING_SYSTEMS, 'criteria': CRITERIA, 'scale': SCALE,
            'levels': LEVELS, 'review_scope': manifest.get('review_scope', 'position')}


def _response(row):
    assignment = json.loads(row['assignment'])
    return {'id': row['id'], 'rating': row['rating'], 'elo': row['elo'],
            'rating_system': row['rating_system'], 'version': row['version'],
            'completed': row['completed_at'] is not None, 'reviews': json.loads(row['reviews']),
            'feedback': row['feedback'], 'rounds': [
                {'game_id': r['game_id'], 'level': r['level'],
                 'candidates': [{'id': c['id'], 'label': chr(65 + i)} for i, c in enumerate(r['candidates'])]}
                for r in assignment]}


def start_response(token, request: GridStartRequest):
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
            (id,survey_id,token_hash,rating,elo,rating_system,assignment,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)''',
                   (response_id, study['id'], survey._digest(response_token), rating_band(request.elo), request.elo,
                    request.rating_system, json.dumps(assignment), now, now))
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


def get_round_board(token, response_token, game_id):
    """The round's game with engine lines and evidence but no comment prose.

    Variations and rule facts are identical for every candidate of a round (same
    engine evidence and level), so participants explore them without seeing any
    model's commentary on other moves.
    """
    with database() as db:
        study = _study(db, token)
        row = _participant(db, study, response_token)
        assigned = next((r for r in json.loads(row['assignment']) if r['game_id'] == game_id), None)
        if not assigned:
            raise survey.SurveyError('Round not found.', 404)
        variant = db.execute('SELECT game FROM comparison_variants WHERE survey_id=? AND variant_id=?',
                             (study['id'], assigned['candidates'][0]['variant_id'])).fetchone()
    game = json.loads(variant['game'])
    for move in game['moves']:
        move.update(comment='', resolved_tokens=[], has_comment=False, final_comment=False,
                    comment_kind='none', commentary_status=None, commentary_reason=None)
    return game


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
        criteria = {c['id'] for c in CRITERIA}
        for review in request.reviews:
            candidates = {c['id'] for c in assignments[review.game_id]['candidates']}
            if review.decision == 'rated':
                if set(review.scores) != candidates or any(set(s) != criteria for s in review.scores.values()):
                    raise survey.SurveyError('Rate every comment on every question in this round.')
            elif review.scores:
                raise survey.SurveyError('Skipped rounds cannot contain ratings.')
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
        rows = db.execute('SELECT * FROM comparison_responses WHERE survey_id=? ORDER BY elo IS NULL, elo, created_at',
                          (study['id'],)).fetchall()
    criteria = [c['id'] for c in CRITERIA]
    cells = {c['id']: {**c, 'rated': 0, 'skipped': 0, 'scores': {k: [] for k in criteria}}
             for c in grid_cells(manifest)}
    completed = [r for r in rows if r['completed_at']]
    notes, exports, legacy = [], [], 0
    for row in completed:
        assignment = {r['game_id']: r for r in json.loads(row['assignment'])}
        order = list(assignment)
        for review in json.loads(row['reviews']):
            if review.get('decision') not in ('rated', 'skipped'):
                legacy += 1  # Best/worst answers from questionnaire version 2.
                continue
            assigned = assignment[review['game_id']]
            if review['feedback']:
                notes.append({'game_id': review['game_id'], 'level': assigned['level'], 'text': review['feedback']})
            for position, candidate in enumerate(assigned['candidates']):
                cell = cells[candidate['variant_id']]
                cell[review['decision']] += 1
                scores = review['scores'].get(candidate['id'], {})
                for key in criteria:
                    if key in scores:
                        cell['scores'][key].append(scores[key])
                exports.append({'response_id': row['id'], 'participant_elo': row['elo'],
                                'participant_rating_system': row['rating_system'], 'participant_rating': row['rating'],
                                'game_id': cell['game_id'], 'model': cell['model'], 'level': cell['level'],
                                'round_order': order.index(cell['game_id']) + 1, 'candidate_order': position + 1,
                                'decision': review['decision'], **{key: scores.get(key) for key in criteria},
                                'round_feedback': review['feedback'], 'overall_feedback': row['feedback'],
                                'submitted_at': row['completed_at']})
    for cell in cells.values():
        values = cell.pop('scores')
        cell['criteria'] = {key: {'n': len(v), 'mean': sum(v) / len(v) if v else None,
                                  'distribution': [v.count(score) for score in range(1, 6)]}
                            for key, v in values.items()}
    return {'title': manifest['title'], 'started': len(rows), 'completed': len(completed),
            'in_progress': len(rows) - len(completed), 'models': manifest['models'], 'levels': LEVELS,
            'criteria': CRITERIA, 'scale': SCALE,
            'games': [{'id': g['id'], 'bracket': g['bracket']} for g in manifest['games']],
            'cells': list(cells.values()), 'notes': notes, 'responses': exports, 'legacy_reviews': legacy,
            'feedback': [r['feedback'] for r in completed if r['feedback']], 'provenance': manifest}
