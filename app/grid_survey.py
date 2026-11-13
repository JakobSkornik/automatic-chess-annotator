"""Blinded comment comparisons with balanced assignment and 1-5 ratings per comment.

Questionnaire version 3 compares language models: every participant rates all ten
positions. Version 4 compares versions of one system (full system, template, model
without claims): every participant rates a balanced subset of the positions, answers
one more question per comment, and then checks single statements on the board.
"""
from __future__ import annotations

import json
import random
import re
import secrets
import statistics
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import chess
from pydantic import Field, StrictInt

from app import survey
from app.models.GameJson import GameJson
from app.research.survey_grid import LEVELS, grid_cells, variant_id


# Each comment is rated on every criterion, 1 = strongly disagree ... 5 = strongly agree.
CRITERIA = [
    {'id': 'understandable', 'label': 'Understandable', 'statement': 'I can follow this explanation.'},
    {'id': 'correct', 'label': 'Correct', 'statement': 'The chess statements seem right to me.'},
    {'id': 'useful', 'label': 'Useful', 'statement': 'It helps me understand the move.'},
]
SCALE = {'min': 1, 'max': 5, 'low': 'Strongly disagree', 'high': 'Strongly agree'}
QUESTIONNAIRE_VERSION = 3
# The version study: declared in the definition, never the default for a new study.
VERSION_STUDY = 4
VERSION_CRITERIA = [*CRITERIA, {'id': 'natural', 'label': 'Natural',
                                'statement': 'This reads like a comment a person would write.'}]
# Asked after each statement-checking task; an item without options uses SCALE.
TASK_ITEMS = [
    {'id': 'could_check', 'label': 'Could check', 'statement': 'I could check this statement on the board.'},
    {'id': 'trust', 'label': 'Trust', 'statement': 'Seeing the evidence changed how much I trust the comment.',
     'options': ['Much less', 'Less', 'No change', 'More', 'Much more']},
]
# Which stored ply counts as the right answer of a task: the ply the rule cites, or
# the ply at which the statement first holds on the board (set by hand per task).
TASK_KEYS = ('support_ply', 'board_ply')
# An unfinished response counts towards the assignment balance only while it is recent.
ACTIVE_HOURS = 6


def is_version_study(manifest):
    return manifest.get('questionnaire_version') == VERSION_STUDY


def criteria_for(manifest):
    return VERSION_CRITERIA if is_version_study(manifest) else CRITERIA


def candidate_label(manifest, index):
    """Version studies label comments X, Y, Z so they are not confused with the versions' own letters."""
    return ('XYZ' if is_version_study(manifest) else 'ABCDEFG')[index]


def position_types(manifest):
    """Position types in definition order (a version study draws the same number of rounds from each)."""
    return list(dict.fromkeys(g['comparison_position'].get('type') for g in manifest['games']))


def round_count(manifest):
    if is_version_study(manifest):
        return len(position_types(manifest)) * manifest['rounds_per_type']
    return len(manifest['games'])


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


# Participants grouped by their reported rating, matching the commentary levels.
PARTICIPANT_GROUPS = [('beginner', 0, 1200), ('intermediate', 1200, 2000), ('expert', 2000, 10000)]


def participant_group(elo):
    """The commentary level a participant's reported rating corresponds to (None: no rating)."""
    if elo is None:
        return None
    return next(name for name, low, high in PARTICIPANT_GROUPS if low <= elo < high)


def rating_band(elo):
    """Map a participant rating onto the game bands so older summaries keep working."""
    if elo < 800:
        return survey.BRACKETS[0]
    if elo >= 2400:
        return survey.BRACKETS[-1]
    return survey.BRACKETS[1 + (elo - 800) // 200]


class TaskAnswer(survey.StrictModel):
    task_id: str = Field(pattern=r'^task-\d{2}$')
    # Ply of the shown line after which the statement first holds; null: "I cannot tell".
    answer_ply: StrictInt | None = Field(ge=1, le=60)
    could_check: survey.Score
    # 1 = much less ... 3 = no change ... 5 = much more.
    trust: survey.Score
    elapsed_ms: StrictInt = Field(ge=0, le=6 * 3600 * 1000)


class ComparisonSave(survey.StrictModel):
    version: int = Field(ge=0)
    reviews: list[ComparisonReview] = Field(max_length=10)
    tasks: list[TaskAnswer] = Field(default_factory=list, max_length=10)
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
                elo INTEGER, rating_system TEXT,
                task_assignment TEXT NOT NULL DEFAULT '[]', task_answers TEXT NOT NULL DEFAULT '[]'
            );
        ''')
        columns = {r[1] for r in db.execute('PRAGMA table_info(comparison_responses)')}
        for column, kind in (('elo', 'INTEGER'), ('rating_system', 'TEXT'),
                             ('task_assignment', "TEXT NOT NULL DEFAULT '[]'"), ('task_answers', "TEXT NOT NULL DEFAULT '[]'")):
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
    if (manifest.get('kind') != 'comparison_grid' or manifest.get('questionnaire_version') not in (2, 3, VERSION_STUDY)
            # Ten games, each in a known rating band (a study may use two games per band).
            or len(games) != 10 or any(g['bracket'] not in survey.BRACKETS for g in games)
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
    if is_version_study(manifest):
        _check_version_study(manifest, frozen)
    return frozen


def _check_version_study(manifest, frozen):
    games = manifest['games']
    types = [g['comparison_position'].get('type') for g in games]
    per_type = manifest.get('rounds_per_type')
    if (None in types or not isinstance(per_type, int) or per_type < 1 or len(manifest['models']) > 3
            or any(types.count(t) < per_type for t in set(types)) or any(not g.get('level') for g in games)):
        raise ValueError('A version study needs typed positions, a fixed level per game and at most three versions')
    tasks = manifest.get('tasks')
    count = manifest.get('task_count')
    if (not isinstance(tasks, list) or not isinstance(count, int) or not 0 <= count <= len(tasks)
            or len({t.get('id') for t in tasks}) != len(tasks) or manifest.get('task_key') not in TASK_KEYS):
        raise ValueError('Invalid checking tasks')
    entries = {g['id']: g for g in games}
    models = {m['id'] for m in manifest['models']}
    for task in tasks:
        entry = entries.get(task.get('game_id'))
        if (not re.fullmatch(r'task-\d{2}', str(task.get('id'))) or entry is None or task.get('condition') not in models
                or task.get('line') not in ('played', 'alternative') or not task.get('statement')):
            raise ValueError(f"Invalid checking task: {task.get('id')}")
        position = entry['comparison_position']
        comment = frozen[variant_id(entry['id'], task['condition'], entry['level'])]['game']['moves'][position['move_index']]['comment']
        plies = len(task['san'])
        if (not task.get('fragment') or task['fragment'] not in comment or task['start_fen'] != position['fen_before']
                or not 1 <= plies <= 60 or len(task['fens']) != plies
                or any(not isinstance(task.get(k), int) or not 1 <= task[k] <= plies for k in TASK_KEYS)):
            raise ValueError(f"Checking task does not match its position: {task['id']}")
        board = chess.Board(task['start_fen'])
        for san, fen in zip(task['san'], task['fens']):
            board.push_san(san)
            if board.fen() != fen:
                raise ValueError(f"Checking task line does not match its positions: {task['id']}")
        series = task.get('series')
        if series is not None and (not series.get('label') or len(series['values']) != plies + 1):
            raise ValueError(f"Checking task chart does not match its line: {task['id']}")


def publish_grid(manifest, variants):
    version = VERSION_STUDY if is_version_study(manifest) else QUESTIONNAIRE_VERSION
    manifest = {**manifest, 'kind': 'comparison_grid', 'questionnaire_version': version,
                'title': manifest.get('title', 'Chess commentary comparison')}
    if version == VERSION_STUDY:
        tasks = manifest.get('tasks', [])
        manifest = {'rounds_per_type': 2, 'task_key': TASK_KEYS[0], 'task_count': len(tasks), **manifest, 'tasks': tasks}
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
            # The stored games were written by the code of their day. Read them through
            # today's game model before comparing, so a field added to the model since
            # (with a default) does not make an unchanged study look different.
            stored = exported_variants(db, definition['id'])
            if stored != frozen:
                stored = freeze_grid(definition['manifest'], stored)
            if (any(existing[k] != definition[k] for k in ('public_hash', 'admin_hash', 'created_at'))
                    or json.loads(existing['manifest']) != definition['manifest'] or stored != frozen):
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
    tasks = manifest.get('task_count', 0) if is_version_study(manifest) else 0
    # A definition may state its own estimate (for example after a pilot).
    duration = manifest.get('duration') or ('15 to 30 minutes' if not is_version_study(manifest)
                                            else 'about 20 minutes' if tasks else 'about 15 minutes')
    return {'title': 'Chess commentary comparison', 'round_count': round_count(manifest),
            'candidate_count': len(manifest['models']), 'rating_brackets': survey.BRACKETS,
            'rating_systems': RATING_SYSTEMS, 'criteria': criteria_for(manifest), 'scale': SCALE,
            'levels': LEVELS, 'review_scope': manifest.get('review_scope', 'position'),
            'task_count': tasks, 'task_items': TASK_ITEMS if tasks else [],
            # Rating rounds of a version study show the engine lines without the rule statements.
            'lines_only': is_version_study(manifest),
            'duration': duration}


def _response(row, manifest):
    assignment = json.loads(row['assignment'])
    return {'id': row['id'], 'rating': row['rating'], 'elo': row['elo'],
            'rating_system': row['rating_system'], 'version': row['version'],
            'completed': row['completed_at'] is not None, 'reviews': json.loads(row['reviews']),
            'feedback': row['feedback'], 'rounds': [
                {'game_id': r['game_id'], 'level': r['level'],
                 'candidates': [{'id': c['id'], 'label': candidate_label(manifest, i)} for i, c in enumerate(r['candidates'])]}
                for r in assignment],
            'tasks': [{'id': task_id} for task_id in json.loads(row['task_assignment'])],
            'task_answers': json.loads(row['task_answers'])}


def _usage(rows):
    """How often each game and each task is already assigned.

    Submitted responses always count. Unfinished ones count while they are recent, so
    an abandoned session stops holding its positions back after ACTIVE_HOURS.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=ACTIVE_HOURS)).isoformat()
    games, tasks = Counter(), Counter()
    for row in rows:
        if row['completed_at'] or row['updated_at'] >= cutoff:
            games.update(r['game_id'] for r in json.loads(row['assignment']))
            tasks.update(json.loads(row['task_assignment']))
    return games, tasks


def _fewest(rng, items, used, count):
    """``count`` of the items, least used first; ties are broken at random."""
    return sorted(items, key=lambda item: (used[item], rng.random()))[:count]


def start_response(token, request: GridStartRequest):
    response_token = secrets.token_urlsafe(32)
    response_id, now = str(uuid.uuid4()), survey._now()
    rng = random.SystemRandom()
    with database() as db, db:
        db.execute('BEGIN IMMEDIATE')
        study = _study(db, token)
        manifest = json.loads(study['manifest'])
        rows = db.execute('SELECT assignment,task_assignment,completed_at,updated_at FROM comparison_responses WHERE survey_id=?',
                          (study['id'],)).fetchall()
        # A game with a fixed level is always shown at that level. Otherwise every
        # consecutive three sessions cover each game's three levels once.
        slot = len(rows) % 3
        chosen, tasks = None, []
        if is_version_study(manifest):
            # The same number of positions of every type, the least used ones first.
            used_games, used_tasks = _usage(rows)
            chosen = set()
            for kind in position_types(manifest):
                of_kind = [g['id'] for g in manifest['games'] if g['comparison_position']['type'] == kind]
                chosen.update(_fewest(rng, of_kind, used_games, manifest['rounds_per_type']))
            tasks = _fewest(rng, [t['id'] for t in manifest['tasks']], used_tasks, manifest['task_count'])
            rng.shuffle(tasks)
        assignment = []
        cells = grid_cells(manifest)
        for i, game in enumerate(manifest['games']):
            if chosen is not None and game['id'] not in chosen:
                continue
            level = game.get('level') or LEVELS[(i + slot) % 3]
            candidates = [{'id': secrets.token_urlsafe(18), 'variant_id': c['id']}
                          for c in cells if c['game_id'] == game['id'] and c['level'] == level]
            rng.shuffle(candidates)
            assignment.append({'game_id': game['id'], 'level': level, 'candidates': candidates})
        rng.shuffle(assignment)
        db.execute('''INSERT INTO comparison_responses
            (id,survey_id,token_hash,rating,elo,rating_system,assignment,task_assignment,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)''',
                   (response_id, study['id'], survey._digest(response_token), rating_band(request.elo), request.elo,
                    request.rating_system, json.dumps(assignment), json.dumps(tasks), now, now))
        row = db.execute('SELECT * FROM comparison_responses WHERE id=?', (response_id,)).fetchone()
        return {**_response(row, manifest), 'token': response_token}


def _participant(db, study, token):
    row = db.execute('SELECT * FROM comparison_responses WHERE survey_id=? AND token_hash=?',
                     (study['id'], survey._digest(token))).fetchone()
    if not row:
        raise survey.SurveyError('Response not found.', 404)
    return row


def get_response(token, response_token):
    with database() as db:
        study = _study(db, token)
        return _response(_participant(db, study, response_token), json.loads(study['manifest']))


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
            move = game['moves'][entry['comparison_position']['move_index']]
            # Every version's links in one form: phrase, line, ply and bare squares. The roles and
            # files that only the system's statements carry would tell the versions apart.
            links = [{'phrase': link['phrase'], 'start': link.get('start'), 'part': link['part'], 'ply': link['ply'],
                      'evidence': {'squares': list((link.get('evidence') or {}).get('squares') or [])}}
                     for link in move.get('comment_links') or []]
            candidates.append({'id': candidate['id'], 'label': candidate_label(manifest, i),
                               'comment': move['comment'], 'links': links})
        # Only what the board needs: the position type and selection notes stay private.
        position = {k: entry['comparison_position'][k] for k in ('move_index', 'ply', 'san', 'fen_before', 'fen_after')}
        return {'game_id': game_id, 'level': assigned['level'], 'bracket': entry['bracket'],
                'position': position, 'candidates': candidates}


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
        lines_only = is_version_study(json.loads(study['manifest']))
    game = json.loads(variant['game'])
    for move in game['moves']:
        move.update(comment='', resolved_tokens=[], comment_links=[], has_comment=False, final_comment=False,
                    comment_kind='none', commentary_status=None, commentary_reason=None)
        # One version of a version study is written without the rule statements, so the
        # rating rounds show the engine lines alone; statements appear only in the tasks.
        facts = move.get('comment_facts') if lines_only else None
        if facts:
            facts['claims'] = []
            if facts.get('better_alternative'):
                facts['better_alternative']['claims'] = []
            (facts.get('selection_metadata') or {}).pop('evidence_suppressions', None)
    return game


def get_task(token, response_token, task_id):
    """One statement-checking task, without its answer.

    A task shows a full comment of one version, so it is served only after every
    rating round is saved; from the first saved task answer the ratings are locked.
    """
    with database() as db:
        study = _study(db, token)
        row = _participant(db, study, response_token)
        manifest = json.loads(study['manifest'])
        task = next((t for t in manifest.get('tasks', []) if t['id'] == task_id), None)
        if task is None or task_id not in json.loads(row['task_assignment']):
            raise survey.SurveyError('Task not found.', 404)
        if {r['game_id'] for r in json.loads(row['reviews'])} != {r['game_id'] for r in json.loads(row['assignment'])}:
            raise survey.SurveyError('Finish the rating rounds first.', 409)
        entry = next(g for g in manifest['games'] if g['id'] == task['game_id'])
        variant = db.execute('SELECT game FROM comparison_variants WHERE survey_id=? AND variant_id=?',
                             (study['id'], variant_id(entry['id'], task['condition'], entry['level']))).fetchone()
    position = entry['comparison_position']
    comment = json.loads(variant['game'])['moves'][position['move_index']]['comment']
    return {'id': task['id'], 'level': entry['level'], 'position': {'ply': position['ply'], 'san': position['san']},
            'comment': comment, 'fragment': task['fragment'], 'statement': task['statement'], 'line': task['line'],
            'start_fen': task['start_fen'], 'san': task['san'], 'fens': task['fens'],
            'squares': task.get('squares', []), 'series': task.get('series')}


def save_response(token, response_token, request: ComparisonSave):
    with database() as db, db:
        db.execute('BEGIN IMMEDIATE')
        study = _study(db, token)
        manifest = json.loads(study['manifest'])
        row = _participant(db, study, response_token)
        assignments = {r['game_id']: r for r in json.loads(row['assignment'])}
        ids = [r.game_id for r in request.reviews]
        if len(ids) != len(set(ids)) or not set(ids) <= set(assignments):
            raise survey.SurveyError('Unknown or duplicate comparison round.')
        assigned_tasks = json.loads(row['task_assignment'])
        task_ids = [t.task_id for t in request.tasks]
        if len(task_ids) != len(set(task_ids)) or not set(task_ids) <= set(assigned_tasks):
            raise survey.SurveyError('Unknown or duplicate checking task.')
        if task_ids and set(ids) != set(assignments):
            raise survey.SurveyError('Finish the rating rounds before the checking tasks.')
        plies = {t['id']: len(t['san']) for t in manifest.get('tasks', [])}
        if any(t.answer_ply is not None and t.answer_ply > plies[t.task_id] for t in request.tasks):
            raise survey.SurveyError('The chosen move is not part of the line.')
        if request.completed and set(ids) != set(assignments):
            raise survey.SurveyError('Complete or skip every round before submitting.')
        if request.completed and set(task_ids) != set(assigned_tasks):
            raise survey.SurveyError('Answer every checking task before submitting.')
        criteria = {c['id'] for c in criteria_for(manifest)}
        for review in request.reviews:
            candidates = {c['id'] for c in assignments[review.game_id]['candidates']}
            if review.decision == 'rated':
                if set(review.scores) != candidates or any(set(s) != criteria for s in review.scores.values()):
                    raise survey.SurveyError('Rate every comment on every question in this round.')
            elif review.scores:
                raise survey.SurveyError('Skipped rounds cannot contain ratings.')
        serialized = json.dumps([r.model_dump() for r in request.reviews])
        answers = json.dumps([t.model_dump() for t in request.tasks])
        if row['completed_at']:
            if (request.completed and row['reviews'] == serialized and row['feedback'] == request.feedback
                    and row['task_answers'] == answers):
                return _response(row, manifest)
            raise survey.SurveyError('This response has already been submitted.', 409)
        if request.version != row['version']:
            raise survey.SurveyError('Response changed in another tab. Reload saved progress.', 409)
        started = {t['task_id'] for t in json.loads(row['task_answers'])}
        if started and (json.loads(row['reviews']) != json.loads(serialized) or not started <= set(task_ids)):
            raise survey.SurveyError('Ratings are locked once the checking tasks have started.', 409)
        now = survey._now()
        db.execute('''UPDATE comparison_responses SET reviews=?,task_answers=?,feedback=?,version=version+1,
            updated_at=?,completed_at=? WHERE id=?''',
                   (serialized, answers, request.feedback, now, now if request.completed else None, row['id']))
        return _response(db.execute('SELECT * FROM comparison_responses WHERE id=?', (row['id'],)).fetchone(), manifest)


def results(token):
    with database() as db:
        study = _study(db, token, admin=True)
        manifest = json.loads(study['manifest'])
        rows = db.execute('SELECT * FROM comparison_responses WHERE survey_id=? ORDER BY elo IS NULL, elo, created_at',
                          (study['id'],)).fetchall()
    criteria = [c['id'] for c in criteria_for(manifest)]
    types = {g['id']: g['comparison_position'].get('type') for g in manifest['games']}
    cells = {c['id']: {**c, 'rated': 0, 'skipped': 0, 'scores': {k: [] for k in criteria}}
             for c in grid_cells(manifest)}
    completed = [r for r in rows if r['completed_at']]
    notes, exports, legacy = [], [], 0
    # Ratings by participant group, per model and per comment level.
    grouped = {}
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
                group = participant_group(row['elo'])
                for key in criteria:
                    if key in scores:
                        cell['scores'][key].append(scores[key])
                        if group:
                            for axis, value in (('model', cell['model']), ('level', cell['level'])):
                                grouped.setdefault((axis, value, group), {k: [] for k in criteria})[key].append(scores[key])
                exports.append({'response_id': row['id'], 'participant_elo': row['elo'],
                                'participant_rating_system': row['rating_system'], 'participant_rating': row['rating'],
                                'game_id': cell['game_id'], 'position_type': types[cell['game_id']],
                                'model': cell['model'], 'level': cell['level'],
                                'round_order': order.index(cell['game_id']) + 1, 'candidate_order': position + 1,
                                'decision': review['decision'], **{key: scores.get(key) for key in criteria},
                                'round_feedback': review['feedback'], 'overall_feedback': row['feedback'],
                                'submitted_at': row['completed_at']})
    def summary(values):
        return {key: {'n': len(v), 'mean': sum(v) / len(v) if v else None,
                      'distribution': [v.count(score) for score in range(1, 6)]}
                for key, v in values.items()}

    for cell in cells.values():
        cell['criteria'] = summary(cell.pop('scores'))
    task_rows, task_summary = _task_results(manifest, completed, types)
    by_group = [{'axis': axis, 'value': value, 'group': group, 'criteria': summary(values)}
                for (axis, value, group), values in sorted(grouped.items())]
    return {'title': manifest['title'], 'started': len(rows), 'completed': len(completed),
            'in_progress': len(rows) - len(completed), 'models': manifest['models'], 'levels': LEVELS,
            'criteria': criteria_for(manifest), 'scale': SCALE,
            'games': [{'id': g['id'], 'bracket': g['bracket'], 'type': types[g['id']], 'level': g.get('level')}
                      for g in manifest['games']],
            'cells': list(cells.values()), 'by_group': by_group,
            'participant_groups': [{'id': name, 'min_elo': low, 'max_elo_exclusive': high} for name, low, high in PARTICIPANT_GROUPS],
            'participants_by_group': {name: sum(1 for r in completed if participant_group(r['elo']) == name)
                                      for name, _, _ in PARTICIPANT_GROUPS},
            'notes': notes, 'responses': exports, 'legacy_reviews': legacy,
            'tasks': task_summary, 'task_responses': task_rows, 'task_key': manifest.get('task_key'),
            'task_items': TASK_ITEMS if task_summary else [], 'versions': is_version_study(manifest),
            'feedback': [r['feedback'] for r in completed if r['feedback']], 'provenance': manifest}


def _task_results(manifest, completed, types):
    """One row per answered checking task, and per task the share of right answers.

    Each row keeps the chosen ply and both stored plies, so the analysis can score the
    answers against either key; ``correct`` uses the key the study declares.
    """
    tasks = {t['id']: t for t in manifest.get('tasks', [])}
    key = manifest.get('task_key', TASK_KEYS[0])
    rows = []
    for row in completed:
        rated = {r['game_id'] for r in json.loads(row['reviews']) if r.get('decision') == 'rated'}
        order = json.loads(row['task_assignment'])
        for answer in json.loads(row['task_answers']):
            task = tasks[answer['task_id']]
            hit = {k: answer['answer_ply'] == task[k] for k in TASK_KEYS}
            rows.append({'response_id': row['id'], 'participant_elo': row['elo'],
                         'participant_rating_system': row['rating_system'], 'participant_rating': row['rating'],
                         'task_id': task['id'], 'game_id': task['game_id'], 'position_type': types[task['game_id']],
                         'line': task['line'], 'rule_id': task.get('rule_id'), 'task_order': order.index(task['id']) + 1,
                         'rated_position': task['game_id'] in rated, 'line_plies': len(task['san']),
                         'answer_ply': answer['answer_ply'], 'support_ply': task['support_ply'],
                         'board_ply': task['board_ply'], 'chart_change_ply': task.get('chart_change_ply'),
                         'correct_support_ply': hit['support_ply'],
                         'correct_board_ply': hit['board_ply'], 'correct': hit[key],
                         'could_check': answer['could_check'], 'trust': answer['trust'],
                         'elapsed_ms': answer['elapsed_ms'], 'submitted_at': row['completed_at']})
    summary = []
    for task in tasks.values():
        mine = [r for r in rows if r['task_id'] == task['id']]
        summary.append({'id': task['id'], 'game_id': task['game_id'], 'line': task['line'], 'statement': task['statement'],
                        'line_plies': len(task['san']), 'n': len(mine), 'correct': sum(r['correct'] for r in mine),
                        'cannot_tell': sum(r['answer_ply'] is None for r in mine),
                        'could_check_mean': statistics.fmean(r['could_check'] for r in mine) if mine else None,
                        'trust_distribution': [sum(r['trust'] == score for r in mine) for score in range(1, 6)],
                        'median_seconds': statistics.median(r['elapsed_ms'] for r in mine) / 1000 if mine else None})
    return rows, summary
