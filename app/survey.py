"""Frozen, capability-link surveys backed by a durable SQLite database.

No participant route calls the engine or a language model. Public, participant,
and researcher tokens are independent and stored only as SHA-256 digests.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.core.output_paths import output_root
from app.models.GameJson import GameJson

QUESTIONS = [
    {'id': 'clarity', 'text': 'The comments were easy to understand.'},
    {'id': 'usefulness', 'text': 'The comments helped me make sense of the moves.'},
    {'id': 'coherence', 'text': 'I could follow how the comments related to the displayed continuations.'},
    {'id': 'presentation', 'text': 'The layout made the comments and supporting information easy to explore.'},
]
BRACKETS = ['Under 800', *[f'{n}–{n+199}' for n in range(800, 2400, 200)], '2400+']
Score = Annotated[StrictInt, Field(ge=1, le=5)]


class SurveyError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.status = status
        super().__init__(message)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class StartRequest(StrictModel):
    consent: Literal[True]
    rating: str = Field(default='Unrated / prefer not to say', max_length=60)


class Ratings(StrictModel):
    clarity: Score | None
    usefulness: Score | None
    coherence: Score | None
    presentation: Score | None


class Review(StrictModel):
    game_id: str = Field(pattern=r'^game-\d{2}$')
    skipped: bool = False
    ratings: Ratings
    feedback: str = Field(default='', max_length=2000)


class SaveRequest(StrictModel):
    version: int = Field(ge=0)
    reviews: list[Review] = Field(max_length=10)
    feedback: str = Field(default='', max_length=4000)
    completed: bool = False


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(token: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_-]{32,64}', token):
        raise SurveyError('Link not found.', 404)
    return hashlib.sha256(token.encode()).hexdigest()


@contextmanager
def database():
    path = Path(os.environ['ACA_SURVEY_DB']) if os.getenv('ACA_SURVEY_DB') else output_root() / 'surveys' / 'surveys.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA foreign_keys=ON')
    connection.executescript('''
        CREATE TABLE IF NOT EXISTS surveys (
            id TEXT PRIMARY KEY, public_hash TEXT UNIQUE NOT NULL,
            admin_hash TEXT UNIQUE NOT NULL, manifest TEXT NOT NULL,
            games TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS survey_responses (
            id TEXT PRIMARY KEY, survey_id TEXT NOT NULL REFERENCES surveys(id),
            token_hash TEXT UNIQUE NOT NULL, rating TEXT NOT NULL,
            game_order TEXT NOT NULL, reviews TEXT NOT NULL DEFAULT '[]',
            feedback TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            completed_at TEXT
        );
    ''')
    try:
        yield connection
    finally:
        connection.close()


def _survey(db, token: str, *, admin=False):
    column = 'admin_hash' if admin else 'public_hash'
    row = db.execute(f'SELECT * FROM surveys WHERE {column}=?', (_digest(token),)).fetchone()
    if row is None:
        raise SurveyError('Link not found.', 404)
    return row


def _freeze_games(manifest: dict, games: dict) -> dict:
    """Validate a complete set and keep only participant-facing game fields."""
    entries = manifest.get('games', [])
    ids = [e['id'] for e in entries]
    if len(ids) != 10 or len(set(ids)) != 10 or set(ids) != set(games):
        raise ValueError('A survey needs exactly ten distinct, completed games.')
    if [e['bracket'] for e in entries] != BRACKETS:
        raise ValueError('Expected one game in each agreed rating band, in order.')
    public_games = {}
    for entry in entries:
        game_id = entry['id']
        if not re.fullmatch(r'game-\d{2}', game_id):
            raise ValueError('Invalid game id')
        original = games[game_id]
        game = GameJson.model_validate(original).model_dump(mode='json')
        if not game.get('commentary_complete') or not any(m.get('comment_kind') == 'llm' and m.get('comment') for m in game['moves']):
            raise ValueError(f'{game_id} is not a completed game with generated comments.')
        white, black = game['metadata'].get('whiteElo'), game['metadata'].get('blackElo')
        if white is None or black is None:
            raise ValueError('Both player ratings are required.')
        mean = (white + black) / 2
        if not entry['min_rating'] <= mean < entry['max_rating_exclusive']:
            raise ValueError(f'Rating band mismatch for {game_id}')
        # Model validation drops unrecognized fields; strip known diagnostics too.
        for move in game['moves']:
            move.pop('debug', None)
        game['metadata']['id'] = game_id
        public_games[game_id] = game
    return public_games


def publish(manifest: dict, games: dict) -> dict:
    """Called by a local CLI only, after every game's generation has completed."""
    public_games = _freeze_games(manifest, games)
    public_token, admin_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    survey_id = str(uuid.uuid4())
    manifest = {**manifest, 'title': manifest.get('title', 'Chess commentary study'),
                'questionnaire_version': 1, 'questions': QUESTIONS}
    with database() as db, db:
        db.execute('INSERT INTO surveys VALUES (?,?,?,?,?,?)', (
            survey_id, _digest(public_token), _digest(admin_token),
            json.dumps(manifest, ensure_ascii=False), json.dumps(public_games, ensure_ascii=False), _now()))
    return {'id': survey_id, 'participant_token': public_token, 'results_token': admin_token}


def public_manifest(token: str):
    with database() as db:
        row = _survey(db, token)
        manifest = json.loads(row['manifest'])
        return {'title': manifest['title'], 'questionnaire_version': manifest['questionnaire_version'],
                'questions': manifest['questions'], 'rating_brackets': BRACKETS,
                'rating_system': manifest.get('rating_system', 'Average of both player ratings'),
                'games': [{'id': e['id'], 'bracket': e['bracket']} for e in manifest['games']]}


def get_game(token: str, game_id: str):
    with database() as db:
        games = json.loads(_survey(db, token)['games'])
        if game_id not in games:
            raise SurveyError('Game not found.', 404)
        return games[game_id]


def _response(row):
    return {'id': row['id'], 'rating': row['rating'], 'order': json.loads(row['game_order']),
            'reviews': json.loads(row['reviews']), 'feedback': row['feedback'],
            'version': row['version'], 'completed': row['completed_at'] is not None}


def start_response(token: str, request: StartRequest):
    if request.rating not in BRACKETS + ['Unrated / prefer not to say']:
        raise SurveyError('Choose one of the listed rating bands.')
    response_token = secrets.token_urlsafe(32)
    response_id, now = str(uuid.uuid4()), _now()
    with database() as db, db:
        survey = _survey(db, token)
        order = [e['id'] for e in json.loads(survey['manifest'])['games']]
        random.SystemRandom().shuffle(order)
        db.execute('''INSERT INTO survey_responses
            (id,survey_id,token_hash,rating,game_order,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?)''', (response_id, survey['id'], _digest(response_token),
                                     request.rating, json.dumps(order), now, now))
        row = db.execute('SELECT * FROM survey_responses WHERE id=?', (response_id,)).fetchone()
        return {**_response(row), 'token': response_token}


def _participant(db, survey, response_token):
    row = db.execute('SELECT * FROM survey_responses WHERE survey_id=? AND token_hash=?',
                     (survey['id'], _digest(response_token))).fetchone()
    if row is None:
        raise SurveyError('Response not found.', 404)
    return row


def get_response(token: str, response_token: str):
    with database() as db:
        return _response(_participant(db, _survey(db, token), response_token))


def save_response(token: str, response_token: str, request: SaveRequest):
    with database() as db, db:
        db.execute('BEGIN IMMEDIATE')
        survey = _survey(db, token)
        row = _participant(db, survey, response_token)
        ids = [r.game_id for r in request.reviews]
        expected = set(json.loads(row['game_order']))
        if len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise SurveyError('Unknown or duplicate game.')
        if request.completed and set(ids) != expected:
            raise SurveyError('Review or explicitly skip all ten games before submitting.')
        for review in request.reviews:
            if review.skipped and any(v is not None for v in review.ratings.model_dump().values()):
                raise SurveyError('Skipped games cannot contain ratings.')
        serialized = json.dumps([r.model_dump() for r in request.reviews])
        if row['completed_at'] is not None:
            if request.completed and serialized == row['reviews'] and request.feedback == row['feedback']:
                return _response(row)  # Safe retry after a lost response.
            raise SurveyError('This response has already been submitted.', 409)
        if request.version != row['version']:
            raise SurveyError('This response changed in another tab. Reload before saving.', 409)
        now = _now()
        db.execute('''UPDATE survey_responses SET reviews=?,feedback=?,version=version+1,
            updated_at=?,completed_at=? WHERE id=?''',
            (serialized, request.feedback, now, now if request.completed else None, row['id']))
        return _response(db.execute('SELECT * FROM survey_responses WHERE id=?', (row['id'],)).fetchone())


def results(token: str):
    with database() as db:
        survey = _survey(db, token, admin=True)
        manifest = json.loads(survey['manifest'])
        rows = db.execute('SELECT * FROM survey_responses WHERE survey_id=? ORDER BY created_at', (survey['id'],)).fetchall()
    completed = [r for r in rows if r['completed_at']]
    game_results = []
    for game in manifest['games']:
        reviews = [review for row in completed for review in json.loads(row['reviews']) if review['game_id'] == game['id']]
        metrics = {}
        for question in manifest['questions']:
            key = question['id']
            values = [r['ratings'][key] for r in reviews if not r['skipped'] and r['ratings'][key] is not None]
            metrics[key] = {'count': len(values), 'mean': sum(values) / len(values) if values else None,
                            'distribution': [values.count(score) for score in range(1, 6)],
                            'cannot_judge': sum(not r['skipped'] and r['ratings'][key] is None for r in reviews)}
        game_results.append({'id': game['id'], 'bracket': game['bracket'], 'metrics': metrics,
                             'skipped': sum(r['skipped'] for r in reviews),
                             'feedback': [r['feedback'] for r in reviews if r['feedback']]})
    return {'title': manifest['title'], 'started': len(rows), 'completed': len(completed),
            'in_progress': len(rows) - len(completed), 'questions': manifest['questions'],
            'games': game_results, 'feedback': [r['feedback'] for r in completed if r['feedback']],
            'provenance': manifest,
            'responses': [{**_response(row), 'created_at': row['created_at'], 'completed_at': row['completed_at']} for row in completed]}
