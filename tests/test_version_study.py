"""Version study (questionnaire version 4): balanced subset of positions, a fourth
question, X/Y/Z labels and statement-checking tasks after the rating rounds."""
import copy
import csv
import io
import json
from collections import Counter

import chess
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import drive_backup, grid_survey, survey
from app.research.survey_grid import LEVELS, grid_cells
from app.research.survey_tasks import build_task
from app.routes.survey_grid import router

TYPES = ['T', 'T', 'T', 'T', 'G', 'G', 'G', 'Q', 'Q', 'Q']
LINE = ['e5', 'Nf3', 'Nc6']


def line_fens(start, sans):
    board, fens = chess.Board(start), []
    for san in sans:
        board.push_san(san)
        fens.append(board.fen())
    return fens


@pytest.fixture
def version_grid():
    start = chess.Board()
    after = chess.Board(); after.push_san('e4')
    games = []
    for i, bracket in enumerate(survey.BRACKETS):
        low, high = (0, 800) if i == 0 else (2400, 4001) if i == 9 else (600 + i * 200, 800 + i * 200)
        games.append({'id': f'game-{i + 1:02}', 'bracket': bracket, 'min_rating': low, 'max_rating_exclusive': high,
                      'level': LEVELS[i % 3],
                      'comparison_position': {'move_index': 0, 'ply': 1, 'san': 'e4', 'type': TYPES[i],
                                              'fen_before': start.fen(), 'fen_after': after.fen()}})
    manifest = {'games': games, 'levels': LEVELS, 'kind': 'comparison_grid', 'questionnaire_version': 4,
                'title': 'TEST VERSION STUDY', 'rounds_per_type': 2, 'task_key': 'board_ply',
                'models': [{'id': 'P', 'provider': 'openai', 'label': 'Full system'},
                           {'id': 'T', 'provider': 'template', 'label': 'Template'},
                           {'id': 'B', 'provider': 'openai', 'label': 'Model without claims'}]}
    fens = line_fens(after.fen(), LINE)
    variants = {}
    for cell in grid_cells(manifest):
        entry = next(g for g in games if g['id'] == cell['game_id'])
        claim = {'rule_id': 'development', 'text': 'TEST STATEMENT', 'support_ply': 3,
                 'features': ['TACTICAL_CAPTURE', 'WHITE_ROOK_OPEN_FILE', 'WHITE_SPACE'],
                 'evidence': {'squares': ['f3', 'c6']}}
        facts = {'claims': [claim], 'display_line': {'start_fen': start.fen(), 'san': ['e4', *LINE],
                                                     'fens': [after.fen(), *fens],
                                                     'feature_series': {'WHITE_ROOK_OPEN_FILE': [0, 0, 0, 0, 0],
                                                                        'WHITE_SPACE': [0, 0, 1, 2, 2]}}}
        game = {'metadata': {'id': 'private-job', 'white': 'Test white', 'black': 'Test black', 'result': '1-0',
                             'whiteElo': entry['min_rating'] + 100, 'blackElo': entry['min_rating'] + 100,
                             'commentary_level': cell['level']},
                'moves': [{'mn': 1, 'color': 'w', 'san': 'e4', 'uci': 'e2e4', 'fen': after.fen(),
                           'comment': f"TEST COMMENT {cell['model']} WITH A CLAIM", 'comment_kind': 'llm',
                           'comment_facts': facts}],
                'commentary_complete': True, 'analysis_info': {'engine': 'test', 'depth': 1, 'multipv': 1, 'timestamp': 0}}
        variants[cell['id']] = {**cell, 'game': game, 'provenance': {'test_fixture': True}}
    manifest['tasks'] = [
        build_task({'id': f'task-{n:02}', 'game_id': game_id, 'condition': 'P', 'line': 'played', 'rule': 'development',
                    'fragment': 'WITH A CLAIM', 'board_ply': 2},
                   variants[f"{game_id}__P__{next(g['level'] for g in games if g['id'] == game_id)}"]['game'],
                   next(g for g in games if g['id'] == game_id))
        for n, game_id in ((1, 'game-01'), (2, 'game-05'))]
    return manifest, variants


@pytest.fixture
def study(tmp_path, monkeypatch, version_grid):
    monkeypatch.setenv('ACA_SURVEY_DB', str(tmp_path / 'test.sqlite3'))
    links = grid_survey.publish_grid(*version_grid)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        yield client, links


def start(client, links, elo=1500):
    response = client.post(f"/surveys/grid/{links['participant_token']}/session",
                           json={'consent': True, 'elo': elo, 'rating_system': 'Lichess'})
    assert response.status_code == 200
    session = response.json()
    return session, {'Authorization': 'Bearer ' + session['token']}


def reviews(session, score=4):
    criteria = [c['id'] for c in grid_survey.VERSION_CRITERIA]
    return [{'game_id': r['game_id'], 'decision': 'rated', 'feedback': '',
             'scores': {c['id']: {k: score for k in criteria} for c in r['candidates']}} for r in session['rounds']]


def answers(session, ply=2):
    return [{'task_id': t['id'], 'answer_ply': ply, 'could_check': 4, 'trust': 3, 'elapsed_ms': 42000}
            for t in session['tasks']]


def test_version_study_manifest_and_blinded_rounds(study):
    client, links = study
    token = links['participant_token']
    manifest = client.get(f'/surveys/grid/{token}').json()
    assert (manifest['round_count'], manifest['task_count'], manifest['candidate_count']) == (6, 2, 3)
    assert [c['id'] for c in manifest['criteria']] == ['understandable', 'correct', 'useful', 'natural']
    assert [i['id'] for i in manifest['task_items']] == ['could_check', 'trust'] and manifest['duration'] == 'about 20 minutes'
    session, headers = start(client, links)
    assert len(session['rounds']) == 6 and len(session['tasks']) == 2 and session['task_answers'] == []
    assert all([c['label'] for c in r['candidates']] == ['X', 'Y', 'Z'] for r in session['rounds'])
    first = session['rounds'][0]
    round_data = client.get(f"/surveys/grid/{token}/round/{first['game_id']}", headers=headers)
    assert [c['label'] for c in round_data.json()['candidates']] == ['X', 'Y', 'Z']
    public = json.dumps(manifest) + json.dumps(session)
    for private in ('variant_id', 'support_ply', 'board_ply', '"type"', 'condition', links['results_token']):
        assert private not in public
    board = client.get(f"/surveys/grid/{token}/round/{first['game_id']}/board", headers=headers).json()
    assert board['moves'][0]['comment'] == '' and board['moves'][0]['comment_facts']['claims'] == []
    assert 'TEST STATEMENT' not in json.dumps(board) and board['moves'][0]['comment_facts']['display_line']['san']
    # The position type is not served with a round either.
    assert set(round_data.json()['position']) == {'move_index', 'ply', 'san', 'fen_before', 'fen_after'}


def test_version_study_assigns_two_per_type_and_balances_positions(study, version_grid):
    client, links = study
    types = {g['id']: g['comparison_position']['type'] for g in version_grid[0]['games']}
    used = Counter()
    for _ in range(12):
        session, _ = start(client, links)
        ids = [r['game_id'] for r in session['rounds']]
        assert Counter(types[i] for i in ids) == {'T': 2, 'G': 2, 'Q': 2}
        used.update(ids)
    # Twelve sessions: each of the four T positions six times, each G and Q position eight times.
    assert {used[i] for i, t in types.items() if t == 'T'} == {6}
    assert {used[i] for i, t in types.items() if t != 'T'} == {8}


def test_version_study_tasks_follow_the_rounds_and_lock_ratings(study):
    client, links = study
    token = links['participant_token']
    session, headers = start(client, links)
    url = f'/surveys/grid/{token}/session'
    task_id = session['tasks'][0]['id']
    task_url = f'/surveys/grid/{token}/task/{task_id}'
    assert client.get(task_url, headers=headers).status_code == 409
    rated = reviews(session)
    wrong = copy.deepcopy(rated); del wrong[0]['scores'][session['rounds'][0]['candidates'][0]['id']]['natural']
    assert client.put(url, headers=headers, json={'version': 0, 'reviews': wrong}).status_code == 400
    # Task answers are refused until every round is saved.
    assert client.put(url, headers=headers, json={'version': 0, 'reviews': rated[:3], 'tasks': answers(session)[:1]}).status_code == 400
    assert client.put(url, headers=headers, json={'version': 0, 'reviews': rated}).status_code == 200
    task = client.get(task_url, headers=headers)
    assert task.status_code == 200
    data = task.json()
    assert data['comment'] == 'TEST COMMENT P WITH A CLAIM' and data['fragment'] in data['comment']
    assert (data['statement'], data['san'], data['squares']) == ('TEST STATEMENT', ['e4', *LINE], ['f3', 'c6'])
    # The chart is the statement's own feature: the first one with a stored series that changes.
    assert (data['series']['feature'], data['series']['values']) == ('WHITE_SPACE', [0, 0, 1, 2, 2])
    assert data['series']['label'].startswith('White: ') and len(data['fens']) == 4 and 'chart_change_ply' not in task.text
    assert 'support_ply' not in task.text and 'board_ply' not in task.text and 'condition' not in task.text
    assert client.get(f'/surveys/grid/{token}/task/task-99', headers=headers).status_code == 404
    assert client.get(task_url).status_code == 401
    given = answers(session)
    for bad in ({'answer_ply': 9}, {'answer_ply': 0}, {'trust': 6}, {'task_id': 'task-99'}):
        assert client.put(url, headers=headers, json={'version': 1, 'reviews': rated, 'tasks': [{**given[0], **bad}]}).status_code in (400, 422)
    assert client.put(url, headers=headers, json={'version': 1, 'reviews': rated, 'tasks': given[:1]}).status_code == 200
    # Part two has started: the ratings can no longer change, and a saved answer cannot be withdrawn.
    changed = copy.deepcopy(rated); changed[0]['scores'][session['rounds'][0]['candidates'][0]['id']]['useful'] = 1
    assert client.put(url, headers=headers, json={'version': 2, 'reviews': changed, 'tasks': given[:1]}).status_code == 409
    assert client.put(url, headers=headers, json={'version': 2, 'reviews': rated, 'tasks': []}).status_code == 409
    assert client.put(url, headers=headers, json={'version': 2, 'reviews': rated, 'tasks': given[:1], 'completed': True}).status_code == 400
    given[1].update(answer_ply=None, trust=5)
    done = client.put(url, headers=headers, json={'version': 2, 'reviews': rated, 'tasks': given, 'completed': True})
    assert done.status_code == 200 and done.json()['completed'] and done.json()['task_answers'] == given


def test_version_study_results_score_tasks_against_both_keys(study, version_grid):
    client, links = study
    session, headers = start(client, links, elo=1720)
    url = f"/surveys/grid/{links['participant_token']}/session"
    rated = reviews(session)
    given = answers(session, ply=2)
    given[1]['answer_ply'] = 3
    assert client.put(url, headers=headers, json={'version': 0, 'reviews': rated}).status_code == 200
    assert client.put(url, headers=headers, json={'version': 1, 'reviews': rated, 'tasks': given, 'completed': True}).status_code == 200
    data = grid_survey.results(links['results_token'])
    assert [c['id'] for c in data['criteria']][-1] == 'natural' and data['task_key'] == 'board_ply'
    assert len(data['cells']) == 30 and sum(c['rated'] for c in data['cells']) == 18
    assert sum(c['criteria']['natural']['n'] for c in data['cells']) == 18
    rows = {r['task_id']: r for r in data['task_responses']}
    first, second = (rows[t['id']] for t in session['tasks'])
    assert (first['answer_ply'], first['correct_board_ply'], first['correct_support_ply'], first['correct']) == (2, True, False, True)
    assert (second['answer_ply'], second['correct_board_ply'], second['correct_support_ply'], second['correct']) == (3, False, True, False)
    assert first['chart_change_ply'] == 2
    assert {r['task_order'] for r in rows.values()} == {1, 2} and first['line_plies'] == 4 and first['elapsed_ms'] == 42000
    rated_games = {r['game_id'] for r in session['rounds']}
    assert all(r['rated_position'] == (r['game_id'] in rated_games) for r in rows.values())
    assert sum(t['correct'] for t in data['tasks']) == 1 and sum(t['n'] for t in data['tasks']) == 2
    types = {g['id']: g['comparison_position']['type'] for g in version_grid[0]['games']}
    export = list(csv.DictReader(io.StringIO(client.get(f"/surveys/grid/results/{links['results_token']}/export.csv").text)))
    assert len(export) == 18 and {r['natural'] for r in export} == {'4'}
    assert all(r['position_type'] == types[r['game_id']] for r in export) and {r['model'] for r in export} == {'P', 'T', 'B'}
    tasks_csv = list(csv.DictReader(io.StringIO(client.get(f"/surveys/grid/results/{links['results_token']}/tasks.csv").text)))
    assert len(tasks_csv) == 2 and {r['participant_elo'] for r in tasks_csv} == {'1720'}
    record = drive_backup.response_record(next(iter(drive_backup._pending())))
    assert record['task_answers'] == given and len(record['task_assignment']) == 2


def test_version_study_refuses_inconsistent_definitions(version_grid):
    manifest, variants = version_grid
    manifest = {**manifest, 'task_count': 2}
    grid_survey.freeze_grid(manifest, variants)

    def broken(**changes):
        tasks = copy.deepcopy(manifest['tasks'])
        tasks[0].update(changes)
        return {**manifest, 'tasks': tasks}
    for bad in (broken(fragment='NOT IN THE COMMENT'), broken(board_ply=9), broken(san=['e4', 'e5', 'Nf3', 'Nf6']),
                broken(condition='Z'), {**manifest, 'task_count': 3}, {**manifest, 'task_key': 'other'},
                {**manifest, 'rounds_per_type': 4}):
        with pytest.raises(ValueError):
            grid_survey.freeze_grid(bad, variants)
    untyped = copy.deepcopy(manifest); del untyped['games'][0]['comparison_position']['type']
    with pytest.raises(ValueError):
        grid_survey.freeze_grid(untyped, variants)


def test_model_comparison_study_takes_no_task_answers(tmp_path, monkeypatch, version_grid):
    manifest, variants = version_grid
    monkeypatch.setenv('ACA_SURVEY_DB', str(tmp_path / 'old.sqlite3'))
    plain = {k: v for k, v in manifest.items() if k not in ('tasks', 'rounds_per_type', 'task_key')}
    links = grid_survey.publish_grid({**plain, 'questionnaire_version': 3}, variants)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        public = client.get(f"/surveys/grid/{links['participant_token']}").json()
        assert (public['round_count'], public['task_count'], len(public['criteria'])) == (10, 0, 3)
        session, headers = start(client, links)
        assert len(session['rounds']) == 10 and session['tasks'] == []
        assert [c['label'] for c in session['rounds'][0]['candidates']] == ['A', 'B', 'C']
        body = {'version': 0, 'reviews': [], 'tasks': [{'task_id': 'task-01', 'answer_ply': 1, 'could_check': 3, 'trust': 3, 'elapsed_ms': 1}]}
        assert client.put(f"/surveys/grid/{links['participant_token']}/session", headers=headers, json=body).status_code == 400


def test_task_chart_can_be_chosen_or_dropped_by_hand(version_grid):
    manifest, variants = version_grid
    entry = manifest['games'][0]
    game = variants[f"game-01__P__{entry['level']}"]['game']
    spec = {'id': 'task-07', 'game_id': 'game-01', 'condition': 'P', 'line': 'played', 'rule': 'development', 'fragment': 'WITH A CLAIM'}
    assert 'series' not in build_task({**spec, 'series': False}, game, entry)
    chosen = build_task({**spec, 'series': {'feature': 'WHITE_ROOK_OPEN_FILE', 'label': 'Rooks on open files'}}, game, entry)
    assert (chosen['series']['label'], chosen['chart_change_ply']) == ('Rooks on open files', None)
    with pytest.raises(ValueError):
        build_task({**spec, 'series': {'feature': 'NOT_STORED'}}, game, entry)


def test_version_study_runs_without_tasks(tmp_path, monkeypatch, version_grid):
    manifest, variants = version_grid
    monkeypatch.setenv('ACA_SURVEY_DB', str(tmp_path / 'plain.sqlite3'))
    links = grid_survey.publish_grid({**manifest, 'tasks': []}, variants)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        token = links['participant_token']
        public = client.get(f'/surveys/grid/{token}').json()
        assert (public['round_count'], public['task_count'], public['task_items'], public['lines_only']) == (6, 0, [], True)
        assert len(public['criteria']) == 4 and public['duration'] == 'about 15 minutes'
        session, headers = start(client, links)
        assert len(session['rounds']) == 6 and session['tasks'] == []
        done = client.put(f'/surveys/grid/{token}/session', headers=headers, json={'version': 0, 'reviews': reviews(session), 'completed': True})
        assert done.status_code == 200 and done.json()['completed']
        data = grid_survey.results(links['results_token'])
        assert (data['completed'], data['tasks'], data['task_responses'], data['versions']) == (1, [], [], True)
