import csv
import gzip
import io
import json
import secrets

import chess
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import survey
from app.routes.surveys import router


@pytest.fixture
def frozen_set():
    entries, games = [], {}
    for i, bracket in enumerate(survey.BRACKETS):
        lo, hi = (0, 800) if i == 0 else (2400, 4001) if i == 9 else (600 + i*200, 800 + i*200)
        game_id = f'game-{i+1:02}'
        entries.append({'id': game_id, 'bracket': bracket, 'min_rating': lo, 'max_rating_exclusive': hi})
        board = chess.Board(); board.push_san('e4')
        games[game_id] = {
            'metadata': {'id': 'private-job-id', 'white': 'Test white', 'black': 'Test black',
                         'result': '1-0', 'whiteElo': lo+100, 'blackElo': lo+100},
            'moves': [{'mn': 1, 'color': 'w', 'san': 'e4', 'uci': 'e2e4', 'fen': board.fen(),
                       'comment': 'TEST FIXTURE ONLY', 'comment_kind': 'llm',
                       'debug': {'private': 'hidden'}, 'llm_debug': {'prompt': 'hidden'}}],
            'commentary_complete': True,
            'analysis_info': {'engine': 'test', 'depth': 1, 'multipv': 1, 'timestamp': 0},
        }
    return {'games': entries, 'seed': 123}, games


@pytest.fixture
def setup(tmp_path, monkeypatch, frozen_set):
    monkeypatch.setenv('ACA_SURVEY_DB', str(tmp_path / 'test.sqlite3'))
    manifest, games = frozen_set
    links = survey.publish(manifest, games)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        yield client, links, tmp_path


def start(client, links):
    public = links['participant_token']
    response = client.post(f'/surveys/{public}/session', json={'consent': True})
    assert response.status_code == 200
    data = response.json()
    return public, data, {'Authorization': f"Bearer {data['token']}"}


def reviews(session, score=4):
    return [{'game_id': id, 'skipped': False, 'ratings': {q['id']: score for q in survey.QUESTIONS}, 'feedback': ''} for id in session['order']]


def test_capabilities_and_public_snapshot(setup):
    client, links, _ = setup
    public, admin = links['participant_token'], links['results_token']
    assert client.get(f'/surveys/results/{public}/summary').status_code == 404
    assert client.get(f'/surveys/{admin}').status_code == 404
    assert client.get(f'/surveys/{secrets.token_urlsafe(32)}').status_code == 404
    response = client.get(f'/surveys/{public}')
    assert response.headers['cache-control'] == 'no-store'
    assert 'noindex' in response.headers['x-robots-tag']
    assert admin not in response.text and 'responses' not in response.json()
    game = client.get(f'/surveys/{public}/games/game-01').json()
    assert game['commentary_complete'] and game['metadata']['id'] == 'game-01'
    assert 'debug' not in game['moves'][0] and 'llm_debug' not in game['moves'][0]
    assert client.get(f'/surveys/{public}/games/unknown').status_code == 404


def test_consent_validation_and_session_isolation(setup, frozen_set):
    client, links, _ = setup
    assert client.post(f"/surveys/{links['participant_token']}/session", json={'consent': False}).status_code == 422
    public, session, headers = start(client, links)
    assert len(set(session['order'])) == 10
    assert client.get(f'/surveys/{public}/session').status_code == 401
    assert client.get(f'/surveys/{public}/session', headers={'Authorization': f'Bearer {public}'}).status_code == 404
    other = survey.publish(*frozen_set)
    assert client.get(f"/surveys/{other['participant_token']}/session", headers=headers).status_code == 404


def test_resume_versions_completion_and_no_duplicates(setup):
    client, links, _ = setup
    public, session, headers = start(client, links)
    url = f'/surveys/{public}/session'
    initial = {'version': 0, 'reviews': reviews(session)[:1]}
    saved = client.put(url, headers=headers, json=initial)
    assert saved.status_code == 200 and saved.json()['version'] == 1
    assert client.get(url, headers=headers).json()['reviews'] == initial['reviews']
    assert client.put(url, headers=headers, json=initial).status_code == 409
    assert client.put(url, headers=headers, json={**initial, 'version': 1, 'completed': True}).status_code == 400
    complete = {'version': 1, 'reviews': reviews(session), 'completed': True}
    assert client.put(url, headers=headers, json=complete).status_code == 200
    assert client.put(url, headers=headers, json=complete).status_code == 200
    assert client.put(url, headers=headers, json={**complete, 'feedback': 'changed'}).status_code == 409
    results = client.get(f"/surveys/results/{links['results_token']}/summary").json()
    assert results['started'] == results['completed'] == 1


@pytest.mark.parametrize('score', [0, 6, '5', True, 2.5])
def test_reject_invalid_scores(setup, score):
    client, links, _ = setup
    public, session, headers = start(client, links)
    assert client.put(f'/surveys/{public}/session', headers=headers, json={'version': 0, 'reviews': reviews(session, score)}).status_code == 422


def test_results_denominators_and_formula_safe_csv(setup):
    client, links, _ = setup
    public, session, headers = start(client, links)
    answers = reviews(session)
    answers[0].update(skipped=True, ratings={q['id']: None for q in survey.QUESTIONS})
    answers[1]['ratings']['clarity'] = None
    answers[1]['feedback'] = '=HYPERLINK("bad")'
    payload = {'version': 0, 'reviews': answers, 'completed': True, 'feedback': 'Useful study'}
    assert client.put(f'/surveys/{public}/session', headers=headers, json=payload).status_code == 200
    start(client, links)  # unfinished session must not affect means
    data = survey.results(links['results_token'])
    assert (data['completed'], data['in_progress']) == (1, 1)
    skipped = next(g for g in data['games'] if g['id'] == answers[0]['game_id'])
    assert skipped['skipped'] == 1 and skipped['metrics']['clarity']['mean'] is None
    unknown = next(g for g in data['games'] if g['id'] == answers[1]['game_id'])
    assert unknown['metrics']['clarity']['cannot_judge'] == 1
    assert unknown['metrics']['usefulness']['mean'] == 4
    export = client.get(f"/surveys/results/{links['results_token']}/export.csv")
    rows = list(csv.DictReader(io.StringIO(export.text)))
    assert len(rows) == 10
    assert next(r for r in rows if r['game_feedback'])['game_feedback'].startswith("'=HYPERLINK")


def test_reject_duplicate_or_unknown_game_and_skipped_scores(setup):
    client, links, _ = setup
    public, session, headers = start(client, links)
    row = reviews(session)[0]
    url = f'/surveys/{public}/session'
    for entries in [[row, row], [{**row, 'game_id': 'game-99'}], [{**row, 'skipped': True}]]:
        assert client.put(url, headers=headers, json={'version': 0, 'reviews': entries}).status_code == 400


def test_publish_rejects_unfinished_games_or_wrong_bands(setup, frozen_set):
    _, _, _ = setup
    manifest, games = frozen_set
    games['game-01']['commentary_complete'] = False
    with pytest.raises(ValueError, match='completed'):
        survey.publish(manifest, games)
    games['game-01']['commentary_complete'] = True
    games['game-01']['metadata']['whiteElo'] = 2600
    with pytest.raises(ValueError, match='band mismatch'):
        survey.publish(manifest, games)


def test_bundle_deploy_and_redeploy_preserve_links_and_responses(setup, monkeypatch):
    from app.survey_bundle import export_bundle, install_bundled_surveys
    client, links, tmp_path = setup
    public, session, headers = start(client, links)
    client.put(f'/surveys/{public}/session', headers=headers, json={
        'version': 0, 'reviews': reviews(session), 'completed': True,
        'feedback': 'PRIVATE RESPONSE MUST NOT SHIP',
    })
    bundles = tmp_path / 'bundles'
    path = bundles / 'study.json.gz'
    export_bundle(public, links['results_token'], path)
    raw = gzip.decompress(path.read_bytes()).decode()
    for private in (public, links['results_token'], session['token'], 'PRIVATE RESPONSE MUST NOT SHIP'):
        assert private not in raw
    assert set(json.loads(raw)['survey']) == {'id', 'public_hash', 'admin_hash', 'manifest', 'games', 'created_at'}

    monkeypatch.setenv('ACA_SURVEY_DB', str(tmp_path / 'deployed.sqlite3'))
    assert install_bundled_surveys(bundles) == 1
    assert len(survey.public_manifest(public)['games']) == 10
    assert survey.results(links['results_token'])['started'] == 0
    _, deployed_session, deployed_headers = start(client, links)
    assert client.put(f'/surveys/{public}/session', headers=deployed_headers, json={
        'version': 0, 'reviews': reviews(deployed_session), 'completed': True,
    }).status_code == 200
    assert install_bundled_surveys(bundles) == 0
    assert survey.results(links['results_token'])['completed'] == 1
    assert survey.get_response(public, deployed_session['token'])['completed']


def test_bundle_refuses_changes_to_a_live_study(setup):
    from app.survey_bundle import export_bundle, install_bundled_surveys
    _, links, tmp_path = setup
    path = tmp_path / 'study.json.gz'
    export_bundle(links['participant_token'], links['results_token'], path)
    bundle = json.loads(gzip.decompress(path.read_bytes()))
    bundle['survey']['games']['game-01']['moves'][0]['comment'] = 'Changed stimulus'
    path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
    with pytest.raises(ValueError, match='differs from bundle'):
        install_bundled_surveys(tmp_path)
    assert survey.get_game(links['participant_token'], 'game-01')['moves'][0]['comment'] == 'TEST FIXTURE ONLY'


def test_survey_app_installs_bundle_on_startup(setup, monkeypatch):
    from app import survey_bundle
    from app.survey_app import app
    _, links, tmp_path = setup
    bundles = tmp_path / 'bundles'
    survey_bundle.export_bundle(links['participant_token'], links['results_token'], bundles / 'study.json.gz')
    monkeypatch.setattr(survey_bundle, 'BUNDLE_DIR', bundles)
    monkeypatch.setenv('ACA_SURVEY_DB', str(tmp_path / 'fresh-start.sqlite3'))
    with TestClient(app) as client:
        assert client.get('/surveys/' + links['participant_token']).status_code == 200
        assert client.get('/jobs').status_code == 404
    with TestClient(app) as client:
        assert client.get('/surveys/results/' + links['results_token'] + '/summary').json()['started'] == 0
