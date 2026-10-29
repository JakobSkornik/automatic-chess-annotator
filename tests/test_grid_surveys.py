import copy
import csv
import gzip
import io
import json

import chess
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import httpx

from app import drive_backup, grid_survey, survey
from app.research.survey_grid import LEVELS, grid_cells
from app.routes.survey_grid import router


@pytest.fixture
def prepared_grid():
    board=chess.Board(); board.push_san('e4')
    games=[]
    for i,bracket in enumerate(survey.BRACKETS):
        low,high=(0,800) if i==0 else (2400,4001) if i==9 else (600+i*200,800+i*200)
        games.append({'id':f'game-{i+1:02}', 'bracket':bracket,'min_rating':low,'max_rating_exclusive':high,
                      'comparison_position':{'move_index':0,'ply':1,'san':'e4','fen_before':chess.STARTING_FEN,'fen_after':board.fen()}})
    manifest={'games':games,'models':[{'id':'gpt-6-astra','provider':'openai','label':'GPT-6 Astra'},
                                    {'id':'grok-4.6','provider':'grok','label':'Grok 4.6'}],
              'levels':LEVELS,'kind':'comparison_grid','questionnaire_version':2,'title':'TEST COMPARISON'}
    variants={}
    for cell in grid_cells(manifest):
        entry=next(g for g in games if g['id']==cell['game_id'])
        game={'metadata':{'id':'private-job','white':'Test white','black':'Test black','result':'1-0',
                          'whiteElo':entry['min_rating']+100,'blackElo':entry['min_rating']+100,'commentary_level':cell['level']},
              'moves':[{'mn':1,'color':'w','san':'e4','uci':'e2e4','fen':board.fen(),
                        'comment':'TEST EXPLANATION ONLY','comment_kind':'llm','debug':{'private':'hidden'}}],
              'commentary_complete':True,'analysis_info':{'engine':'test','depth':1,'multipv':1,'timestamp':0}}
        variants[cell['id']]={**cell,'game':game,'provenance':{'test_fixture':True}}
    return manifest,variants


@pytest.fixture
def setup(tmp_path,monkeypatch,prepared_grid):
    monkeypatch.setenv('ACA_SURVEY_DB',str(tmp_path/'test.sqlite3'))
    links=grid_survey.publish_grid(*prepared_grid)
    app=FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        yield client,links,tmp_path


def start(client,links,elo=1500,system='Lichess'):
    public=links['participant_token']
    response=client.post(f'/surveys/grid/{public}/session',json={'consent':True,'elo':elo,'rating_system':system})
    assert response.status_code==200
    session=response.json()
    return session,{'Authorization':'Bearer '+session['token']}


def reviews(session, score=4):
    criteria = [c['id'] for c in grid_survey.CRITERIA]
    return [{'game_id':r['game_id'],'decision':'rated','feedback':'',
             'scores':{c['id']:{k:score for k in criteria} for c in r['candidates']}}
            for r in session['rounds']]


def test_grid_blinding_isolation_and_balanced_assignment(setup):
    client,links,_=setup
    token=links['participant_token']
    manifest=client.get(f'/surveys/grid/{token}')
    assert manifest.status_code==200 and 'noindex' in manifest.headers['x-robots-tag']
    assert client.get(f"/surveys/grid/results/{token}/summary").status_code==404
    assert client.post(f'/surveys/grid/{token}/session',json={'consent':False}).status_code==422
    sessions=[start(client,links) for _ in range(3)]
    for game_id in [f'game-{i:02}' for i in range(1,11)]:
        assert {next(r['level'] for r in s['rounds'] if r['game_id']==game_id) for s,_ in sessions}==set(LEVELS)
    session,headers=sessions[0]
    first=session['rounds'][0]
    response=client.get(f"/surveys/grid/{token}/round/{first['game_id']}",headers=headers)
    assert response.status_code==200
    public_data=manifest.text+json.dumps(session)+response.text
    for private in ('gpt-6-astra','grok-4.6','variant_id',links['results_token']):
        assert private not in public_data
    candidate=first['candidates'][0]['id']
    game=client.get(f'/surveys/grid/{token}/game/{candidate}',headers=headers)
    assert game.status_code==200 and 'debug' not in game.json()['moves'][0]
    assert client.get(f'/surveys/grid/{token}/game/{candidate}',headers=sessions[1][1]).status_code==404
    assert client.get(f'/surveys/grid/{token}/game/{candidate}').status_code==401
    board=client.get(f"/surveys/grid/{token}/round/{first['game_id']}/board",headers=headers)
    assert board.status_code==200
    assert all(m['comment']=='' and not m['has_comment'] for m in board.json()['moves'])
    assert board.json()['moves'][0]['fen']==game.json()['moves'][0]['fen']
    assert client.get(f"/surveys/grid/{token}/round/{first['game_id']}/board").status_code==401


def test_grid_save_ratings_versions_completion_and_statistics(setup):
    client,links,_=setup
    session,headers=start(client,links)
    url=f"/surveys/grid/{links['participant_token']}/session"
    answers=reviews(session)
    first=session['rounds'][0]['candidates'][0]['id']
    wrong=copy.deepcopy(answers); del wrong[0]['scores'][first]['useful']
    assert client.put(url,headers=headers,json={'version':0,'reviews':wrong}).status_code==400
    wrong=copy.deepcopy(answers); del wrong[0]['scores'][first]
    assert client.put(url,headers=headers,json={'version':0,'reviews':wrong}).status_code==400
    wrong=copy.deepcopy(answers); wrong[0]['scores'][first]['correct']=6
    assert client.put(url,headers=headers,json={'version':0,'reviews':wrong}).status_code==422
    wrong=copy.deepcopy(answers); wrong[0]['decision']='skipped'
    assert client.put(url,headers=headers,json={'version':0,'reviews':wrong}).status_code==400
    answers[0].update(decision='skipped',scores={})
    answers[1]['scores'][session['rounds'][1]['candidates'][0]['id']]={'understandable':1,'correct':2,'useful':3}
    answers[3]['feedback']='=FORMULA_TEST'
    partial={'version':0,'reviews':answers[:2]}
    assert client.put(url,headers=headers,json=partial).status_code==200
    assert client.put(url,headers=headers,json=partial).status_code==409
    assert client.get(url,headers=headers).json()['reviews']==answers[:2]
    assert client.put(url,headers=headers,json={'version':1,'reviews':answers[:2],'completed':True}).status_code==400
    payload={'version':1,'reviews':answers,'completed':True,'feedback':'Useful comparison'}
    assert client.put(url,headers=headers,json=payload).status_code==200
    assert client.put(url,headers=headers,json=payload).status_code==200
    start(client,links)
    data=grid_survey.results(links['results_token'])
    assert (data['completed'],data['in_progress'])==(1,1)
    assert len(data['cells'])==60
    assert sum(c['rated'] for c in data['cells'])==18 and sum(c['skipped'] for c in data['cells'])==2
    assert sum(c['criteria']['useful']['n'] for c in data['cells'])==18
    assert sum(sum(c['criteria']['understandable']['distribution']) for c in data['cells'])==18
    low=next(c for c in data['cells'] if c['criteria']['understandable']['distribution'][0])
    assert (low['criteria']['understandable']['mean'],low['criteria']['correct']['mean'])==(1,2)
    assert all(c['criteria']['useful']['mean'] is None for c in data['cells'] if not c['rated'])
    assert data['notes']==[{'game_id':answers[3]['game_id'],'level':session['rounds'][3]['level'],'text':'=FORMULA_TEST'}]
    export=client.get(f"/surveys/grid/results/{links['results_token']}/export.csv")
    rows=list(csv.DictReader(io.StringIO(export.text)))
    assert len(rows)==20 and any(r['round_feedback']=="'=FORMULA_TEST" for r in rows)
    assert {r['useful'] for r in rows if r['decision']=='rated'}=={'3','4'}
    assert {(r['participant_elo'],r['participant_rating_system'],r['participant_rating']) for r in rows}=={('1500','Lichess','1400–1599')}


def test_grid_requires_participant_elo_and_sorts_by_it(setup):
    client,links,_=setup
    url=f"/surveys/grid/{links['participant_token']}/session"
    for body in ({'consent':True},{'consent':True,'elo':1500},{'consent':True,'elo':'1500','rating_system':'FIDE'},
                 {'consent':True,'elo':50,'rating_system':'FIDE'},{'consent':True,'elo':1500,'rating_system':'Other'}):
        assert client.post(url,json=body).status_code==422
    for elo,system in ((2100,'FIDE'),(650,'Estimate (unrated)'),(2400,'Chess.com')):
        session,headers=start(client,links,elo,system)
        assert (session['elo'],session['rating_system'])==(elo,system)
        assert client.put(url,headers=headers,json={'version':0,'reviews':reviews(session),'completed':True}).status_code==200
    data=grid_survey.results(links['results_token'])
    seen=list(dict.fromkeys((r['participant_elo'],r['participant_rating']) for r in data['responses']))
    assert seen==[(650,'Under 800'),(2100,'2000–2199'),(2400,'2400+')]


def test_grid_adds_rating_columns_to_existing_database(tmp_path,monkeypatch):
    monkeypatch.setenv('ACA_SURVEY_DB',str(tmp_path/'old.sqlite3'))
    with survey.database() as db:
        db.execute('''CREATE TABLE comparison_responses (id TEXT PRIMARY KEY, survey_id TEXT NOT NULL,
            token_hash TEXT UNIQUE NOT NULL, rating TEXT NOT NULL, assignment TEXT NOT NULL,
            reviews TEXT NOT NULL DEFAULT '[]', feedback TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT)''')
        db.commit()
    with grid_survey.database() as db:
        assert {'elo','rating_system'} <= {r[1] for r in db.execute('PRAGMA table_info(comparison_responses)')}


def test_grid_bundle_preserves_definition_and_existing_answers(setup,monkeypatch):
    from app.survey_bundle import export_bundle,install_bundled_surveys
    client,links,tmp=setup
    session,headers=start(client,links)
    answers=reviews(session)
    client.put(f"/surveys/grid/{links['participant_token']}/session",headers=headers,json={'version':0,'reviews':answers,'completed':True})
    folder=tmp/'bundles'; path=folder/'comparison.json.gz'
    export_bundle(links['participant_token'],links['results_token'],path)
    raw=gzip.decompress(path.read_bytes()).decode()
    assert json.loads(raw)['schema_version']==2
    assert links['results_token'] not in raw and session['token'] not in raw and 'Clear explanation' not in raw
    assert install_bundled_surveys(folder)==0
    assert grid_survey.results(links['results_token'])['completed']==1
    monkeypatch.setenv('ACA_SURVEY_DB',str(tmp/'fresh.sqlite3'))
    assert install_bundled_surveys(folder)==1
    assert grid_survey.results(links['results_token'])['started']==0
    assert len(grid_survey.results(links['results_token'])['cells'])==60


def test_grid_refuses_missing_cells_and_changed_positions(prepared_grid):
    manifest,variants=prepared_grid
    missing=dict(variants); missing.pop(next(iter(missing)))
    with pytest.raises(ValueError,match='Every model'):
        grid_survey.freeze_grid(manifest,missing)
    changed=copy.deepcopy(variants)
    next(iter(changed.values()))['game']['moves'][0]['fen']=chess.STARTING_FEN
    with pytest.raises(ValueError,match='position'):
        grid_survey.freeze_grid(manifest,changed)


def test_spend_ledger_reserves_before_calls_and_keeps_uncertain_charges(tmp_path):
    from app.research.grid_budget import SpendLedger,BudgetExhausted,charge_bound
    ledger=SpendLedger(tmp_path/'spend.sqlite3',1)
    first=ledger.reserve('gpt-6-luna','one',.8)
    assert ledger.reserve('gpt-6-luna','two',.8) is None
    usage={'input_tokens':1000,'output_tokens':100,'cached_tokens':900}
    ledger.settle(first,'gpt-6-luna',usage)
    assert ledger.summary()['committed_upper_bound_usd']==pytest.approx(charge_bound('gpt-6-luna',usage))
    second=ledger.reserve('gpt-6-luna','two',.9)
    ledger.settle(second,'gpt-6-luna',{},error=True)
    with pytest.raises(BudgetExhausted):
        ledger.reserve('gpt-6-luna','three',.2)
    assert ledger.summary()['uncertain']==1


@pytest.fixture
def drive_env(monkeypatch):
    for key in drive_backup.ENV:
        monkeypatch.setenv(key, 'test-'+key.lower())


def fake_drive(uploads, fail=False):
    def handler(request):
        if request.url.host == 'oauth2.googleapis.com':
            assert b'refresh_token=test-google_drive_refresh_token' in request.content
            return httpx.Response(200, json={'access_token': 'access'})
        assert request.headers['authorization'] == 'Bearer access'
        if fail:
            return httpx.Response(500, json={'error': 'unavailable'})
        uploads.append(request.content.decode())
        return httpx.Response(200, json={'id': f'file-{len(uploads)}'})
    return httpx.MockTransport(handler)


def complete(client, links, elo=1500):
    session, headers = start(client, links, elo)
    url = f"/surveys/grid/{links['participant_token']}/session"
    assert client.put(url, headers=headers, json={'version':0,'reviews':reviews(session),'completed':True}).status_code == 200
    return session


def test_drive_backup_uploads_completed_responses_once(setup, drive_env):
    client, links, _ = setup
    uploads = []
    assert drive_backup.backup_pending(fake_drive(uploads)) == 0
    start(client, links)  # In progress: not backed up.
    session = complete(client, links, elo=1720)
    assert drive_backup.backup_pending(fake_drive(uploads)) == 1
    assert len(uploads) == 1 and 'test-google_drive_folder_id' in uploads[0]
    record = json.loads(uploads[0].split('\r\n\r\n')[2].split('\r\n--')[0])
    assert (record['response_id'], record['elo'], len(record['reviews'])) == (session['id'], 1720, 10)
    assert 'token_hash' not in uploads[0] and session['token'] not in uploads[0]
    assert drive_backup.backup_pending(fake_drive(uploads)) == 0


def test_drive_backup_failure_keeps_response_queued(setup, drive_env):
    client, links, _ = setup
    complete(client, links)
    with pytest.raises(httpx.HTTPStatusError):
        drive_backup.backup_pending(fake_drive([], fail=True))
    uploads = []
    assert drive_backup.backup_pending(fake_drive(uploads)) == 1 and len(uploads) == 1


def test_drive_backup_is_off_without_credentials_and_triggered_on_submit(setup, monkeypatch):
    client, links, _ = setup
    for key in drive_backup.ENV:
        monkeypatch.delenv(key, raising=False)
    complete(client, links)
    assert drive_backup.backup_pending(httpx.MockTransport(lambda r: pytest.fail('no network without credentials'))) == 0
    calls = []
    monkeypatch.setattr(drive_backup, 'backup_quietly', lambda: calls.append(1))
    session, headers = start(client, links)
    url = f"/surveys/grid/{links['participant_token']}/session"
    client.put(url, headers=headers, json={'version':0,'reviews':reviews(session)[:2]})
    assert calls == []
    client.put(url, headers=headers, json={'version':1,'reviews':reviews(session),'completed':True})
    assert calls == [1]


def test_participant_groups_follow_reported_rating():
    from app.grid_survey import participant_group
    assert [participant_group(e) for e in (None, 650, 1199, 1200, 1999, 2000, 2700)] == \
        [None, 'beginner', 'beginner', 'intermediate', 'intermediate', 'expert', 'expert']


def test_a_game_with_a_fixed_level_has_one_cell_per_model():
    from app.research.survey_grid import grid_cells
    definition = {'levels': ['beginner', 'intermediate', 'expert'],
                  'models': [{'id': 'a', 'provider': 'openai'}, {'id': 'b', 'provider': 'grok'}],
                  'games': [{'id': 'game-01', 'level': 'expert'}, {'id': 'game-02'}]}
    cells = grid_cells(definition)
    assert {(c['game_id'], c['level']) for c in cells} == {
        ('game-01', 'expert'), ('game-02', 'beginner'), ('game-02', 'intermediate'), ('game-02', 'expert')}
    assert len(cells) == 2 + 6
