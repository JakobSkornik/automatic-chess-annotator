"""Private comparison-grid participant and researcher endpoints."""
import csv
import io

from fastapi import APIRouter, BackgroundTasks, Request, Response

from app import drive_backup, grid_survey, survey
from app.routes.surveys import PrivateSurveyRoute, participant_token

router = APIRouter(prefix='/surveys/grid', tags=['Comparison surveys'], route_class=PrivateSurveyRoute)


@router.get('/{token}')
def manifest(token: str):
    return grid_survey.public_manifest(token)


@router.post('/{token}/session')
def start(token: str, body: grid_survey.GridStartRequest):
    return grid_survey.start_response(token, body)


@router.get('/{token}/session')
def resume(token: str, request: Request):
    return grid_survey.get_response(token, participant_token(request))


@router.put('/{token}/session')
def save(token: str, request: Request, body: grid_survey.ComparisonSave, background: BackgroundTasks):
    saved = grid_survey.save_response(token, participant_token(request), body)
    if saved['completed']:
        background.add_task(drive_backup.backup_quietly)
    return saved


@router.get('/{token}/round/{game_id}')
def round_data(token: str, game_id: str, request: Request):
    return grid_survey.get_round(token, participant_token(request), game_id)


@router.get('/{token}/round/{game_id}/board')
def round_board(token: str, game_id: str, request: Request):
    return grid_survey.get_round_board(token, participant_token(request), game_id)


@router.get('/{token}/task/{task_id}')
def task_data(token: str, task_id: str, request: Request):
    return grid_survey.get_task(token, participant_token(request), task_id)


@router.get('/{token}/game/{candidate_id}')
def candidate_game(token: str, candidate_id: str, request: Request):
    return grid_survey.get_candidate_game(token, participant_token(request), candidate_id)


@router.get('/results/{token}/summary')
def summary(token: str):
    return grid_survey.results(token)


def _csv(rows, fields, filename):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "'" + value if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')) else value
                         for key, value in row.items()})
    return Response(output.getvalue(), media_type='text/csv', headers={
        'Content-Disposition': f'attachment; filename="{filename}"'})


@router.get('/results/{token}/export.csv')
def export_csv(token: str):
    data = grid_survey.results(token)
    fields = ['response_id', 'participant_elo', 'participant_rating_system', 'participant_rating', 'game_id', 'position_type',
              'model', 'level', 'round_order', 'candidate_order', 'decision', *[c['id'] for c in data['criteria']],
              'round_feedback', 'overall_feedback', 'submitted_at']
    return _csv(data['responses'], fields, 'chess-comparison-responses.csv')


@router.get('/results/{token}/tasks.csv')
def export_tasks_csv(token: str):
    fields = ['response_id', 'participant_elo', 'participant_rating_system', 'participant_rating', 'task_id', 'game_id',
              'position_type', 'line', 'rule_id', 'task_order', 'rated_position', 'line_plies', 'answer_ply', 'support_ply',
              'board_ply', 'chart_change_ply', 'correct_support_ply', 'correct_board_ply', 'correct', 'could_check', 'trust', 'elapsed_ms',
              'submitted_at']
    return _csv(grid_survey.results(token)['task_responses'], fields, 'chess-comparison-tasks.csv')
