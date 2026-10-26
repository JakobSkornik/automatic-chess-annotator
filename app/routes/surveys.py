"""Participant APIs and a separately authorized researcher results API."""
import csv
import io

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app import survey


class PrivateSurveyRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def private_handler(request: Request):
            if len(await request.body()) > 64000:
                response = JSONResponse({'detail': 'Request too large.'}, status_code=413)
            else:
                try:
                    response = await handler(request)
                except survey.SurveyError as exc:
                    response = JSONResponse({'detail': str(exc)}, status_code=exc.status)
            response.headers['Cache-Control'] = 'no-store'
            response.headers['X-Robots-Tag'] = 'noindex, nofollow, noarchive'
            response.headers['Referrer-Policy'] = 'no-referrer'
            return response
        return private_handler


router = APIRouter(prefix='/surveys', tags=['Surveys'], route_class=PrivateSurveyRoute)


def participant_token(request: Request):
    value = request.headers.get('authorization', '')
    if not value.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Participant session required.')
    return value[7:]


@router.get('/{token}')
def manifest(token: str):
    return survey.public_manifest(token)


@router.get('/{token}/games/{game_id}')
def game(token: str, game_id: str):
    return survey.get_game(token, game_id)


@router.post('/{token}/session')
def start(token: str, body: survey.StartRequest):
    return survey.start_response(token, body)


@router.get('/{token}/session')
def resume(token: str, request: Request):
    return survey.get_response(token, participant_token(request))


@router.put('/{token}/session')
def save(token: str, request: Request, body: survey.SaveRequest):
    return survey.save_response(token, participant_token(request), body)


@router.get('/results/{token}/summary')
def summary(token: str):
    return survey.results(token)


@router.get('/results/{token}/export.csv')
def export_csv(token: str):
    data = survey.results(token)
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['response_id', 'participant_rating_band', 'game_id', 'game_rating_band',
                     'presentation_order', 'skipped', *[q['id'] for q in data['questions']],
                     'game_feedback', 'overall_feedback', 'submitted_at'])
    bands = {g['id']: g['bracket'] for g in data['games']}

    def safe(value):
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
            return "'" + value
        return value

    for response in data['responses']:
        for review in response['reviews']:
            writer.writerow([safe(v) for v in [response['id'], response['rating'], review['game_id'],
                bands[review['game_id']], response['order'].index(review['game_id']) + 1,
                review['skipped'], *[review['ratings'][q['id']] for q in data['questions']],
                review['feedback'], response['feedback'], response['completed_at']]])
    return Response(output.getvalue(), media_type='text/csv', headers={
        'Content-Disposition': 'attachment; filename="chess-survey-responses.csv"'})
