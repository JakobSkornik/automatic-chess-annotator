"""Copy completed comparison responses to a private Google Drive folder.

SQLite stays the authoritative store; Drive is an off-server copy. A response is
marked as backed up only after Drive confirms the file, so failed uploads are
retried on the next submission, at startup, or by scripts/backup_survey_to_drive.py.
"""
import json
import logging
import os
import secrets
import threading

import httpx

from app import survey
from app.grid_survey import database

logger = logging.getLogger(__name__)

ENV = ('GOOGLE_DRIVE_CLIENT_ID', 'GOOGLE_DRIVE_CLIENT_SECRET',
       'GOOGLE_DRIVE_REFRESH_TOKEN', 'GOOGLE_DRIVE_FOLDER_ID')
TOKEN_URL = 'https://oauth2.googleapis.com/token'
UPLOAD_URL = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id'
_lock = threading.Lock()


def configured():
    return all(os.getenv(key) for key in ENV)


def _pending():
    with database() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS drive_backups (
            response_id TEXT PRIMARY KEY REFERENCES comparison_responses(id),
            file_id TEXT NOT NULL, uploaded_at TEXT NOT NULL)''')
        return db.execute('''SELECT r.* FROM comparison_responses r
            LEFT JOIN drive_backups d ON d.response_id = r.id
            WHERE r.completed_at IS NOT NULL AND d.response_id IS NULL ORDER BY r.completed_at''').fetchall()


def response_record(row):
    """Everything needed to analyse a response, without its access token hash."""
    return {'response_id': row['id'], 'survey_id': row['survey_id'], 'elo': row['elo'],
            'rating_system': row['rating_system'], 'rating_band': row['rating'],
            'assignment': json.loads(row['assignment']), 'reviews': json.loads(row['reviews']),
            'feedback': row['feedback'], 'created_at': row['created_at'], 'completed_at': row['completed_at']}


def _access_token(client):
    response = client.post(TOKEN_URL, data={
        'client_id': os.environ['GOOGLE_DRIVE_CLIENT_ID'], 'client_secret': os.environ['GOOGLE_DRIVE_CLIENT_SECRET'],
        'refresh_token': os.environ['GOOGLE_DRIVE_REFRESH_TOKEN'], 'grant_type': 'refresh_token'})
    response.raise_for_status()
    return response.json()['access_token']


def _upload(client, token, name, content):
    boundary = secrets.token_hex(16)
    metadata = {'name': name, 'parents': [os.environ['GOOGLE_DRIVE_FOLDER_ID']], 'mimeType': 'application/json'}
    body = (f'--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{json.dumps(metadata)}\r\n'
            f'--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{content}\r\n'
            f'--{boundary}--\r\n').encode('utf-8')
    response = client.post(UPLOAD_URL, content=body, headers={
        'Authorization': f'Bearer {token}', 'Content-Type': f'multipart/related; boundary={boundary}'})
    response.raise_for_status()
    return response.json()['id']


def backup_pending(transport=None):
    """Upload every completed response not yet in Drive; returns how many were uploaded."""
    if not configured():
        return 0
    with _lock:
        rows = _pending()
        if not rows:
            return 0
        uploaded = 0
        with httpx.Client(timeout=30, transport=transport) as client:
            token = _access_token(client)
            for row in rows:
                name = f"response-{row['completed_at'][:19].replace(':', '-')}-{row['id']}.json"
                content = json.dumps(response_record(row), ensure_ascii=False, indent=2)
                file_id = _upload(client, token, name, content)
                with database() as db, db:
                    db.execute('INSERT OR IGNORE INTO drive_backups VALUES (?,?,?)', (row['id'], file_id, survey._now()))
                uploaded += 1
        return uploaded


def backup_quietly():
    """Background-task entry point: never fail a participant's request over a backup."""
    try:
        backup_pending()
    except Exception as exc:
        logger.warning('Drive backup failed; responses stay queued: %s: %s', type(exc).__name__, exc)
