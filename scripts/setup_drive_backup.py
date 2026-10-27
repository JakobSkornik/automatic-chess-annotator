"""One-time authorisation of a personal Google account for survey response backups.

Download a "Desktop app" OAuth client JSON from Google Cloud (see docs/survey.md),
then run:  python scripts/setup_drive_backup.py google-client.json
A browser opens for consent; the script creates the backup folder in your Drive
and prints the four GOOGLE_DRIVE_* values to set on the survey server.
"""
import argparse
import base64
import hashlib
import http.server
import json
import secrets
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

SCOPE = 'https://www.googleapis.com/auth/drive.file'
AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
TOKEN_URL = 'https://oauth2.googleapis.com/token'


def authorise(client_id, client_secret):
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    state = secrets.token_urlsafe(16)
    received = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(self.path).query))
            if query.get('state') != state:
                self.send_response(400); self.end_headers(); return
            received.update(query)
            self.send_response(200); self.send_header('Content-Type', 'text/plain; charset=utf-8'); self.end_headers()
            self.wfile.write('Authorisation received. You can close this tab.'.encode())

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    redirect = f'http://127.0.0.1:{server.server_port}'
    url = AUTH_URL + '?' + urllib.parse.urlencode({
        'client_id': client_id, 'redirect_uri': redirect, 'response_type': 'code', 'scope': SCOPE,
        'access_type': 'offline', 'prompt': 'consent', 'state': state,
        'code_challenge': challenge, 'code_challenge_method': 'S256'})
    print('Opening the browser for consent. If it does not open, visit:\n' + url)
    webbrowser.open(url)
    while not received:
        server.handle_request()
    if 'code' not in received:
        raise SystemExit('Authorisation was not granted: ' + received.get('error', 'unknown error'))
    response = httpx.post(TOKEN_URL, data={
        'client_id': client_id, 'client_secret': client_secret, 'code': received['code'],
        'code_verifier': verifier, 'grant_type': 'authorization_code', 'redirect_uri': redirect})
    response.raise_for_status()
    tokens = response.json()
    if 'refresh_token' not in tokens:
        raise SystemExit('Google returned no refresh token; remove the app at myaccount.google.com/permissions and retry.')
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('client_json', type=Path)
    parser.add_argument('--folder-name', default='Chess survey responses')
    args = parser.parse_args()
    client = json.loads(args.client_json.read_text(encoding='utf-8'))
    client = client.get('installed') or client.get('web') or {}
    if not client.get('client_id') or not client.get('client_secret'):
        raise SystemExit('Expected a Google OAuth client JSON (Desktop app type).')
    tokens = authorise(client['client_id'], client['client_secret'])
    folder = httpx.post('https://www.googleapis.com/drive/v3/files?fields=id,webViewLink',
                        headers={'Authorization': f"Bearer {tokens['access_token']}"},
                        json={'name': args.folder_name, 'mimeType': 'application/vnd.google-apps.folder'})
    folder.raise_for_status()
    folder = folder.json()
    print(f"\nCreated Drive folder: {folder.get('webViewLink', folder['id'])}")
    print('\nSet these on the survey server (keep them secret, like a password):\n')
    print(f"GOOGLE_DRIVE_CLIENT_ID={client['client_id']}")
    print(f"GOOGLE_DRIVE_CLIENT_SECRET={client['client_secret']}")
    print(f"GOOGLE_DRIVE_REFRESH_TOKEN={tokens['refresh_token']}")
    print(f"GOOGLE_DRIVE_FOLDER_ID={folder['id']}")


if __name__ == '__main__':
    main()
