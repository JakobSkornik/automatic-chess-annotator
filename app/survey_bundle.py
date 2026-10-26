"""Ship frozen study material with the code, without access tokens or responses."""
from __future__ import annotations

import gzip
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from app import survey

BUNDLE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'surveys'
FIELDS = {'id', 'public_hash', 'admin_hash', 'manifest', 'games', 'created_at'}


def export_bundle(participant_token: str, results_token: str, destination: Path):
    """Export only one study definition. Receipts and participant rows stay local."""
    with survey.database() as db:
        row = survey._survey(db, participant_token)
        if row['admin_hash'] != survey._digest(results_token):
            raise ValueError('The results link does not belong to this survey.')
        definition = {key: row[key] for key in FIELDS}
    for key in ('manifest', 'games'):
        definition[key] = json.loads(definition[key])
    version = 1
    if definition['manifest'].get('kind') == 'comparison_grid':
        from app import grid_survey
        with grid_survey.database() as db:
            definition['variants'] = grid_survey.exported_variants(db, definition['id'])
        version = 2
    payload = json.dumps({'schema_version': version, 'survey': definition},
                         ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    compressed = gzip.compress(payload, mtime=0)
    if destination.exists() and destination.read_bytes() != compressed:
        raise FileExistsError('Use a new bundle filename for a different study.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(compressed)


def install_bundled_surveys(folder: Path | None = None) -> int:
    """Import missing surveys at startup; never replace data or participant rows."""
    installed = 0
    for path in sorted((folder or BUNDLE_DIR).glob('*.json.gz')):
        bundle = json.loads(gzip.decompress(path.read_bytes()))
        if set(bundle) != {'schema_version', 'survey'} or bundle['schema_version'] not in (1, 2):
            raise ValueError(f'Unsupported survey bundle: {path.name}')
        definition = bundle['survey']
        expected_fields = FIELDS | {'variants'} if bundle['schema_version'] == 2 else FIELDS
        if set(definition) != expected_fields:
            raise ValueError(f'Invalid survey bundle fields: {path.name}')
        uuid.UUID(definition['id'])
        datetime.fromisoformat(definition['created_at'])
        for key in ('public_hash', 'admin_hash'):
            if not re.fullmatch(r'[a-f0-9]{64}', definition[key]):
                raise ValueError(f'Invalid capability digest: {path.name}')
        manifest = definition['manifest']
        if bundle['schema_version'] == 2:
            from app.grid_survey import install_definition
            installed += install_definition(definition)
            continue
        if manifest.get('questionnaire_version') != 1 or manifest.get('questions') != survey.QUESTIONS:
            raise ValueError(f'Unsupported questionnaire: {path.name}')
        games = survey._freeze_games(manifest, definition['games'])
        with survey.database() as db, db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT * FROM surveys WHERE id=?', (definition['id'],)).fetchone()
            if existing is not None:
                if (any(existing[k] != definition[k] for k in ('public_hash', 'admin_hash', 'created_at'))
                        or json.loads(existing['manifest']) != manifest
                        or json.loads(existing['games']) != games):
                    raise ValueError(f'Existing frozen survey differs from bundle: {path.name}')
                continue
            db.execute('INSERT INTO surveys (id,public_hash,admin_hash,manifest,games,created_at) VALUES (?,?,?,?,?,?)', (
                definition['id'], definition['public_hash'], definition['admin_hash'],
                json.dumps(manifest, ensure_ascii=False), json.dumps(games, ensure_ascii=False), definition['created_at']))
            installed += 1
    return installed
