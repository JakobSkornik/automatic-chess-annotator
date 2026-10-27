"""Publish ten complete, frozen analyses and print separate share/results links."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from app.survey import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--base-url', required=True, help='Frontend URL (no trailing slash)')
    args = parser.parse_args()
    receipt = args.folder / 'survey-links.private.json'
    if receipt.exists():
        links = json.loads(receipt.read_text(encoding='utf-8'))
    else:
        manifest = json.loads((args.folder / 'generation.json').read_text(encoding='utf-8'))
        if not manifest.get('generation', {}).get('completed_at'):
            raise SystemExit('Generation has not completed; no survey was published.')
        games = {}
        for entry in manifest['games']:
            path = args.folder / f"{entry['id']}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['artifact_sha256']:
                raise SystemExit(f"Saved analysis changed: {entry['id']}")
            games[entry['id']] = json.loads(path.read_text(encoding='utf-8'))
        links = publish(manifest, games)
        receipt.write_text(json.dumps(links, indent=2), encoding='utf-8')
    print('Participant link: ' + args.base_url.rstrip('/') + '/survey/' + links['participant_token'])
    print('PRIVATE results link: ' + args.base_url.rstrip('/') + '/survey-results/' + links['results_token'])


if __name__ == '__main__':
    main()
