"""Package a published survey for the application's normal merge deployment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from app.survey_bundle import export_bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, help='Preparation folder containing survey-links.private.json')
    parser.add_argument('--output', type=Path, required=True, help='Repository bundle path ending in .json.gz')
    args = parser.parse_args()
    if not args.output.name.endswith('.json.gz'):
        parser.error('--output must end in .json.gz')
    receipt = json.loads((args.folder / 'survey-links.private.json').read_text(encoding='utf-8'))
    export_bundle(receipt['participant_token'], receipt['results_token'], args.output)
    print(f'Exported frozen survey to {args.output}; no raw access tokens or participant responses included.')
