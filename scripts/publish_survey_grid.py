"""Publish only a fully generated model/game/level grid, then bundle it for deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1]/'.env')
from app.grid_survey import publish_grid
from app.research.survey_grid import grid_cells
from app.survey_bundle import export_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--review-scope',choices=['position','full_game'],default='position')
    args=parser.parse_args()
    definition=json.loads((args.folder/'grid.json').read_text(encoding='utf-8'))
    variants={}
    records=[]
    for cell in grid_cells(definition):
        directory=args.folder/'variants'/cell['id']
        record_path=directory/'record.json'
        if not record_path.exists():
            raise SystemExit('Grid is incomplete: '+cell['id'])
        record=json.loads(record_path.read_text(encoding='utf-8'))
        path=directory/'game.json'
        if record['status']!='complete' or hashlib.sha256(path.read_bytes()).hexdigest()!=record['artifact_sha256']:
            raise SystemExit('Grid variant is incomplete or changed: '+cell['id'])
        if record['source_sha256']!=definition['architecture']['sha256'] or record['settings']!=definition['settings']:
            raise SystemExit('Grid variant uses another architecture or configuration: '+cell['id'])
        variants[cell['id']]={**cell,'game':json.loads(path.read_text(encoding='utf-8')),'provenance':record}
        records.append(record)
    receipt=args.folder/'comparison-links.private.json'
    if receipt.exists():
        links=json.loads(receipt.read_text(encoding='utf-8'))
    else:
        definition.update(review_scope=args.review_scope, generation={'completed_at':datetime.now(timezone.utc).isoformat(),
                          'variant_count':len(variants),'variants':records})
        links=publish_grid(definition,variants)
        links.update(participant_path='/survey-grid/'+links['participant_token'],results_path='/survey-grid-results/'+links['results_token'])
        receipt.write_text(json.dumps(links,indent=2),encoding='utf-8')
    export_bundle(links['participant_token'],links['results_token'],args.bundle)
    print('Participant path: '+links['participant_path'])
    print('PRIVATE results path: '+links['results_path'])


if __name__=='__main__':
    main()
