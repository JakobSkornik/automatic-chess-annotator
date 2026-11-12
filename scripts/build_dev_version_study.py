"""Build a LOCAL test study of the version design from a published model-comparison bundle.

For trying the survey mechanics before the real material exists. No model is called:
  P  the stored full-system comment of one model,
  T  the template rendered again from the stored facts,
  B  a marked placeholder (the model-without-claims writer does not exist yet).
The study is written to the database given with --db, never to the default one, and
must not be used to collect answers.
"""
import argparse
import copy
import gzip
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLACEHOLDER = ('[Placeholder for the model-without-claims version. The real comment is written by the same '
               'language model from the game, the evaluation and both lines, without the prepared statements.]')
TASKS = [
    {'id': 'task-01', 'game_id': 'game-05', 'condition': 'P', 'line': 'alternative', 'rule': 'fork',
     'fragment': "Black's queen checks White's king on b2 while attacking White's knight on f2"},
    {'id': 'task-02', 'game_id': 'game-07', 'condition': 'P', 'line': 'played', 'rule': 'isolated_pawn_created',
     'fragment': "White's pawn on a2 is isolated"},
]


def numbered(start_fen, sans):
    """SAN moves with move numbers, starting from the side to move in ``start_fen``."""
    parts = start_fen.split()
    white, number, out = parts[1] == 'w', int(parts[5]), []
    for i, san in enumerate(sans):
        if white:
            out.append(f'{number}.{san}')
        else:
            out.append((f'{number}…' if i == 0 else '') + san)
            number += 1
        white = not white
    return ' '.join(out)


def stored_facts(move, previous, position):
    """The facts stored with a move of a published game, as the object the composer works on."""
    import chess
    from app.models.comment_facts import CommentFacts

    stored = move['comment_facts']
    board = chess.Board(position['fen_before'])

    def line(record, root):
        record = dict(record)
        walk, ucis = chess.Board(record['start_fen']), []
        for san in record['san']:
            step = walk.parse_san(san)
            ucis.append(step.uci())
            walk.push(step)
        record.update(line_san=record['san'], line_uci=ucis, root_eval_cp=root, depth=stored.get('depth'))
        return record

    def claims(items):
        return [dict(c, features_involved=c.get('features', [])) for c in items]

    shown = stored['display_line']
    quality = stored.get('move_quality')
    facts = dict(stored, ply=position['ply'], san=move['san'], uci=move['uci'],
                 mover='White' if board.turn else 'Black', phase=move['phase'], move_number=board.fullmove_number,
                 eval_before_cp=((previous or {}).get('score') or {}).get('cp'),
                 refutation_san=shown['san'][1] if quality in ('mistake', 'blunder') and len(shown['san']) >= 2 else None,
                 display_line=line(shown, stored.get('eval_cp')), claims=claims(stored['claims']))
    alternative = stored.get('better_alternative')
    if alternative:
        facts['better_alternative'] = dict(alternative, claims=claims(alternative['claims']),
                                           display_line=line(alternative['display_line'], alternative.get('eval_cp')))
    return CommentFacts.model_validate(facts)


def template_comment(move, previous, position, level):
    """The template version of a comment, rendered from the facts stored with the move."""
    from app.core.commentary.phases.composer import _audience_view, comment_archetype, render_facts_template

    text = render_facts_template(_audience_view(stored_facts(move, previous, position), level), archetype=comment_archetype(None))
    text = re.sub(r'\[pv:([^\]]+)\]', lambda m: numbered(position['fen_before'], m.group(1).split()), text)
    return text.replace('...', '…')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path,
                        default=ROOT / 'data' / 'surveys' / 'comparison-luna-grok-composer-v18-20261105.json.gz')
    parser.add_argument('--db', required=True, type=Path, help='Test database to create the study in')
    parser.add_argument('--model', default='gpt-6-luna', help='Model of the bundle whose comments stand for version P')
    parser.add_argument('--tasks', help="'builtin' for two example tasks, or a JSON file with task specifications; "
                                        'left out, the study ends after the rating rounds')
    parser.add_argument('--task-key', choices=['support_ply', 'board_ply'], default='support_ply')
    parser.add_argument('--receipt', type=Path, help='Where to write the private links (default: next to the database)')
    args = parser.parse_args()
    # Both must be set before the application modules are imported.
    os.environ['ACA_SURVEY_DB'] = str(args.db.resolve())
    os.environ.setdefault('ACA_PROFILE', 'v18')
    from app.grid_survey import publish_grid
    from app.core.commentary import pipeline_flags
    from app.core.commentary.phases.composer import narrative
    from app.research.survey_grid import variant_id
    from app.research.survey_tasks import build_task

    source = json.loads(gzip.decompress(args.bundle.read_bytes()))['survey']
    manifest = copy.deepcopy(source['manifest'])
    manifest.update(
        questionnaire_version=4, rounds_per_type=2, task_key=args.task_key,
        title='TEST version study (v18 material, not for data collection)',
        models=[{'id': 'P', 'provider': 'openai', 'label': 'Full system'},
                {'id': 'T', 'provider': 'template', 'label': 'Template without model'},
                {'id': 'B', 'provider': 'openai', 'label': 'Model without claims'}],
        test_study={'source_bundle': args.bundle.name, 'P': f'stored comments of {args.model}',
                    'T': 'template rendered again from the stored facts', 'B': 'placeholder text'})
    variants = {}
    for entry in manifest['games']:
        position = entry['comparison_position']
        stored = source['variants'][variant_id(entry['id'], args.model, entry['level'])]
        move = stored['game']['moves'][position['move_index']]
        previous = stored['game']['moves'][position['move_index'] - 1] if position['move_index'] else None
        full = move['comment']
        if pipeline_flags.enabled('comment_blocks'):
            # The stored comment keeps its wording (one paragraph); only the move numbers of
            # the new layout are added, so the test study shows how numbered moves look.
            _, evidence = narrative.prompt(stored_facts(move, previous, position), entry['level'])
            full = narrative.number_moves(full, evidence)
        texts = {'P': full, 'T': template_comment(move, previous, position, entry['level']), 'B': PLACEHOLDER}
        for version, text in texts.items():
            game = copy.deepcopy(stored['game'])
            game['moves'][position['move_index']]['comment'] = text
            variants[variant_id(entry['id'], version, entry['level'])] = {
                'game_id': entry['id'], 'model': version, 'level': entry['level'], 'game': game,
                'provenance': {'test_study': True, 'version': version, 'source_variant': stored.get('provenance', {}).get('id')}}
    specs = [] if not args.tasks else TASKS if args.tasks == 'builtin' else json.loads(Path(args.tasks).read_text(encoding='utf-8'))
    entries = {g['id']: g for g in manifest['games']}
    manifest['tasks'] = [build_task(spec, variants[variant_id(spec['game_id'], spec['condition'], entries[spec['game_id']]['level'])]['game'],
                                    entries[spec['game_id']]) for spec in specs]
    links = publish_grid(manifest, variants)
    links.update(participant_path='/survey-grid/' + links['participant_token'],
                 results_path='/survey-grid-results/' + links['results_token'], database=str(args.db.resolve()))
    receipt = args.receipt or args.db.with_suffix('.links.private.json')
    receipt.write_text(json.dumps(links, indent=2), encoding='utf-8')
    print('Participant path: ' + links['participant_path'])
    print('PRIVATE results path: ' + links['results_path'])
    for task in manifest['tasks']:
        chart = f"chart {task['series']['label']!r} first changes after ply {task['chart_change_ply']}" if task.get('series') else 'no chart'
        print(f"{task['id']}: {task['game_id']} {task['line']} line, support ply {task['support_ply']}, "
              f"board ply {task['board_ply']}, {chart}: {task['statement']}")
    if not manifest['tasks']:
        print('No checking tasks: the study ends after the rating rounds.')


if __name__ == '__main__':
    main()
