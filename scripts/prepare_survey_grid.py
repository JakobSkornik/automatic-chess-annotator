"""Freeze the comparison grid and run fresh engine analysis, without any model calls.

Engine states are private, locally-created pickle files. Never load a cache
from another source. Source and cache hashes are verified by the generation runner.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from app.research.survey_grid import MODELS, LEVELS, ROOT, source_record, run_settings, versions
from app.core.commentary import pipeline_flags
from app.core.engine.analysis_retriever import run_engine_analysis_to_json, assemble_game_json, rebuild_comment_facts
from app.core.engine.engine_connector import get_global_engine_connector, _shutdown_global_engine
from app.core.commentary.pipeline.game_pipeline import _promote_turning_points, _key_moment_order, _coverage_order
from app.core.engine.analysis.moment_kinds import complexity, moment_kind
from app.core.engine.analysis_retriever import _apply_back_to_back_key_moment_suppression
from app.core.commentary.phases.composer.narrative import VERSION


def _flavour(event) -> str:
    """'tactical' when a tactic appears in the played or the better line, else 'positional'."""
    from app.core.engine.analysis.moment_kinds import TACTIC_IN_LINE
    facts = event.comment_facts
    lines = [facts] + ([facts.better_alternative] if facts.better_alternative is not None else [])
    return 'tactical' if any(c.rule_id in TACTIC_IN_LINE for record in lines for c in record.claims) else 'positional'


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


async def prepare(args):
    async def progress(percentage, message):
        await asyncio.sleep(0)

    selection = json.loads((args.source / 'selection.json').read_text(encoding='utf-8'))
    chosen = [m for m in MODELS if not args.models or m['id'] in args.models]
    if args.models and set(args.models) != {m['id'] for m in chosen}:
        raise ValueError('Unknown model in requested grid')
    if args.versions:
        # Version study: one language model, three ways of writing the comment.
        model = next((m for m in MODELS if m['id'] == args.versions), None)
        if model is None or args.models or not args.positions:
            raise ValueError('--versions needs a known model, no --models and fixed --positions with their types')
        chosen = versions(model)
        args.level = args.level or 'expert'
    args.folder.mkdir(parents=True, exist_ok=True)
    source = source_record()
    # Optional fixed positions: {game_id: {"move_index": int | "rule": "earliest_key_moment",
# "type": str}, "_description": str}.
    fixed = json.loads(args.positions.read_text(encoding='utf-8')) if args.positions else None
    reused = json.loads((args.reuse_engine_from / 'grid.json').read_text(encoding='utf-8')) if args.reuse_engine_from else None
    definition = {**selection, 'kind': 'comparison_grid', 'models': chosen, 'levels': LEVELS,
                  'settings': run_settings(args.effort), 'architecture': source,
                  'prompt_version': VERSION, 'prepared_at': datetime.now(timezone.utc).isoformat(),
                  'position_selection': fixed['_description'] if fixed else 'Highest absolute engine evaluation swing among eligible key moments; ties use earliest ply. Chosen before model generation.',
                  'stop_at_comparison': args.stop_at_comparison,
                  # Questionnaire version 4 (app.grid_survey.VERSION_STUDY): versions, not models.
                  **({'questionnaire_version': 4, 'rounds_per_type': 2, 'tasks': [], 'task_count': 0,
                      'language_model': args.versions} if args.versions else {}),
                  **({'fixed_level': args.level} if args.level else {}),
                  **({'engine_evidence_from': {'folder': args.reuse_engine_from.name,
                      'engine_sha256': {g['id']: g['engine_sha256'] for g in reused['games']
                                        if any(s['id'] == g['id'] and s['pgn_sha256'] == g['pgn_sha256'] for s in selection['games'])},
                      'note': 'Engine search reused for the listed games; move events and comment facts rebuilt with this '
                              'study code. Games not listed got a fresh engine analysis.'}}
                     if reused else {}),
                  'generation_scope': (('Conversation from the first key moment up to the comparison position; later moves are not commented. '
                                        if pipeline_flags.enabled('writer_forward_order') else
                                        'Reverse-order conversation from the last key moment down to the comparison position; earlier moves are not commented. ')
                                       if args.stop_at_comparison else 'Complete game conversations; ')
                                      + 'identical frozen engine evidence across all model/level variants.'}
    manifest_path = args.folder / 'grid.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        for key in ('models', 'levels', 'settings', 'architecture'):
            if previous[key] != definition[key]:
                raise ValueError(f'Grid {key} changed; choose a new preparation folder')
        if [(g['id'], g['pgn_sha256']) for g in previous['games']] != [(g['id'], g['pgn_sha256']) for g in selection['games']]:
            raise ValueError('Game selection changed')
        definition = previous
    else:
        save_json(manifest_path, definition)
        with zipfile.ZipFile(args.folder / 'analysis-source.private.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for relative in source['files']:
                archive.write(ROOT / relative, relative)
    engines = args.folder / 'engine'
    engines.mkdir(exist_ok=True)
    for entry in definition['games']:
        pgn = (args.source / f"{entry['id']}.pgn").read_text(encoding='utf-8')
        if hashlib.sha256(pgn.encode()).hexdigest() != entry['pgn_sha256']:
            raise ValueError('Source PGN changed')
        (args.folder / f"{entry['id']}.pgn").write_text(pgn, encoding='utf-8')
        cache = engines / f"{entry['id']}.private.pkl"
        if entry.get('engine_sha256'):
            if not cache.exists() or hashlib.sha256(cache.read_bytes()).hexdigest() != entry['engine_sha256']:
                raise ValueError('Engine cache changed')
            print(f"Reusing verified engine evidence for {entry['id']}", flush=True)
            continue
        source_entry = next((g for g in reused['games'] if g['id'] == entry['id']
                             and g['pgn_sha256'] == entry['pgn_sha256']), None) if reused else None
        if source_entry:
            # Identical engine evidence from another study; only events and facts are rebuilt.
            raw = (args.reuse_engine_from / 'engine' / f"{entry['id']}.private.pkl").read_bytes()
            if hashlib.sha256(raw).hexdigest() != source_entry['engine_sha256']:
                raise ValueError(f"{entry['id']}: reused engine cache does not match its study")
            print(f"Reused engine analysis, facts rebuilt: {entry['id']} ({entry['bracket']})", flush=True)
            state = pickle.loads(raw)  # locally created cache, hash-verified against its grid.json
            rebuild_comment_facts(state, get_global_engine_connector())
        else:
            print(f"Fresh engine analysis: {entry['id']} ({entry['bracket']})", flush=True)
            _, state = await run_engine_analysis_to_json(pgn, get_global_engine_connector(), progress, metadata_id=entry['id'])
        # Identify the displayed comparison position before any model sees it.
        candidate_state = copy.copy(state)
        candidate_state.move_events = copy.deepcopy(state.move_events)
        candidate_state.context = copy.deepcopy(state.context)
        candidate_state.context.move_events = candidate_state.move_events
        _promote_turning_points(candidate_state.move_events, candidate_state.context)
        _apply_back_to_back_key_moment_suppression(candidate_state.move_events, candidate_state.context)
        level = args.level or (fixed.get(entry['id'], {}).get('level') if fixed else None)
        if level:
            entry['level'] = level  # this game is written and surveyed at this level only
        if level:
            candidates, _covered = _coverage_order(candidate_state.move_events, 'both', level)
        else:
            candidates = _key_moment_order(candidate_state.move_events, 'both')
        if not candidates:
            raise ValueError(f"No eligible comparison position for {entry['id']}")
        if fixed:
            chosen_position = fixed[entry['id']]
            if chosen_position.get('rule') == 'most_complex':
                # The most demanding open position among the moves this level comments
                # (moment_kinds.complexity); ties go to the earlier move.
                scored = [(complexity(candidate_state.move_events, i), i) for i in candidates]
                scored = [(s, i) for s, i in scored if s is not None]
                if not scored:
                    raise ValueError(f"{entry['id']}: no open, non-trivial position for level {level}")
                score, index = max(scored, key=lambda x: (x[0], -candidate_state.move_events[x[1]].ply))
                chosen_position = {**chosen_position, 'type': moment_kind(candidate_state.move_events, index),
                                   'complexity': score}
            elif chosen_position.get('rule') == 'by_kind':
                # The level's preferred kind of moment first, then the largest verified loss.
                kinds = {i: moment_kind(candidate_state.move_events, i) for i in candidates}
                order = chosen_position['prefer']
                preferred = [i for i in candidates if kinds[i] in order]
                if not preferred:
                    raise ValueError(f"{entry['id']}: no {order} moment for level {level}")
                # Only positions the selector accepts (facts present, a merit for sound moves,
                # not a lost-position error), then the requested flavour: 'tactical' needs a
                # tactic in either line, 'positional' needs facts without one.
                usable = [i for i in preferred if complexity(candidate_state.move_events, i) is not None]
                preferred = usable or preferred
                flavour = chosen_position.get('flavour')
                if flavour:
                    flavoured = [i for i in preferred if _flavour(candidate_state.move_events[i]) == flavour]
                    if flavoured:
                        preferred = flavoured
                    else:
                        print(f"{entry['id']}: no {flavour} {order} moment; flavour ignored", flush=True)
                def rank(i):
                    event = candidate_state.move_events[i]
                    loss = (event.comment_facts.selection_metadata.get('comment_selection') or {}).get('verified_loss_cp') or 0
                    return (order.index(kinds[i]), -max(loss, abs(event.eval_swing_cp or 0)), event.ply)
                index = min(preferred, key=rank)
                chosen_position = {**chosen_position, 'type': kinds[index]}
            elif chosen_position.get('rule') == 'earliest_key_moment':
                # First commentable key moment after the opening book.
                index = min(candidates, key=lambda i: candidate_state.move_events[i].ply)
            else:
                index = next((i for i in candidates if candidate_state.move_events[i].move_index == chosen_position['move_index']), None)
            if index is None:
                raise ValueError(f"{entry['id']}: fixed position {chosen_position.get('move_index')} is not an eligible key moment; eligible: "
                                 f"{sorted(candidate_state.move_events[i].move_index for i in candidates)}")
        else:
            index = max(candidates, key=lambda i: (abs(candidate_state.move_events[i].eval_swing_cp or 0), -candidate_state.move_events[i].ply))
        event = candidate_state.move_events[index]
        entry['comparison_position'] = {'move_index': event.move_index, 'ply': event.ply,
                                        'san': event.san, 'fen_before': event.fen_before, 'fen_after': event.fen_after,
                                        **({'type': chosen_position['type']} if fixed else {}),
                                        **({'complexity': chosen_position['complexity']} if fixed and 'complexity' in chosen_position else {})}
        baseline = assemble_game_json(state).model_dump(mode='json')
        save_json(engines / f"{entry['id']}.json", baseline)
        # The commentary phase does not use the retriever. Omit its live engine
        # process; each worker unpickles an independent copy of all other state.
        state.retriever = None
        cache.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        entry['engine_sha256'] = hashlib.sha256(cache.read_bytes()).hexdigest()
        save_json(manifest_path, definition)
    if source_record()['sha256'] != source['sha256']:
        raise RuntimeError('Analysis source changed during preparation; use a new folder')
    definition['engine_preparation_completed_at'] = datetime.now(timezone.utc).isoformat()
    save_json(manifest_path, definition)
    print(f"Prepared {len(chosen)} {'versions' if args.versions else 'models'} x {len(definition['games'])} games x "
          f"{1 if args.level else 3} level(s). No model API calls made.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--folder', required=True, type=Path)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--effort', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--versions', metavar='MODEL',
                        help='Version study on this language model: P full system, T template, B model without claims')
    parser.add_argument('--level', choices=LEVELS,
                        help='Write and survey every game at this level (default for --versions: expert)')
    parser.add_argument('--positions', type=Path, help='JSON file fixing the comparison position of each game')
    parser.add_argument('--reuse-engine-from', type=Path,
                        help='Study folder whose verified engine caches are reused (facts are rebuilt)')
    parser.add_argument('--stop-at-comparison', action='store_true',
                        help='Comment only from the last key moment down to the comparison position')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.folder / 'engine-preparation.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    try:
        asyncio.run(prepare(args))
    finally:
        _shutdown_global_engine()
