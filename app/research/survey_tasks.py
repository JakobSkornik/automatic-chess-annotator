"""Statement-checking tasks of the version study, frozen from the stored facts of a position."""
from __future__ import annotations

from app.core.commentary.features.guid_features.feature_catalog import CATALOG


def feature_label(feature):
    """Chart title of a feature series, e.g. ``WHITE_PAWN_ISOLATED`` -> ``White: Isolated pawns``."""
    for prefix, side in (('WHITE_', 'White: '), ('BLACK_', 'Black: ')):
        if feature.startswith(prefix) and feature[len(prefix):] in CATALOG:
            return side + CATALOG[feature[len(prefix):]].label
    return CATALOG[feature].label if feature in CATALOG else feature.replace('_', ' ').capitalize()


def _chart(spec, claim, line):
    """The statement's own feature along the line, as the application charts it.

    Without instructions this is the first feature of the claim whose stored series
    changes along the line; tactical statements have no series and so no chart.
    ``series: false`` in the specification drops the chart, a mapping picks the feature
    (and optionally the label) by hand.
    """
    wanted = spec.get('series')
    if wanted is False:
        return None
    stored = line.get('feature_series') or {}
    for feature in [wanted['feature']] if wanted else claim.get('features', []):
        values = stored.get(feature)
        if values is None:
            if wanted:
                raise ValueError(f"{spec['id']}: no stored series for {feature}")
            continue
        if wanted or len(set(values)) > 1:
            return {'feature': feature, 'label': (wanted or {}).get('label') or feature_label(feature), 'values': values}
    return None


def build_task(spec, game, entry):
    """Freeze one task: a statement of the shown comment, its line, its chart and the candidate answers.

    ``spec`` names the position (``game_id``), the version whose comment is shown
    (``condition``), the line (``played`` or ``alternative``), the rule of the claim
    (``rule``; ``statement`` picks one when a rule fired more than once) and the words
    of the comment that carry the claim (``fragment``). ``board_ply`` is the ply after
    which the statement first holds on the board; it defaults to the claim's support
    ply. ``chart_change_ply`` records where the charted feature first leaves its
    starting value, which is what a reader of the chart will point at.
    """
    move = game['moves'][entry['comparison_position']['move_index']]
    facts = move.get('comment_facts') or {}
    source = facts if spec['line'] == 'played' else (facts.get('better_alternative') or {})
    line = source.get('display_line') or {}
    claims = [c for c in source.get('claims', [])
              if c['rule_id'] == spec['rule'] and spec.get('statement') in (None, c['text'])]
    if len(claims) != 1:
        raise ValueError(f"{spec['id']}: {len(claims)} claims match rule {spec['rule']} on the {spec['line']} line")
    claim = claims[0]
    task = {'id': spec['id'], 'game_id': entry['id'], 'condition': spec['condition'], 'line': spec['line'],
            'rule_id': claim['rule_id'], 'statement': claim['text'], 'fragment': spec['fragment'],
            'squares': (claim.get('evidence') or {}).get('squares', []),
            'start_fen': line['start_fen'], 'san': line['san'], 'fens': line['fens'],
            'support_ply': claim['support_ply'], 'board_ply': spec.get('board_ply', claim['support_ply']),
            'chart_change_ply': None}
    chart = _chart(spec, claim, line)
    if chart:
        task['series'] = chart
        task['chart_change_ply'] = next((i for i, v in enumerate(chart['values']) if v != chart['values'][0]), None)
    return task
