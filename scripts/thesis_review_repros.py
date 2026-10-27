"""Deterministic reproductions for the Chapter 5 review; no provider calls."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
from app.core.commentary.features.envisioned import build_envisioned_line, trim_envisioned_line_by_probe, is_quiescent
from app.core.commentary.phases.composer import _audience_line, render_facts_template, validate_facts_comment
from app.models.comment_facts import CommentFacts

board = chess.Board()
ucis = []
for san in ['e4', 'd6', 'Bb5+', 'Bd7', 'Bxd7+', 'Qxd7', 'Nf3']:
    move = board.parse_san(san)
    ucis.append(move.uci())
    board.push(move)
line = build_envisioned_line(chess.STARTING_FEN, ucis, root_eval_cp=300)
short = _audience_line(line, 'beginner')
probed = trim_envisioned_line_by_probe(line, lambda fen: -300 if fen in line.fens[5:] else 300)
facts = CommentFacts(ply=1, san='e4', uci='e2e4', mover='White', phase='mid', verdict='leads to equality', eval_cp=0, depth=16, display_line=line)
invented = render_facts_template(facts) + ' White has an extra queen.'
result = {
    'audience_shortening': {'full_line': line.line_san, 'shown_line': short.line_san, 'reported_quiet': short.leaf_quiescent, 'actual_quiet': is_quiescent(chess.Board(short.leaf_fen)), 'in_check': chess.Board(short.leaf_fen).is_check()},
    'probe_trim': {'line': probed.line_san, 'reported_quiet': probed.leaf_quiescent, 'actual_quiet': is_quiescent(chess.Board(probed.leaf_fen)), 'in_check': chess.Board(probed.leaf_fen).is_check()},
    'validator': {'unsupported_material_claim_accepted': validate_facts_comment(invented, facts), 'candidate': invented},
}
assert result['audience_shortening']['in_check'] and result['audience_shortening']['reported_quiet']
assert result['probe_trim']['in_check'] and result['probe_trim']['reported_quiet']
assert result['validator']['unsupported_material_claim_accepted']
out = ROOT / 'data/experiments/thesis_review_20260905/reproductions.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
