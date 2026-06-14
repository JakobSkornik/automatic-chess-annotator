"""Rule-based Expert Module (Guid §5.4): feature diffs -> declarative claims."""

from app.core.commentary.rules.engine import (  # noqa: F401
    build_comment_facts,
    run_rules,
    verdict_for_eval,
)
