#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV="chess_v3.10"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing $VENV. Create it with Python 3.10, then install requirements." >&2
  exit 1
fi

exec "$VENV/bin/python" -m uvicorn app.main:app --reload
