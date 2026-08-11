#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — Streamlit Dashboard
# ═══════════════════════════════════════════════════════
#
# Starts the dashboard at http://localhost:8501. Talks to the API over HTTP
# (API_URL, default http://localhost:8000) — start that first.
#
# Usage:
#   ./shell_scripts/run_ui.sh
# ───────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv found. Run: make venv && make install" >&2
    exit 1
fi

API_URL="${API_URL:-http://localhost:8000}"

if ! curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    echo "WARNING: cannot reach the API at ${API_URL} — start it first with: make api" >&2
fi

echo "FinSight dashboard -> http://localhost:8501"
exec .venv/bin/python -m streamlit run src/ui/app.py --server.port 8501 "$@"
