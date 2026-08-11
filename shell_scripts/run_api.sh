#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — API Server
# ═══════════════════════════════════════════════════════
#
# Starts the FastAPI app. Interactive docs at http://localhost:8000/docs
#
# Usage:
#   ./shell_scripts/run_api.sh              # with autoreload
#   ./shell_scripts/run_api.sh --no-reload  # as it runs in a container
# ───────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv found. Run: make venv && make install" >&2
    exit 1
fi

if [[ ! -f .env ]]; then
    echo "No .env found. Run: cp .env.example .env  (then see docs/api_keys.md)" >&2
    exit 1
fi

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

# The filings specialist needs Qdrant. The API starts without it — every
# numeric question is still answerable — but say so rather than let the
# degradation be discovered through an empty answer.
if ! curl -sf http://localhost:6335/collections >/dev/null 2>&1; then
    echo "WARNING: Qdrant is not reachable on :6335 — filings retrieval will be degraded." >&2
    echo "         Start it with: make qdrant" >&2
fi

RELOAD="--reload"
if [[ "${1:-}" == "--no-reload" ]]; then
    RELOAD=""
    shift
fi

echo "FinSight API -> http://localhost:${PORT}/docs"
exec .venv/bin/python -m uvicorn src.api.main:app --host "$HOST" --port "$PORT" $RELOAD "$@"
