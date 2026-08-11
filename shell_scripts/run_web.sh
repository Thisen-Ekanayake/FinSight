#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — Web Dashboard (dev)
# ═══════════════════════════════════════════════════════
#
# Starts the Vite dev server at http://localhost:5173 with hot reload. It
# proxies /api to the FastAPI backend (API_URL, default http://localhost:8000)
# — start that first with `make api` or `docker compose up -d api`.
#
# For the production bundle instead, use `make web-build` or bring up the
# `web` service in docker-compose.yml.
#
# Usage:
#   ./shell_scripts/run_web.sh
# ───────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../frontend"

if [[ ! -d node_modules ]]; then
    echo "Installing frontend dependencies (first run)…" >&2
    npm install
fi

API_URL="${API_URL:-http://localhost:8000}"
export API_URL

if ! curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    echo "WARNING: cannot reach the API at ${API_URL} — start it first with: make api" >&2
fi

echo "FinSight dashboard -> http://localhost:5173  (API: ${API_URL})"
exec npm run dev -- --host "${VITE_HOST:-localhost}" "$@"
