#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — Filing Ingest
# ═══════════════════════════════════════════════════════
#
# SEC EDGAR filings -> chunks -> embeddings -> Qdrant.
#
# Safe to re-run: point IDs are deterministic (uuid5 of accession+chunk), so
# re-ingesting overwrites rather than duplicating.
#
# Usage:
#   ./shell_scripts/run_ingest.sh                        # the configured watchlist
#   ./shell_scripts/run_ingest.sh --ticker AAPL          # one ticker
#   ./shell_scripts/run_ingest.sh --form 10-K --limit 2  # pass through any ingest flag
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

# Qdrant must be up before we try to write to it.
if ! curl -sf http://localhost:6335/collections >/dev/null 2>&1; then
    echo "Qdrant is not reachable on :6335. Start it with: make qdrant" >&2
    exit 1
fi

# Default to the watchlist when no ticker selection is given.
if [[ $# -eq 0 ]] || [[ ! " $* " =~ " --ticker " && ! " $* " =~ " --watchlist " ]]; then
    set -- --watchlist "$@"
fi

exec .venv/bin/python -m src.vectorstore.ingest "$@"
