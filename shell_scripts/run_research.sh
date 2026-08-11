#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — Research Query
# ═══════════════════════════════════════════════════════
#
# Ask a financial research question. Routes to specialist agents, fans out in
# parallel, and returns a grounded answer with citations.
#
# Usage:
#   ./shell_scripts/run_research.sh "How did Apple's gross margin trend?"
#   ./shell_scripts/run_research.sh --audit "Compare AAPL and MSFT revenue"
#   ./shell_scripts/run_research.sh --plan-only "What is the Fed funds rate?"
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

if [[ $# -eq 0 ]]; then
    echo "Usage: ./shell_scripts/run_research.sh [--audit] [--plan-only] \"your question\"" >&2
    exit 1
fi

# The filings_rag specialist needs an ingested corpus; warn rather than
# silently returning an answer with no narrative grounding.
if ! curl -sf http://localhost:6335/collections >/dev/null 2>&1; then
    echo "WARNING: Qdrant is not reachable on :6335 — filings retrieval will fail." >&2
    echo "         Start it with: make qdrant" >&2
fi

exec .venv/bin/python -m src.research.cli "$@"
