#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — Autonomous Monitoring
# ═══════════════════════════════════════════════════════
#
# Watch the ticker list, deduplicate what it finds, and report what is new.
#
# Usage:
#   ./run_monitor.sh --once --warmup      # FIRST RUN: index, report nothing
#   ./run_monitor.sh --once               # a real cycle
#   ./run_monitor.sh --watchlist          # show it
#   ./run_monitor.sh --add TSLA
#   ./run_monitor.sh --decisions          # why things were suppressed
#
# Run --warmup once before the first real cycle. A cold dedup index has nothing
# to match against, so cycle 1 would otherwise report every open filing, every
# recent article, and every price move in one burst.
# ───────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
    echo "No .venv found. Run: make venv && make install" >&2
    exit 1
fi

if [[ ! -f .env ]]; then
    echo "No .env found. Run: cp .env.example .env  (then see docs/api_keys.md)" >&2
    exit 1
fi

# Unlike research, the dedup engine CANNOT degrade without Qdrant: with no
# index every candidate reads as new and every duplicate fires. Refuse rather
# than produce a cycle that looks like it worked.
if ! curl -sf http://localhost:6335/collections >/dev/null 2>&1; then
    echo "Qdrant is not reachable on :6335 — the dedup index is unavailable." >&2
    echo "Without it every candidate would read as new and every duplicate would fire." >&2
    echo "Start it with: make qdrant" >&2
    exit 1
fi

exec .venv/bin/python -m src.monitor.cli "$@"
