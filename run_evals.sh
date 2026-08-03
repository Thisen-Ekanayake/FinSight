#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# FinSight — Evaluation Suites
# ═══════════════════════════════════════════════════════
#
# Suite A (research citation faithfulness) is Phase 5.
# Suite B (the dedup TAU_HIGH threshold sweep) is Phase 7.
#
# Usage:
#   ./run_evals.sh research                          # baseline, all 40 examples
#   ./run_evals.sh research --variant strict-src
#   ./run_evals.sh research --limit 5 --no-judges    # free harness smoke run
#   ./run_evals.sh build                             # rebuild the golden dataset
#   ./run_evals.sh check                             # is the committed dataset current?
#   ./run_evals.sh alerts                             # dedup threshold sweep
#   ./run_evals.sh alerts --min-precision 0.95
#
# An eval run is the largest quota spike in this project — the runner prints an
# estimate and waits for confirmation before spending anything. Suite B is the
# exception: no LLM call anywhere in it, only the local bge-small embedder, so
# it costs nothing and needs no confirmation.
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

SUITE="${1:-research}"
shift || true

case "$SUITE" in
    research)
        # filings_rag needs the ingested corpus; without it the narrative
        # archetype scores zero for a reason that has nothing to do with the
        # change under test.
        if ! curl -sf http://localhost:6335/collections >/dev/null 2>&1; then
            echo "Qdrant is not reachable on :6335 — the narrative archetype would score 0." >&2
            echo "Start it with: make qdrant" >&2
            exit 1
        fi
        exec .venv/bin/python -m evals.run_research_eval "$@"
        ;;
    build)
        exec .venv/bin/python -m evals.build_dataset "$@"
        ;;
    check)
        exec .venv/bin/python -m evals.build_dataset --check
        ;;
    alerts)
        exec .venv/bin/python -m evals.run_monitor_eval "$@"
        ;;
    *)
        echo "Usage: ./run_evals.sh {research|build|check|alerts} [options]" >&2
        exit 1
        ;;
esac
