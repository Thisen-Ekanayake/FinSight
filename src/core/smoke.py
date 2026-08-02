# ═══════════════════════════════════════════════════════
# FinSight — Phase 0 Smoke Test
# ═══════════════════════════════════════════════════════
#
# Purpose : Prove the Phase 0 setup gate is closed — config loads, Qdrant is
#           reachable AND isolated, and one traced Gemini call round-trips
#           with a LangSmith run URL to open.
#
# Usage:
#   make smoke
#   python -m src.core.smoke
#   python -m src.core.smoke --skip-llm     # no Gemini quota consumed
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import sys

from src.core.config import (
    EMBEDDING_MODEL,
    ENVIRONMENT,
    GEMINI_MODEL_FLASH,
    GEMINI_RPM,
    LANGSMITH_PROJECT,
    QDRANT_URL,
    ensure_data_dirs,
)
from src.core.errors import FinSightError
from src.core.logging_setup import configure_logging
from src.core.tracing import configure_tracing, is_tracing_enabled

logger = logging.getLogger(__name__)

PROMPT = "In one sentence, what is an SEC Form 8-K used for?"


# ── Individual checks ───────────────────────────────────
def check_config() -> bool:
    """Print resolved configuration. Never fails — this is orientation."""
    print("── Configuration " + "─" * 45)
    print(f"  environment      {ENVIRONMENT}")
    print(f"  gemini (flash)   {GEMINI_MODEL_FLASH}  @ {GEMINI_RPM['flash']} req/min")
    print(f"  embeddings       {EMBEDDING_MODEL}")
    print(f"  qdrant           {QDRANT_URL}")
    print(f"  langsmith proj   {LANGSMITH_PROJECT}")
    print()
    return True


def check_qdrant() -> bool:
    """Connect to Qdrant and run the cross-project isolation assertion."""
    print("── Qdrant " + "─" * 52)
    try:
        from src.vectorstore.client import get_qdrant_client

        client = get_qdrant_client()
        names = sorted(c.name for c in client.get_collections().collections)
        print(f"  reachable at {QDRANT_URL}")
        print("  isolation    OK (no foreign collections)")
        print(f"  collections  {names or '[] (expected — nothing ingested yet)'}")
        print()
        return True
    except FinSightError as exc:
        print(f"  FAILED: {exc}\n")
        return False


def check_llm() -> bool:
    """Make one rate-limited Gemini call and report the LangSmith run URL."""
    print("── Gemini + LangSmith " + "─" * 40)
    try:
        from src.core.llm import get_llm

        llm = get_llm("flash")

        if is_tracing_enabled():
            # collect_runs gives us the run id so we can print a clickable URL.
            from langchain_core.tracers.context import collect_runs

            with collect_runs() as cb:
                response = llm.invoke(PROMPT)
                run_id = str(cb.traced_runs[0].id) if cb.traced_runs else None
        else:
            response = llm.invoke(PROMPT)
            run_id = None

        text = str(response.content).strip()
        print(f"  prompt   {PROMPT}")
        print(f"  response {text[:160]}{'...' if len(text) > 160 else ''}")

        if run_id:
            from src.core.tracing import get_run_url

            url = get_run_url(run_id)
            print(f"  trace    {url or f'run_id={run_id} (URL lookup failed)'}")
        else:
            print("  trace    (tracing off — set LANGSMITH_TRACING=true + LANGSMITH_API_KEY)")
        print()
        return True

    except FinSightError as exc:
        print(f"  FAILED: {exc}\n")
        return False
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}\n")
        return False


# ── Entrypoint ──────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Run the Phase 0 smoke checks. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="FinSight Phase 0 smoke test")
    parser.add_argument("--skip-llm", action="store_true", help="skip the Gemini call (consumes no quota)")
    parser.add_argument("--skip-qdrant", action="store_true", help="skip the Qdrant connection check")
    args = parser.parse_args(argv)

    configure_logging()
    configure_tracing()
    ensure_data_dirs()

    print()
    print("═" * 62)
    print("  FinSight — Phase 0 smoke test")
    print("═" * 62)
    print()

    results: dict[str, bool] = {"config": check_config()}
    if not args.skip_qdrant:
        results["qdrant"] = check_qdrant()
    if not args.skip_llm:
        results["gemini"] = check_llm()

    print("═" * 62)
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        print("═" * 62)
        print()
        return 1

    print("  All checks passed — Phase 0 gate is closed.")
    print("═" * 62)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
