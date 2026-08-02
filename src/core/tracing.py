# ═══════════════════════════════════════════════════════
# FinSight — LangSmith Tracing
# ═══════════════════════════════════════════════════════
#
# Purpose : Owns every LangSmith environment variable so no other module
#           touches them, and provides the metadata conventions that make
#           traces filterable rather than merely present.
#
# Public API:
#   configure_tracing(project=None)
#   trace_metadata(**fields) -> dict
#   get_run_url(run_id) -> str | None
#   is_tracing_enabled() -> bool
#
# Convention:
#   Anything flowing through langchain-core or langgraph traces itself.
#   Raw functions (EDGAR fetches, Qdrant searches, the dedup decision) get
#   @traceable from the langsmith package with an explicit run_type:
#
#       @traceable(run_type="retriever", name="qdrant.filings_search")
#       @traceable(run_type="tool",      name="edgar.companyfacts")
#       @traceable(run_type="chain",     name="dedup.decide")
#
#   Always attach metadata (ticker, cycle_id, phase) and tags. Traces you
#   cannot filter are decorative; traces you can filter are a debugger.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import os
from uuid import UUID

from src.core.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure_tracing(*, project: str | None = None, force: bool = False) -> bool:
    """
    Enable LangSmith tracing for this process.

    Call once from an entrypoint, before building any graph. Split projects by
    entrypoint so interactive debugging does not pollute eval history:
    ``finsight-dev`` (CLI/API), ``finsight-eval`` (experiments),
    ``finsight-prod`` (scheduled monitoring cycles).

    Parameters
    ----------
    project : str, optional
        Overrides LANGSMITH_PROJECT for this process.
    force : bool, default False
        Reconfigure even if already configured.

    Returns
    -------
    bool
        True if tracing is active, False if disabled or missing a key.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return is_tracing_enabled()

    if not LANGSMITH_TRACING:
        logger.info("LangSmith tracing disabled (LANGSMITH_TRACING is not true)")
        _CONFIGURED = True
        return False

    if not LANGSMITH_API_KEY:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty — tracing is OFF")
        _CONFIGURED = True
        return False

    # langchain-core reads these at call time, so setting them here is enough.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = project or LANGSMITH_PROJECT

    logger.info("LangSmith tracing ON — project=%s", project or LANGSMITH_PROJECT)
    _CONFIGURED = True
    return True


def is_tracing_enabled() -> bool:
    """True if LangSmith tracing is currently active in this process."""
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def trace_metadata(
    *,
    ticker: str | None = None,
    cycle_id: str | None = None,
    thread_id: str | None = None,
    phase: str | None = None,
    **extra: object,
) -> dict[str, object]:
    """
    Build a metadata dict for a traced run, dropping empty fields.

    These are the keys worth filtering on in the LangSmith UI. Passing them
    consistently is what lets you answer "show me every dedup decision for
    NVDA in cycle X" instead of scrolling.

    Returns
    -------
    dict
        Suitable for ``config={"metadata": ...}`` or ``@traceable(metadata=...)``.
    """
    fields: dict[str, object] = {
        "ticker": ticker,
        "cycle_id": cycle_id,
        "thread_id": thread_id,
        "phase": phase,
        **extra,
    }
    return {k: v for k, v in fields.items() if v is not None}


def get_run_url(run: object) -> str | None:
    """
    Resolve a traced run to a shareable LangSmith URL.

    Parameters
    ----------
    run : object
        A run object, typically ``collect_runs().traced_runs[0]``. A bare run
        id (str or UUID) is also accepted and read back from the API first,
        since ``Client.get_run_url`` itself requires the full run.

    Returns
    -------
    str or None
        The run URL, or None if tracing is off or the lookup fails.
    """
    if not is_tracing_enabled():
        return None

    try:
        from langsmith import Client

        client = Client()
        # get_run_url needs the run object; accept an id for convenience.
        if isinstance(run, (str, UUID)):
            run = client.read_run(run_id=str(run))
        url: str = client.get_run_url(run=run)  # type: ignore[arg-type]
        return url
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not resolve LangSmith run URL: %s", exc)
        return None
