# ═══════════════════════════════════════════════════════
# FinSight — FastAPI Application
# ═══════════════════════════════════════════════════════
#
# Purpose : Build the app and own the lifetime of every shared resource.
#
# Public API:
#   create_app()  -> FastAPI
#   app             module-level instance for uvicorn
#
# ══ WHY EVERYTHING IS BUILT IN lifespan() ══
#   The checkpointer is an async context manager whose connection closes when
#   its scope exits, so it cannot be built per request and it cannot be built
#   by a helper that returns. It has to be opened once and held for as long as
#   any graph might run — which is the application's lifetime. The graph is
#   built there too, since it needs the saver, and the Qdrant client because
#   its constructor runs the isolation assertion and startup is when we want
#   to hear about a misconfiguration.
#
# ══ UNAVAILABLE IS NOT THE SAME AS DANGEROUS ══
#   Qdrant being down degrades one specialist; the API still answers every
#   numeric question, so it starts and reports the degradation at /health.
#   Qdrant being the WRONG instance — Athena's, on 6333 — is a configuration
#   error that could write into another project's data, so it aborts startup.
#
# Usage:
#   ./run_api.sh
#   uvicorn src.api.main:app --reload
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import ENVIRONMENT
from src.core.errors import QdrantIsolationError
from src.core.logging_setup import configure_logging
from src.core.tracing import configure_tracing
from src.persistence.checkpointer import async_checkpointer
from src.persistence.db import init_db
from src.research.graph import build_research_graph

logger = logging.getLogger(__name__)

API_TITLE = "FinSight"
API_VERSION = "0.7.0"
API_DESCRIPTION = """
Multi-agent financial research with enforced citations.

A question is decomposed by a router, fanned out across specialist agents in
parallel, merged with source-trust conflict resolution, and then **verified** —
every number in the answer is matched against a value some tool actually
returned before the answer is released.

A second subsystem watches a ticker list on a schedule (or automatically —
see `MONITOR_SCHEDULER_ENABLED`), scores what it finds by rule, and
**deduplicates** it against everything already reported. Every HIGH-severity
finding pauses for a human decision — durably, via a LangGraph `interrupt()`
checkpointed to disk — before it reaches console, file, or email.

* `POST /research/query` — ask a question
* `GET /research/threads/{thread_id}` — replay the run's full audit trail
* `POST /monitor/cycles` — run a monitoring cycle now (may pause for approval)
* `POST /monitor/cycles/{cycle_id}/resume` — decide a paused cycle's HIGH alerts
* `GET /monitor/decisions` — every dedup decision, with the score behind it
* `GET /admin/budgets` — daily API usage
"""


def _connect_qdrant() -> tuple[Any, str]:
    """
    Connect to Qdrant, distinguishing unavailable from misconfigured.

    Returns
    -------
    tuple
        ``(client_or_None, detail)``.

    Raises
    ------
    QdrantIsolationError
        If the configured instance is another project's. Deliberately fatal:
        continuing would risk writing into it.
    """
    from src.vectorstore.client import get_qdrant_client

    try:
        return get_qdrant_client(), "connected"
    except QdrantIsolationError:
        # Never downgrade this to a warning. The whole point of the assertion
        # is that it stops the process before anything is written.
        logger.critical("Qdrant isolation assertion FAILED — refusing to start")
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant unavailable (%s) — filings retrieval will be degraded", exc)
        return None, f"unavailable: {type(exc).__name__}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Start and stop every shared resource exactly once.

    The checkpointer's ``async with`` spans the whole yield: leaving that scope
    closes its connection, so anything narrower would hand the graph a dead
    saver. The scheduler is started INSIDE that scope for the same reason —
    every cycle it runs needs the same live checkpointer connection that
    ``POST /monitor/cycles/{id}/resume`` expects to find a paused cycle in.
    """
    from src.monitor.scheduler import start_scheduler, stop_scheduler
    from src.vectorstore.collections import ensure_collections

    configure_logging()
    configure_tracing()
    init_db()

    client, detail = _connect_qdrant()

    # The collections have to exist before anything reads them, and the monitor
    # is what breaks first: a dedup lookup against a missing collection raises
    # rather than returning no matches, so on a fresh deploy the first cycle
    # dies in alert_synthesizer with a bare Qdrant 404. src/monitor/cli.py calls
    # this for exactly that reason — the API needs it too, because the UI and
    # the scheduler both reach run_cycle through here and never touch the CLI.
    #
    # Idempotent (see ensure_collection), so on every later boot this is two
    # existence checks. Non-fatal by the same logic as _connect_qdrant above:
    # a degraded API that still answers research questions beats one that
    # refuses to start, and /health already reports the Qdrant state.
    if client is not None:
        try:
            for name, created in ensure_collections().items():
                if created:
                    logger.info("Created Qdrant collection %s", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not ensure Qdrant collections (%s) — monitor cycles will fail", exc)

    async with async_checkpointer() as saver:
        app.state.checkpointer = saver
        app.state.qdrant = client
        app.state.qdrant_detail = detail
        app.state.research_graph = build_research_graph(checkpointer=saver)
        app.state.monitor_scheduler = start_scheduler(saver)

        logger.info("FinSight API ready (env=%s, qdrant=%s)", ENVIRONMENT, detail)
        yield

        stop_scheduler(app.state.monitor_scheduler)

    logger.info("FinSight API stopped")


def create_app() -> FastAPI:
    """
    Build the FastAPI application.

    A factory rather than a bare module-level app so tests can construct an
    isolated instance with their own overrides.

    Returns
    -------
    FastAPI
    """
    from src.api.admin_routes import router as admin_router
    from src.api.monitor_routes import router as monitor_router
    from src.api.research_routes import router as research_router
    from src.api.watchlist_routes import router as watchlist_router

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
    )

    # Wide open because Phase 8's Streamlit UI is a separate origin and this
    # API is single-user with no credentials to steal. Tighten before any
    # deployment that is not a portfolio demo.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research_router)
    app.include_router(watchlist_router)
    app.include_router(monitor_router)
    app.include_router(admin_router)
    return app


app = create_app()
