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
API_VERSION = "0.4.0"
API_DESCRIPTION = """
Multi-agent financial research with enforced citations.

A question is decomposed by a router, fanned out across specialist agents in
parallel, merged with source-trust conflict resolution, and then **verified** —
every number in the answer is matched against a value some tool actually
returned before the answer is released.

* `POST /research/query` — ask a question
* `GET /research/threads/{thread_id}` — replay the run's full audit trail
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
    saver.
    """
    configure_logging()
    configure_tracing()
    init_db()

    client, detail = _connect_qdrant()

    async with async_checkpointer() as saver:
        app.state.checkpointer = saver
        app.state.qdrant = client
        app.state.qdrant_detail = detail
        app.state.research_graph = build_research_graph(checkpointer=saver)

        logger.info("FinSight API ready (env=%s, qdrant=%s)", ENVIRONMENT, detail)
        yield

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
    from src.api.research_routes import router as research_router

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
    app.include_router(admin_router)
    return app


app = create_app()
