# ═══════════════════════════════════════════════════════
# FinSight — LangGraph Checkpointer
# ═══════════════════════════════════════════════════════
#
# Purpose : Build the SQLite checkpointer that makes graph runs durable and
#           replayable.
#
# Public API:
#   async_checkpointer()   async context manager — for the API
#   sync_checkpointer()    context manager — for the CLI and scripts
#
# ══ WHY A CHECKPOINTER AT ALL ══
#   Without one, a graph run is a function call: state exists while it runs and
#   is gone afterwards. With one, every superstep is persisted under a
#   thread_id, which buys three things this project needs:
#
#     1. The audit trail. GET /research/threads/{id} replays the checkpoint
#        history — the router's plan, each branch's partial state, the
#        verification report — rather than trusting a summary written after
#        the fact.
#     2. Durable execution. Phase 7's approval gate pauses a graph with
#        interrupt() and resumes it later; the process may restart in between,
#        and the run survives because the state is on disk, not in memory.
#     3. Resumption after failure. A run that dies mid-fan-out can continue
#        from its last completed superstep instead of re-paying for it.
#
# ══ THE LIFETIME RULE ══
#   from_conn_string() is a CONTEXT MANAGER, not a factory. Leaving its scope
#   closes the connection, so a saver built inside a helper that returns and
#   exits is already dead when the graph tries to use it. It must be held open
#   for as long as any graph might run — which for the API means the whole
#   application lifespan, and is why main.py opens it in lifespan() rather
#   than per request.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

from src.core.config import CHECKPOINT_DB, ensure_data_dirs
from src.persistence.config import SQLITE_PRAGMAS

logger = logging.getLogger(__name__)


def _prepare_checkpoint_db() -> str:
    """
    Ensure the checkpoint database exists with the pragmas we want, and return its path.

    The saver opens its own connection and does not take pragma settings, but
    ``journal_mode`` is persistent database state rather than connection state
    — set once, it stays set. So WAL is applied here, before the saver
    connects, and every later connection inherits it.
    """
    ensure_data_dirs()
    path = str(CHECKPOINT_DB)

    connection = sqlite3.connect(path)
    try:
        connection.execute(f"PRAGMA journal_mode={SQLITE_PRAGMAS['journal_mode']}")
        connection.execute(f"PRAGMA synchronous={SQLITE_PRAGMAS['synchronous']}")
        connection.commit()
    finally:
        connection.close()

    return path


@asynccontextmanager
async def async_checkpointer() -> AsyncIterator[Any]:
    """
    Yield an AsyncSqliteSaver bound to the checkpoint database.

    Hold this open for the lifetime of anything that might run a graph. See the
    lifetime rule in the module docstring.

    Yields
    ------
    AsyncSqliteSaver

    Examples
    --------
    >>> async with async_checkpointer() as saver:      # doctest: +SKIP
    ...     graph = build_research_graph(checkpointer=saver)
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = _prepare_checkpoint_db()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        await saver.setup()
        logger.info("Checkpointer ready (async): %s", path)
        yield saver


@contextmanager
def sync_checkpointer() -> Iterator[Any]:
    """
    Yield a SqliteSaver bound to the checkpoint database.

    The synchronous twin of ``async_checkpointer``, for the CLI and scripts
    that invoke the graph without an event loop. Same database and same
    thread semantics, so a run started from the CLI is visible to the API's
    audit-trail endpoint.

    Yields
    ------
    SqliteSaver
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = _prepare_checkpoint_db()
    with SqliteSaver.from_conn_string(path) as saver:
        saver.setup()
        logger.info("Checkpointer ready (sync): %s", path)
        yield saver
