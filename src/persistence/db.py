# ═══════════════════════════════════════════════════════
# FinSight — Database Engine & Sessions
# ═══════════════════════════════════════════════════════
#
# Purpose : One engine, one session factory, one place SQLite pragmas are
#           applied.
#
# Public API:
#   get_engine()      lazily-built, process-wide engine
#   init_db()         create tables and the data directory
#   session_scope()   transactional context manager
#   reset_engine()    drop the cached engine (tests)
#
# Usage:
#   with session_scope() as session:
#       session.add(ResearchRun(...))
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import DATABASE_URL, ensure_data_dirs
from src.persistence.config import SQLITE_CONNECT_ARGS, SQLITE_PRAGMAS
from src.persistence.models import Base

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _apply_pragmas(dbapi_connection: Any, _record: Any) -> None:
    """
    Set SQLite pragmas on every new connection.

    Per-connection, not once at startup: pragmas are connection state, and the
    pool opens new connections as concurrency grows. Setting WAL on only the
    first connection would leave later ones on the default journal.
    """
    cursor = dbapi_connection.cursor()
    try:
        for pragma, value in SQLITE_PRAGMAS.items():
            cursor.execute(f"PRAGMA {pragma}={value}")
    finally:
        cursor.close()


def get_engine() -> Engine:
    """
    Return the process-wide engine, building it on first use.

    Lazy rather than module-level so importing this package does not create a
    database file — unit tests import models freely without touching disk.

    Returns
    -------
    Engine
    """
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    is_sqlite = DATABASE_URL.startswith("sqlite")
    if is_sqlite:
        ensure_data_dirs()

    _engine = create_engine(
        DATABASE_URL,
        connect_args=SQLITE_CONNECT_ARGS if is_sqlite else {},
        future=True,
    )

    if is_sqlite:
        event.listen(_engine, "connect", _apply_pragmas)

    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    logger.debug("Database engine ready: %s", DATABASE_URL)
    return _engine


def init_db() -> None:
    """
    Create every table that does not yet exist.

    Idempotent, so it is safe to call on every application start. There is no
    migration tool here on purpose: the schema is small, additive, and a
    single developer's, and Alembic on a five-table SQLite project is ceremony
    without benefit. Revisit if this ever grows a second writer.
    """
    Base.metadata.create_all(get_engine())
    logger.info("Database ready: %s", DATABASE_URL)


@contextmanager
def session_scope() -> Iterator[Session]:
    """
    Provide a transactional session, committing on success and rolling back on error.

    Yields
    ------
    Session

    Raises
    ------
    Exception
        Re-raised after rollback — a caller must not be told a write succeeded
        because the session tidied up quietly.
    """
    get_engine()
    assert _session_factory is not None  # set by get_engine
    session = _session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose of the cached engine. For tests that repoint DATABASE_URL."""
    global _engine, _session_factory

    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
