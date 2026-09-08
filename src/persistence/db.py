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

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateColumn

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


def _sync_additive_schema(engine: Engine) -> None:
    """
    Add columns and indexes the models declare and the database lacks.

    ══ WHY THIS EXISTS ══
      create_all() creates absent TABLES and skips any table that already
      exists — including that table's declared indexes. So the first column
      ever added to a live table was, until this helper, a real migration and
      nothing less. Alembic on a nine-table single-writer SQLite project is
      ceremony; this is the 5% of it that is actually load-bearing.

    ══ ADDITIVE ONLY, AND THAT IS THE WHOLE SAFETY ARGUMENT ══
      It never drops, renames, retypes or reorders. Every operation it
      performs is one that a rollback of the application code survives
      untouched — deploy, revert, and the database still works with the old
      code. Anything else is a hand-written script and a maintenance window,
      not this.

    Idempotent: on a database create_all() just built, every column and index
    is already present, so this is two introspection queries and no DDL.

    Raises
    ------
    RuntimeError
        If a declared column cannot be added by ADD COLUMN at all. Better to
        fail in review than at 3am on the VM.
    """
    inspector = inspect(engine)

    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue  # create_all built it just now, with everything on it

        present = {column["name"] for column in inspector.get_columns(table.name)}

        for column in table.columns:
            if column.name in present:
                continue

            # ADD COLUMN cannot create a key, and SQLite rejects a NOT NULL
            # column outright unless the statement carries a default it can
            # backfill with ("Cannot add a NOT NULL column with default value
            # NULL"). Catching both here turns a bad model change into a test
            # failure rather than a broken deploy.
            if column.primary_key or column.unique:
                raise RuntimeError(
                    f"{table.name}.{column.name} is a key column; ADD COLUMN cannot create it. "
                    "This needs a real migration."
                )
            if not column.nullable and column.server_default is None:
                raise RuntimeError(
                    f"{table.name}.{column.name} is NOT NULL with no server_default; "
                    "ADD COLUMN has nothing to backfill existing rows with. "
                    "Give it a server_default, or make it nullable."
                )

            statement = f"ALTER TABLE {table.name} ADD COLUMN {CreateColumn(column).compile(engine).string}"
            try:
                with engine.begin() as connection:
                    connection.execute(text(statement))
            except OperationalError as exc:
                # The API's lifespan and a monitor CLI on the same host can
                # both call init_db() at once. The loser sees "duplicate
                # column name", which is the state we wanted anyway.
                if "duplicate column" not in str(exc).lower():
                    raise
                logger.debug("%s.%s was added concurrently", table.name, column.name)
            else:
                logger.info("Schema: added %s.%s", table.name, column.name)

        for index in table.indexes:
            # checkfirst issues the dialect's own existence check, which is
            # portable in a way CREATE INDEX IF NOT EXISTS is not.
            index.create(bind=engine, checkfirst=True)


def init_db() -> None:
    """
    Bring the database up to the schema the models declare.

    Idempotent, so it is safe to call on every application start.

    There is still no migration TOOL here on purpose — the schema is small and
    a single developer's, and Alembic would be ceremony. What there is instead
    is create_all() for new tables plus _sync_additive_schema for new columns
    and indexes on existing ones. That covers every schema change this project
    has actually needed. It deliberately does not cover the destructive ones;
    those want a script and a person watching.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    _sync_additive_schema(engine)
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
