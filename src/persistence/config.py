# ═══════════════════════════════════════════════════════
# FinSight — Persistence Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : SQLite tuning and retention policy as module constants.
#
# Public API:
#   SQLITE_PRAGMAS, SQLITE_CONNECT_ARGS
#   RESEARCH_RUN_RETENTION_DAYS, BUDGET_RETENTION_DAYS
# ═══════════════════════════════════════════════════════

from __future__ import annotations

# Applied to every new SQLite connection.
#
# journal_mode=WAL is the load-bearing one. The API writes a research run on
# the same database the scheduler reads budgets from, and the default rollback
# journal takes a database-wide write lock that blocks readers. WAL lets
# readers proceed during a write, which is the difference between a monitoring
# cycle being slow and it timing out.
#
# synchronous=NORMAL is the standard WAL companion: durable against process
# crash, and only at risk from an OS-level crash mid-transaction. For research
# runs and call counters that trade is obviously right.
SQLITE_PRAGMAS: dict[str, str] = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "foreign_keys": "ON",
    # Wait rather than raising "database is locked" the instant a writer holds
    # the lock. Milliseconds.
    "busy_timeout": "5000",
}

# check_same_thread=False because FastAPI runs sync endpoints in a threadpool,
# so a connection is legitimately used from a thread other than the one that
# created it. Safe here: SQLAlchemy's pool hands one connection to one thread
# at a time.
SQLITE_CONNECT_ARGS: dict[str, object] = {"check_same_thread": False}

# ── Retention ───────────────────────────────────────────
# Research runs are an audit trail, not a cache; they stay long enough to be
# useful and not so long that a demo database becomes a data-retention
# question.
RESEARCH_RUN_RETENTION_DAYS: int = 90

# Budget rows are one per provider per day and tiny, but there is no reason to
# keep a year of them.
BUDGET_RETENTION_DAYS: int = 30
