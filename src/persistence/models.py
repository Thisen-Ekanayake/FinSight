# ═══════════════════════════════════════════════════════
# FinSight — Database Models
# ═══════════════════════════════════════════════════════
#
# Purpose : SQLAlchemy 2.0 declarative models for the application database.
#
# Public API:
#   Base, ResearchRun, ApiBudget
#
# Scope note:
#   Only what Phase 4 needs. The watchlist, alert, and cycle tables arrive with
#   the monitoring subsystem in Phase 6 — declaring them early would mean
#   guessing their shape before the code that uses them exists.
#
# ══ WHAT IS AND IS NOT STORED HERE ══
#   ResearchRun is a SUMMARY, not the run. The full intermediate state —
#   every superstep, every partial update — lives in the LangGraph
#   checkpointer, keyed by the same thread_id. Duplicating it here would mean
#   two sources of truth for the audit trail and a schema that has to track
#   LangGraph's.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware UTC now. SQLite has no native timestamp type, so this is explicit."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every FinSight table."""


class ResearchRun(Base):
    """
    One completed research query, summarised for listing and metrics.

    ``thread_id`` is the join key into the checkpointer: given a row here, the
    full state history is one ``get_state_history`` call away. That is what the
    audit-trail endpoint returns.
    """

    __tablename__ = "research_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ── Verification outcome — the metric Phase 5 tracks across experiments ──
    citation_coverage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    verification_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    unsupported_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repair_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── Shape of the run ──
    agents_used: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    tickers: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    finding_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    __table_args__ = (Index("ix_research_runs_created_coverage", "created_at", "citation_coverage"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ResearchRun {self.thread_id} coverage={self.citation_coverage:.2f}>"


class ApiBudget(Base):
    """
    Calls made to one provider on one UTC day.

    Free tiers are metered per day, so the counter is per day and the reset is
    implicit: a new day is a new row rather than a scheduled job that has to
    actually run. A cron that fails to fire cannot leave a stale count behind
    if there is no reset to miss.
    """

    __tablename__ = "api_budgets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # ISO date string rather than a DATE column: SQLite stores dates as text
    # anyway, and a string compares and groups without a driver-specific cast.
    day: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (UniqueConstraint("provider", "day", name="uq_api_budget_provider_day"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ApiBudget {self.provider} {self.day}={self.call_count}>"
