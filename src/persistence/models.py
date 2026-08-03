# ═══════════════════════════════════════════════════════
# FinSight — Database Models
# ═══════════════════════════════════════════════════════
#
# Purpose : SQLAlchemy 2.0 declarative models for the application database.
#
# Public API:
#   Base, ResearchRun, ApiBudget
#   WatchItem, MonitorCheckpoint, AlertRecord, MonitorCycle, DedupDecision
#
# ══ WHAT IS AND IS NOT STORED HERE ══
#   ResearchRun is a SUMMARY, not the run. The full intermediate state —
#   every superstep, every partial update — lives in the LangGraph
#   checkpointer, keyed by the same thread_id. Duplicating it here would mean
#   two sources of truth for the audit trail and a schema that has to track
#   LangGraph's.
#
# ══ TWO STORES, TWO JOBS (Phase 6) ══
#   An alert exists in BOTH SQLite and Qdrant, and that is deliberate:
#
#     Qdrant   the dedup INDEX. One point per live alert, carrying the vector
#              that incoming candidates are compared against. Pruned on a
#              retention window, so it is a working set, not a history.
#     SQLite   the durable RECORD. Every alert ever fired, every suppression
#              decision, every cycle. Never pruned.
#
#   Putting the history in Qdrant would make the vector search slower every
#   week for no retrieval benefit. Putting the index in SQLite would mean no
#   vector search at all.
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


# ═══════════════════════════════════════════════════════
# Monitoring subsystem (Phase 6)
# ═══════════════════════════════════════════════════════


class WatchItem(Base):
    """
    One ticker the monitoring subsystem watches.

    Soft-deleted via ``active`` rather than removed, so a ticker that is
    dropped and re-added later keeps its ``added_at`` — and, more importantly,
    keeps its MonitorCheckpoint rows. Re-adding a ticker with a wiped
    checkpoint would make the next cycle re-report every filing since the
    beginning of time as new.
    """

    __tablename__ = "watch_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    company_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")

    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    # A newly added ticker must warm up on its own before it can fire, or
    # adding NVDA in month three spams every filing it has on file. Per-ticker,
    # not per-system: the rest of the watchlist is already warm.
    warmed_up: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<WatchItem {self.ticker}{'' if self.active else ' (inactive)'}>"


class MonitorCheckpoint(Base):
    """
    The last time one monitor successfully checked one ticker.

    THIS IS WHAT MAKES A MONITOR A MONITOR RATHER THAN A POLLER. ``filing``
    asks EDGAR only for filings since this timestamp, so a cycle costs one
    small request per ticker regardless of how much history exists.

    Only advanced on SUCCESS. A failed fetch leaves the timestamp where it was,
    so the next cycle re-covers the gap instead of silently skipping whatever
    was published during the outage.
    """

    __tablename__ = "monitor_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    monitor: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    last_checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (UniqueConstraint("ticker", "monitor", name="uq_checkpoint_ticker_monitor"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MonitorCheckpoint {self.ticker}/{self.monitor} @ {self.last_checked_at:%Y-%m-%d %H:%M}>"


class AlertRecord(Base):
    """
    One alert that was fired — the durable half of the two-store split.

    ``occurrence_count`` is incremented in place when a later candidate is
    suppressed against this row, so an alert reads as "this happened, 4 times,
    most recently at X" rather than four near-identical rows.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    cycle_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="FIRED", index=True)

    headline: Mapped[str] = mapped_column(Text, nullable=False, default="")
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # The numeric-stripped text that was actually embedded. Kept because a
    # dedup decision cannot be understood — or re-labelled in Phase 7 — without
    # seeing what the vector was built from.
    canonical_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # JSON array of {source_type, source_id, url, as_of}. A blob rather than a
    # table: it is written once, read whole, and never queried into.
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    fired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    # Set when this alert escalated an existing one — a MERGE that arrived with
    # higher severity than its parent.
    parent_alert_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (Index("ix_alerts_ticker_type_fired", "ticker", "alert_type", "fired_at"),)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AlertRecord {self.ticker} {self.alert_type} {self.severity} x{self.occurrence_count}>"


class DedupDecision(Base):
    """
    Every dedup decision, with the score that produced it.

    ══ THIS TABLE IS PHASE 7's TRAINING DATA ══
    The thresholds in vectorstore/config.py were calibrated on hand-written
    text. Recalibrating them on REAL candidate pairs needs those pairs to have
    been recorded with their scores at the moment the decision was made — which
    cannot be reconstructed afterwards, because the parent's vector drifts as
    the centroid updates.

    So every decision is logged, including FIRE (score 0.0, no parent). A table
    of only the suppressions would be a biased sample of exactly the wrong kind.
    """

    __tablename__ = "dedup_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    alert_type: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(8), nullable=False, default="LOW")

    # FIRE | SUPPRESS_EXACT | SUPPRESS_SEMANTIC | MERGE | ESCALATE | WARMUP
    decision: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")

    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    parent_alert_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # 0.0 when the decision never reached a vector search — an exact-key hit or
    # a cold collection. Nullable would be more honest but makes every sweep
    # query carry a NULL guard; `reason` says which path was taken.
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DedupDecision {self.ticker} {self.decision} @ {self.score:.3f}>"


class MonitorCycle(Base):
    """
    One run of the monitoring graph, summarised.

    ``cycle_id`` doubles as the checkpointer thread id (``monitor:{cycle_id}``),
    so a row here is the entry point into the full replayable state — the same
    relationship ResearchRun has with its thread.
    """

    __tablename__ = "monitor_cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    warmup: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tickers: Mapped[str] = mapped_column(String(256), nullable=False, default="")

    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fired_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    suppressed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    merged_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    api_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MonitorCycle {self.cycle_id} fired={self.fired_count}>"
