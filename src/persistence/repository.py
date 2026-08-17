# ═══════════════════════════════════════════════════════
# FinSight — Repository
# ═══════════════════════════════════════════════════════
#
# Purpose : Every query the application makes, in one place, returning plain
#           TypedDicts rather than ORM objects.
#
# Public API:
#   record_research_run(state, thread_id, subject, latency_ms) -> str
#   list_research_runs(subject, unlimited, limit) -> list[ResearchRunSummary]
#   get_research_run(thread_id, subject, unlimited) -> ResearchRunSummary | None
#   can_use_thread(thread_id, subject, unlimited)   -> bool
#   record_api_call(provider, count)        -> int
#   get_budget_status()                     -> list[BudgetStatus]
#
#   ── free tier ──
#   get_free_quota(subject, limit)          -> FreeQuotaStatus
#   consume_free_query(subject, limit)      -> FreeQuotaStatus | None
#   refund_free_query(subject)              -> None
#
#   ── monitoring (Phase 6) ──
#   list_watchlist / add_watch_item / remove_watch_item / mark_warmed_up
#   get_checkpoints / set_checkpoint
#   record_alert / bump_alert_occurrence / list_alerts / get_alert
#   record_dedup_decisions / list_dedup_decisions
#   record_cycle / list_cycles / get_cycle
#
# ══ WHY DETACHED DICTS, NOT ORM OBJECTS ══
#   A session_scope() commits and closes when it exits. Handing an ORM object
#   back across that boundary gives the caller something whose unloaded
#   attributes raise DetachedInstanceError the moment a route tries to
#   serialise it. Reading everything inside the scope and returning a dict
#   makes that impossible rather than merely unlikely.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, TypedDict

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from src.data.config import BUDGET_SOFT_LIMIT, DAILY_BUDGETS
from src.persistence.db import session_scope
from src.persistence.models import (
    AlertRecord,
    ApiBudget,
    DedupDecision,
    FreeQueryQuota,
    MonitorCheckpoint,
    MonitorCycle,
    ResearchRun,
    WatchItem,
    utcnow,
)

logger = logging.getLogger(__name__)


class ResearchRunSummary(TypedDict):
    """One row of the research-run history."""

    thread_id: str
    query: str
    final_answer: str
    citation_coverage: float
    verification_passed: bool
    unsupported_count: int
    repair_count: int
    agents_used: list[str]
    tickers: list[str]
    finding_count: int
    tool_call_count: int
    error_count: int
    latency_ms: int
    created_at: str


class BudgetStatus(TypedDict):
    """One provider's usage against its daily allowance."""

    provider: str
    day: str
    used: int
    limit: int
    remaining: int
    soft_limit_reached: bool
    exhausted: bool


def _today() -> str:
    """The current UTC date as an ISO string — the budget bucket key."""
    return datetime.now(timezone.utc).date().isoformat()


def _iso_utc(value: datetime) -> str:
    """
    Serialise a stored timestamp as an explicitly UTC ISO string.

    ══ SQLITE HAS NO TIMESTAMP TYPE ══
    ``DateTime(timezone=True)`` round-trips to a NAIVE datetime on SQLite
    however it went in — the offset is simply not stored. Handing that
    straight back out is fine for display, but the monitor checkpoints are
    read, parsed, and compared against ``datetime.now(timezone.utc)``, and
    comparing naive to aware raises TypeError.

    Worse, it would raise inside a Send() branch, where the specialist wrapper
    turns it into one monitor quietly contributing nothing — a silent gap in
    coverage rather than a crash. So the offset is reattached on the way out,
    once, here.
    """
    return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()


def _to_summary(run: ResearchRun) -> ResearchRunSummary:
    """Flatten an ORM row into a detached dict, inside the session."""
    return ResearchRunSummary(
        thread_id=run.thread_id,
        query=run.query,
        final_answer=run.final_answer,
        citation_coverage=run.citation_coverage,
        verification_passed=run.verification_passed,
        unsupported_count=run.unsupported_count,
        repair_count=run.repair_count,
        agents_used=[a for a in run.agents_used.split(",") if a],
        tickers=[t for t in run.tickers.split(",") if t],
        finding_count=run.finding_count,
        tool_call_count=run.tool_call_count,
        error_count=run.error_count,
        latency_ms=run.latency_ms,
        created_at=_iso_utc(run.created_at),
    )


# ── Research runs ───────────────────────────────────────
# The owner value on every row written before there was an owner column, and
# on everything the CLI records — it has no identity to record. NOT NULL '',
# never NULL, so every predicate below is a plain equality with no IS NULL
# branch for anyone to forget. See ResearchRun.subject.
LEGACY_SUBJECT = ""


def _visible_owners(subject: str, *, unlimited: bool) -> list[str]:
    """
    The owner values this caller may see. THE visibility rule, in one place.

    Everyone sees their own runs. The unlimited tier additionally sees the
    unattributable ones — rows written before this column existed, and CLI
    runs — because those belong to nobody and the operator is the only account
    that can reasonably be handed them. Nothing is reassigned or deleted; the
    rows simply become visible to the operator and to no one else.

    Note what this is NOT: the unlimited tier does not see another *account's*
    runs. It is own + unattributable, not everything.

    ══ WHY A LIST RATHER THAN A PREDICATE ══
      This is consumed in two evaluation contexts — as SQL (``subject.in_()``
      in the listing) and in Python (``can_use_thread``, which must tell "no
      row" apart from "not yours" and so cannot be a WHERE clause). A list is
      the one shape both use, which keeps the rule from being written twice
      and drifting apart.
    """
    if not subject:
        # require_identity cannot produce this — it falls back to
        # f"email:{email}" when a token somehow carries no `sub`. If it ever
        # did, an empty subject would match every legacy row, which is the
        # exact failure this whole module exists to prevent.
        raise ValueError("a run owner must have a non-empty subject")
    return [subject, LEGACY_SUBJECT] if unlimited else [subject]


def record_research_run(
    state: dict[str, Any], *, thread_id: str, subject: str, email: str = "", latency_ms: int
) -> str:
    """
    Summarise one finished research run.

    Upserts on ``thread_id`` so a resumed or re-run thread updates its row
    rather than accumulating duplicates.

    Parameters
    ----------
    state : dict
        The final ResearchState.
    thread_id : str
        Checkpoint thread id — the join key into the checkpointer.
    subject : str
        Who ran it. Keyword-only and with no default on purpose: every call
        site must say whose run this is, so adding one cannot silently produce
        an unattributed row.
    email : str, optional
        Denormalised label for whoever reads the table directly.
    latency_ms : int
        Wall-clock duration of the run.

    Returns
    -------
    str
        The thread id, for convenience at the call site.
    """
    report = state.get("verification") or {}
    plan = state.get("plan") or {}

    with session_scope() as session:
        run = session.scalar(select(ResearchRun).where(ResearchRun.thread_id == thread_id))
        if run is None:
            run = ResearchRun(thread_id=thread_id)
            session.add(run)
        elif run.subject not in (subject, LEGACY_SUBJECT):
            # Unreachable through the API — _resolve_thread refuses a foreign
            # thread id before the graph ever runs. Reached only by a race or
            # a non-HTTP caller, and a query that already cost real money must
            # not turn into a 500, so this drops the summary and shouts.
            logger.error(
                "Refusing to overwrite run %s (owned by another account) on behalf of %s",
                thread_id,
                subject,
            )
            return thread_id

        # LEGACY_SUBJECT is in the tuple above so that re-asking something from
        # before this column existed CLAIMS the old row rather than orphaning it.
        run.subject = subject
        run.owner_email = email or run.owner_email
        run.query = state.get("query", "")
        run.final_answer = state.get("final_answer") or state.get("draft_answer") or ""
        run.citation_coverage = float(report.get("citation_coverage", 0.0))
        run.verification_passed = bool(report.get("passed", False))
        run.unsupported_count = len(report.get("unsupported_claims", []))
        run.repair_count = int(state.get("repair_count", 0))
        run.agents_used = ",".join(plan.get("selected_agents", []))
        run.tickers = ",".join(plan.get("tickers", []))
        run.finding_count = len(state.get("findings", []))
        run.tool_call_count = len(state.get("tool_calls", []))
        run.error_count = len(state.get("errors", []))
        run.latency_ms = latency_ms

    logger.info("Recorded run %s (coverage %.2f)", thread_id, run.citation_coverage)
    return thread_id


def list_research_runs(*, subject: str, unlimited: bool, limit: int = 50) -> list[ResearchRunSummary]:
    """Return this caller's recent runs, newest first."""
    with session_scope() as session:
        rows = session.scalars(
            select(ResearchRun)
            .where(ResearchRun.subject.in_(_visible_owners(subject, unlimited=unlimited)))
            .order_by(ResearchRun.created_at.desc())
            .limit(limit)
        ).all()
        return [_to_summary(row) for row in rows]


def get_research_run(thread_id: str, *, subject: str, unlimited: bool) -> ResearchRunSummary | None:
    """
    Return one of this caller's runs by thread id, or None.

    ``None`` means "no such run, as far as you are concerned". It deliberately
    does not distinguish a run that does not exist from one that belongs to
    somebody else, and callers must not try to — the difference is exactly
    what an enumeration attack wants.
    """
    with session_scope() as session:
        run = session.scalar(
            select(ResearchRun).where(
                ResearchRun.thread_id == thread_id,
                ResearchRun.subject.in_(_visible_owners(subject, unlimited=unlimited)),
            )
        )
        return _to_summary(run) if run else None


def can_use_thread(thread_id: str, *, subject: str, unlimited: bool) -> bool:
    """
    Whether this caller may read or append to a checkpoint thread.

    ══ WHY THIS IS NOT `get_research_run(...) is not None` ══
      The transcript lives in the CHECKPOINTER, whose tables are LangGraph's
      and carry no identity at all — thread_id and nothing else. A row in
      research_runs is therefore the only thing in the system that can say who
      a thread belongs to.

      Which makes the missing-row case the interesting one. A thread with no
      row is not "denied", it is UNATTRIBUTABLE: a run that crashed before its
      summary was written, a CLI run, a monitor cycle's thread. Those get
      exactly the same treatment as a legacy row — the unlimited tier and
      nobody else. That keeps it fail-closed, since a free account can never
      reach a thread this table cannot vouch for.
    """
    with session_scope() as session:
        owner = session.scalar(select(ResearchRun.subject).where(ResearchRun.thread_id == thread_id))

    return (LEGACY_SUBJECT if owner is None else owner) in _visible_owners(subject, unlimited=unlimited)


# ── API budgets ─────────────────────────────────────────
def record_api_call(provider: str, *, count: int = 1) -> int:
    """
    Increment today's call counter for a provider.

    Parameters
    ----------
    provider : str
        Provider key as used in ``DAILY_BUDGETS``.
    count : int, default 1
        Calls to add.

    Returns
    -------
    int
        The provider's new total for today.
    """
    day = _today()

    with session_scope() as session:
        row = session.scalar(select(ApiBudget).where(ApiBudget.provider == provider, ApiBudget.day == day))
        if row is None:
            row = ApiBudget(provider=provider, day=day, call_count=0)
            session.add(row)

        row.call_count += count
        row.updated_at = utcnow()
        total = row.call_count

    limit = DAILY_BUDGETS.get(provider, 0)
    if limit and total >= limit * BUDGET_SOFT_LIMIT:
        logger.warning("Budget: %s at %d/%d calls today", provider, total, limit)

    return total


# ── Free tier ───────────────────────────────────────────
class FreeQuotaStatus(TypedDict):
    """One account's lifetime free-query standing."""

    subject: str
    email: str
    used: int
    limit: int
    remaining: int
    exhausted: bool


def _quota_status(subject: str, email: str, used: int, limit: int) -> FreeQuotaStatus:
    """Derive the reportable view. ``remaining`` floors at zero — a lowered limit is not a debt."""
    return FreeQuotaStatus(
        subject=subject,
        email=email,
        used=used,
        limit=limit,
        remaining=max(0, limit - used),
        exhausted=used >= limit,
    )


def get_free_quota(subject: str, *, limit: int) -> FreeQuotaStatus:
    """
    Report an account's standing without spending anything.

    An account that has never queried has no row, and that is reported as
    ``used=0`` rather than as an error — the row is created by the first
    successful ``consume_free_query``, not by signing in.
    """
    with session_scope() as session:
        row = session.scalar(select(FreeQueryQuota).where(FreeQueryQuota.subject == subject))
        if row is None:
            return _quota_status(subject, "", 0, limit)
        return _quota_status(subject, row.email, row.used, limit)


def consume_free_query(subject: str, *, limit: int, email: str = "") -> FreeQuotaStatus | None:
    """
    Atomically spend one lifetime free query.

    ══ WHY A CONDITIONAL UPDATE AND NOT read-modify-write ══
      record_api_call above selects, mutates and commits, and is racy — two
      concurrent calls can lose an increment. There it is harmless: a soft
      budget warning that fires one call late costs nothing. Here a lost
      increment IS a free query, and the increment is the entire feature. So
      the spend is a single statement whose WHERE clause carries the limit:

          UPDATE ... SET used = used + 1 WHERE subject = ? AND used < ?

      The database decides whether there was an allowance left, and rowcount
      reports what it decided. Two requests racing on the last unit produce
      exactly one rowcount of 1.

    The row is only ever INSERTed with ``used=0``, never with 1, so the insert
    path cannot itself spend a unit and the retry below is safe to repeat.

    Parameters
    ----------
    subject : str
        Google's ``sub`` claim. Never the email.
    limit : int
        The lifetime allowance. ``0`` refuses everything.
    email : str, optional
        Stored as a label for the operator, and refreshed on every spend.

    Returns
    -------
    FreeQuotaStatus | None
        The new standing, or ``None`` when the allowance is already spent.
    """
    if limit <= 0:
        return None

    # Two passes at most: the first can miss because the row does not exist
    # yet, and the second runs after it has been created.
    for _ in range(2):
        with session_scope() as session:
            spent = session.execute(
                update(FreeQueryQuota)
                .where(FreeQueryQuota.subject == subject, FreeQueryQuota.used < limit)
                .values(
                    used=FreeQueryQuota.used + 1,
                    # Keep the label current without ever letting it become the
                    # key: an account that changed address stays one row.
                    email=email or FreeQueryQuota.email,
                    updated_at=utcnow(),
                )
                .execution_options(synchronize_session=False)
            ).rowcount

            if spent == 1:
                row = session.scalar(select(FreeQueryQuota).where(FreeQueryQuota.subject == subject))
                # Built inside the scope: the row is detached once it exits.
                return _quota_status(subject, row.email if row else email, row.used if row else limit, limit)

            exists = session.scalar(select(FreeQueryQuota.id).where(FreeQueryQuota.subject == subject)) is not None

        if exists:
            return None  # The row is there and at its limit. Genuinely exhausted.

        try:
            with session_scope() as session:
                session.add(FreeQueryQuota(subject=subject, email=email, used=0))
        except IntegrityError:
            # A concurrent first request won the insert. The unique index on
            # `subject` is what turns that race into this exception rather than
            # into two rows and two allowances.
            logger.debug("Quota row for %s was created concurrently; retrying the spend", subject)

    return None


def refund_free_query(subject: str) -> None:
    """
    Return one unit to an account whose query never produced an answer.

    The ``used > 0`` predicate is a floor, not an optimisation: a double
    refund must not mint credit that was never spent.
    """
    with session_scope() as session:
        session.execute(
            update(FreeQueryQuota)
            .where(FreeQueryQuota.subject == subject, FreeQueryQuota.used > 0)
            .values(used=FreeQueryQuota.used - 1, updated_at=utcnow())
            .execution_options(synchronize_session=False)
        )


def get_budget_status() -> list[BudgetStatus]:
    """
    Report today's usage for every provider that has a daily allowance.

    Providers with no calls yet are still listed at zero — an empty response
    would read as "budgets are not being tracked", which is a different and
    much worse thing than "nothing has been spent".

    Returns
    -------
    list of BudgetStatus
    """
    day = _today()

    with session_scope() as session:
        rows = session.scalars(select(ApiBudget).where(ApiBudget.day == day)).all()
        used_by_provider = {row.provider: row.call_count for row in rows}

    statuses: list[BudgetStatus] = []
    for provider, limit in sorted(DAILY_BUDGETS.items()):
        used = used_by_provider.get(provider, 0)
        statuses.append(
            BudgetStatus(
                provider=provider,
                day=day,
                used=used,
                limit=limit,
                remaining=max(limit - used, 0),
                soft_limit_reached=bool(limit) and used >= limit * BUDGET_SOFT_LIMIT,
                # A limit of 0 means the provider is disabled entirely, which
                # is exhausted by definition — Alpha Vantage's 25/day is
                # unusable, so DAILY_BUDGETS sets it to 0 rather than omitting
                # it, and this must not read as "unlimited".
                exhausted=used >= limit,
            )
        )

    return statuses


# ═══════════════════════════════════════════════════════
# Monitoring subsystem (Phase 6)
# ═══════════════════════════════════════════════════════


class WatchItemRow(TypedDict):
    """One watched ticker."""

    ticker: str
    company_name: str
    active: bool
    warmed_up: bool
    added_at: str


class AlertRow(TypedDict):
    """One fired alert, flattened out of the session."""

    alert_id: str
    cycle_id: str
    ticker: str
    alert_type: str
    severity: str
    status: str
    headline: str
    detail: str
    canonical_text: str
    dedup_key: str
    evidence: list[dict[str, Any]]
    metrics: dict[str, Any]
    occurrence_count: int
    first_seen_at: str
    last_seen_at: str
    fired_at: str
    parent_alert_id: str | None


class DedupDecisionRow(TypedDict):
    """One recorded dedup decision, with the score behind it."""

    cycle_id: str
    ticker: str
    alert_type: str
    severity: str
    decision: str
    reason: str
    dedup_key: str
    candidate_text: str
    parent_alert_id: str | None
    parent_text: str
    score: float
    decided_at: str


class CycleRow(TypedDict):
    """One monitoring cycle, summarised."""

    cycle_id: str
    status: str
    warmup: bool
    tickers: list[str]
    candidate_count: int
    fired_count: int
    suppressed_count: int
    merged_count: int
    error_count: int
    api_call_count: int
    started_at: str
    duration_ms: int


# ── Watchlist ───────────────────────────────────────────
def _to_watch_row(item: WatchItem) -> WatchItemRow:
    return WatchItemRow(
        ticker=item.ticker,
        company_name=item.company_name,
        active=item.active,
        warmed_up=item.warmed_up,
        added_at=_iso_utc(item.added_at),
    )


def list_watchlist(*, active_only: bool = True) -> list[WatchItemRow]:
    """
    Return the watched tickers, alphabetically.

    Parameters
    ----------
    active_only : bool, default True
        Exclude soft-deleted entries.

    Returns
    -------
    list of WatchItemRow
    """
    with session_scope() as session:
        statement = select(WatchItem).order_by(WatchItem.ticker)
        if active_only:
            statement = statement.where(WatchItem.active.is_(True))
        return [_to_watch_row(row) for row in session.scalars(statement).all()]


def add_watch_item(ticker: str, *, company_name: str = "") -> WatchItemRow:
    """
    Add a ticker to the watchlist, or reactivate one that was removed.

    Reactivation deliberately does NOT reset ``warmed_up``. A ticker that has
    already been through a warmup cycle has its events in the dedup index; re-
    warming it would re-upsert the same points and delay real alerts for
    nothing.

    Parameters
    ----------
    ticker : str
        US-listed symbol. Upper-cased.
    company_name : str, optional
        Display name, used in the canonical alert text.

    Returns
    -------
    WatchItemRow
    """
    symbol = ticker.strip().upper()

    with session_scope() as session:
        item = session.scalar(select(WatchItem).where(WatchItem.ticker == symbol))
        if item is None:
            item = WatchItem(ticker=symbol, company_name=company_name)
            session.add(item)
        else:
            item.active = True
            if company_name:
                item.company_name = company_name
        session.flush()
        row = _to_watch_row(item)

    logger.info("Watchlist: %s active (warmed_up=%s)", symbol, row["warmed_up"])
    return row


def remove_watch_item(ticker: str) -> bool:
    """
    Soft-delete a ticker. Returns True if it was present and active.

    Soft rather than hard so the MonitorCheckpoint rows survive — see the
    WatchItem docstring for why re-adding with a wiped checkpoint is a
    problem.
    """
    symbol = ticker.strip().upper()

    with session_scope() as session:
        item = session.scalar(select(WatchItem).where(WatchItem.ticker == symbol))
        if item is None or not item.active:
            return False
        item.active = False

    logger.info("Watchlist: %s deactivated", symbol)
    return True


def mark_warmed_up(tickers: Iterable[str]) -> int:
    """Flag tickers as having completed a warmup cycle. Returns rows touched."""
    symbols = [t.strip().upper() for t in tickers if t.strip()]
    if not symbols:
        return 0

    with session_scope() as session:
        rows = session.scalars(select(WatchItem).where(WatchItem.ticker.in_(symbols))).all()
        for row in rows:
            row.warmed_up = True
        return len(rows)


# ── Monitor checkpoints ─────────────────────────────────
def get_checkpoints(tickers: Iterable[str] | None = None) -> dict[str, str]:
    """
    Return ``{"TICKER:monitor": iso_timestamp}`` for the last successful check.

    A flat string key rather than nested dicts because this travels through
    graph state, and LangGraph state is serialised to the checkpointer —
    tuple keys would not survive the round trip.

    Parameters
    ----------
    tickers : iterable of str, optional
        Restrict to these symbols. All if None.

    Returns
    -------
    dict
        Empty for a ticker/monitor pair that has never run, which the monitors
        read as "fall back to the default lookback".
    """
    with session_scope() as session:
        statement = select(MonitorCheckpoint)
        if tickers is not None:
            symbols = [t.strip().upper() for t in tickers if t.strip()]
            if not symbols:
                return {}
            statement = statement.where(MonitorCheckpoint.ticker.in_(symbols))
        rows = session.scalars(statement).all()
        return {f"{row.ticker}:{row.monitor}": _iso_utc(row.last_checked_at) for row in rows}


def set_checkpoint(ticker: str, monitor: str, *, when: datetime | None = None) -> None:
    """
    Advance one monitor's watermark for one ticker.

    Call this ONLY after a successful fetch. Advancing it on failure would skip
    whatever was published during the outage, silently and permanently.
    """
    symbol = ticker.strip().upper()
    stamp = when or utcnow()

    with session_scope() as session:
        row = session.scalar(
            select(MonitorCheckpoint).where(
                MonitorCheckpoint.ticker == symbol,
                MonitorCheckpoint.monitor == monitor,
            )
        )
        if row is None:
            row = MonitorCheckpoint(ticker=symbol, monitor=monitor)
            session.add(row)
        row.last_checked_at = stamp


# ── Alerts ──────────────────────────────────────────────
def _to_alert_row(record: AlertRecord) -> AlertRow:
    return AlertRow(
        alert_id=record.alert_id,
        cycle_id=record.cycle_id,
        ticker=record.ticker,
        alert_type=record.alert_type,
        severity=record.severity,
        status=record.status,
        headline=record.headline,
        detail=record.detail,
        canonical_text=record.canonical_text,
        dedup_key=record.dedup_key,
        evidence=json.loads(record.evidence_json or "[]"),
        metrics=json.loads(record.metrics_json or "{}"),
        occurrence_count=record.occurrence_count,
        first_seen_at=_iso_utc(record.first_seen_at),
        last_seen_at=_iso_utc(record.last_seen_at),
        fired_at=_iso_utc(record.fired_at),
        parent_alert_id=record.parent_alert_id,
    )


def record_alert(alert: dict[str, Any], *, cycle_id: str) -> str:
    """
    Persist one fired alert, upserting on ``alert_id``.

    Parameters
    ----------
    alert : dict
        An ``Alert`` as defined in src/monitor/state.py. Passed as a plain dict
        so persistence stays below monitor in the import order — the same
        arrangement record_research_run has with ResearchState.
    cycle_id : str
        The cycle that produced it.

    Returns
    -------
    str
        The alert id.
    """
    with session_scope() as session:
        record = session.scalar(select(AlertRecord).where(AlertRecord.alert_id == alert["alert_id"]))
        if record is None:
            record = AlertRecord(alert_id=alert["alert_id"])
            session.add(record)

        record.cycle_id = cycle_id
        record.ticker = alert.get("ticker", "")
        record.alert_type = alert.get("alert_type", "")
        record.severity = alert.get("severity", "LOW")
        record.status = alert.get("status", "FIRED")
        record.headline = alert.get("headline", "")
        record.detail = alert.get("detail", "")
        record.canonical_text = alert.get("canonical_text", "")
        record.dedup_key = alert.get("dedup_key", "")
        record.evidence_json = json.dumps(alert.get("evidence", []))
        record.metrics_json = json.dumps(alert.get("metrics", {}))
        record.occurrence_count = int(alert.get("occurrence_count", 1))
        record.parent_alert_id = alert.get("parent_alert_id")

        for field in ("first_seen_at", "last_seen_at", "fired_at"):
            raw = alert.get(field)
            if raw:
                setattr(record, field, datetime.fromisoformat(raw))

    return str(alert["alert_id"])


def bump_alert_occurrence(alert_id: str, *, last_seen_at: str | None = None) -> int:
    """
    Increment a parent alert's occurrence count after a suppression.

    Returns
    -------
    int
        The new count, or 0 if the alert is not in SQLite — which happens
        legitimately when the Qdrant point outlives its row (a restored
        database, a manually seeded index), and must not raise mid-cycle.
    """
    with session_scope() as session:
        record = session.scalar(select(AlertRecord).where(AlertRecord.alert_id == alert_id))
        if record is None:
            logger.warning("Suppression matched %s, which has no SQLite row", alert_id)
            return 0
        record.occurrence_count += 1
        record.last_seen_at = datetime.fromisoformat(last_seen_at) if last_seen_at else utcnow()
        return record.occurrence_count


def list_alerts(
    *,
    limit: int = 50,
    ticker: str | None = None,
    severity: str | None = None,
    status: str | None = None,
) -> list[AlertRow]:
    """Return recent alerts, newest first, optionally filtered."""
    with session_scope() as session:
        statement = select(AlertRecord).order_by(AlertRecord.fired_at.desc()).limit(limit)
        if ticker:
            statement = statement.where(AlertRecord.ticker == ticker.strip().upper())
        if severity:
            statement = statement.where(AlertRecord.severity == severity.upper())
        if status:
            statement = statement.where(AlertRecord.status == status.upper())
        return [_to_alert_row(row) for row in session.scalars(statement).all()]


def get_alert(alert_id: str) -> AlertRow | None:
    """Return one alert by id, or None."""
    with session_scope() as session:
        record = session.scalar(select(AlertRecord).where(AlertRecord.alert_id == alert_id))
        return _to_alert_row(record) if record else None


# ── Dedup decisions ─────────────────────────────────────
def record_dedup_decisions(decisions: Iterable[dict[str, Any]], *, cycle_id: str) -> int:
    """
    Append every dedup decision from a cycle, including the FIREs.

    Recording only suppressions would leave Phase 7's threshold sweep with a
    sample containing no negatives — a dataset that can only ever justify the
    threshold that produced it.

    Returns
    -------
    int
        Rows written.
    """
    rows = list(decisions)
    if not rows:
        return 0

    with session_scope() as session:
        for decision in rows:
            session.add(
                DedupDecision(
                    cycle_id=cycle_id,
                    ticker=decision.get("ticker", ""),
                    alert_type=decision.get("alert_type", ""),
                    severity=decision.get("severity", "LOW"),
                    decision=decision.get("decision", ""),
                    reason=decision.get("reason", ""),
                    dedup_key=decision.get("dedup_key", ""),
                    candidate_text=decision.get("candidate_text", ""),
                    parent_alert_id=decision.get("parent_alert_id"),
                    parent_text=decision.get("parent_text", ""),
                    score=float(decision.get("score", 0.0)),
                )
            )
    return len(rows)


def list_dedup_decisions(*, limit: int = 200, decision: str | None = None) -> list[DedupDecisionRow]:
    """Return recent dedup decisions, newest first — the Phase 7 sweep's input."""
    with session_scope() as session:
        statement = select(DedupDecision).order_by(DedupDecision.decided_at.desc()).limit(limit)
        if decision:
            statement = statement.where(DedupDecision.decision == decision.upper())
        return [
            DedupDecisionRow(
                cycle_id=row.cycle_id,
                ticker=row.ticker,
                alert_type=row.alert_type,
                severity=row.severity,
                decision=row.decision,
                reason=row.reason,
                dedup_key=row.dedup_key,
                candidate_text=row.candidate_text,
                parent_alert_id=row.parent_alert_id,
                parent_text=row.parent_text,
                score=row.score,
                decided_at=_iso_utc(row.decided_at),
            )
            for row in session.scalars(statement).all()
        ]


# ── Cycles ──────────────────────────────────────────────
def _to_cycle_row(cycle: MonitorCycle) -> CycleRow:
    return CycleRow(
        cycle_id=cycle.cycle_id,
        status=cycle.status,
        warmup=cycle.warmup,
        tickers=[t for t in cycle.tickers.split(",") if t],
        candidate_count=cycle.candidate_count,
        fired_count=cycle.fired_count,
        suppressed_count=cycle.suppressed_count,
        merged_count=cycle.merged_count,
        error_count=cycle.error_count,
        api_call_count=cycle.api_call_count,
        started_at=_iso_utc(cycle.started_at),
        duration_ms=cycle.duration_ms,
    )


def record_cycle(state: dict[str, Any], *, duration_ms: int, status: str = "COMPLETE") -> str:
    """
    Summarise one monitoring cycle — paused or finished.

    Upserts on ``cycle_id``, so a cycle that pauses for approval and is later
    resumed calls this TWICE: once with ``status="PENDING_APPROVAL"`` when
    run_cycle's invoke() returns with an interrupt, and once with
    ``status="COMPLETE"`` when resume_cycle finishes it. The second call
    overwrites the counts recorded by the first — by then `fired`/`merged`
    carry the post-decision truth (REJECTED alerts removed, dispatched ones
    resolved), which the first call could not have known.

    Parameters
    ----------
    state : dict
        The (possibly partial) final MonitorState.
    duration_ms : int
        Wall-clock duration of THIS invocation — the pause itself is not
        counted, since a human may sit on a decision for hours or days.
    status : str, default "COMPLETE"
        "PENDING_APPROVAL" if the graph interrupted before persist_cycle ran.

    Returns
    -------
    str
        The cycle id.
    """
    cycle_id = str(state.get("cycle_id", ""))

    with session_scope() as session:
        cycle = session.scalar(select(MonitorCycle).where(MonitorCycle.cycle_id == cycle_id))
        if cycle is None:
            cycle = MonitorCycle(cycle_id=cycle_id)
            session.add(cycle)

        cycle.status = status
        cycle.warmup = bool(state.get("warmup", False))
        cycle.tickers = ",".join(item["ticker"] for item in state.get("watchlist", []))
        cycle.candidate_count = len(state.get("candidates", []))
        cycle.fired_count = len(state.get("fired", []))
        cycle.suppressed_count = len(state.get("suppressed", []))
        cycle.merged_count = len(state.get("merged", []))
        cycle.error_count = len(state.get("monitor_errors", []))
        cycle.api_call_count = len(state.get("api_calls", []))
        cycle.duration_ms = duration_ms

        started = state.get("started_at")
        if started:
            cycle.started_at = datetime.fromisoformat(started)

    logger.info("Recorded cycle %s (%s, %d fired)", cycle_id, status, len(state.get("fired", [])))
    return cycle_id


def list_cycles(*, limit: int = 50, status: str | None = None) -> list[CycleRow]:
    """Return recent cycles, newest first, optionally filtered by status."""
    with session_scope() as session:
        statement = select(MonitorCycle).order_by(MonitorCycle.started_at.desc()).limit(limit)
        if status:
            statement = statement.where(MonitorCycle.status == status.upper())
        rows = session.scalars(statement).all()
        return [_to_cycle_row(row) for row in rows]


def get_cycle(cycle_id: str) -> CycleRow | None:
    """Return one cycle by id, or None."""
    with session_scope() as session:
        cycle = session.scalar(select(MonitorCycle).where(MonitorCycle.cycle_id == cycle_id))
        return _to_cycle_row(cycle) if cycle else None
