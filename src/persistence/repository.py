# ═══════════════════════════════════════════════════════
# FinSight — Repository
# ═══════════════════════════════════════════════════════
#
# Purpose : Every query the application makes, in one place, returning plain
#           TypedDicts rather than ORM objects.
#
# Public API:
#   record_research_run(state, latency_ms)  -> str
#   list_research_runs(limit)               -> list[ResearchRunSummary]
#   get_research_run(thread_id)             -> ResearchRunSummary | None
#   record_api_call(provider, count)        -> int
#   get_budget_status()                     -> list[BudgetStatus]
#
# ══ WHY DETACHED DICTS, NOT ORM OBJECTS ══
#   A session_scope() commits and closes when it exits. Handing an ORM object
#   back across that boundary gives the caller something whose unloaded
#   attributes raise DetachedInstanceError the moment a route tries to
#   serialise it. Reading everything inside the scope and returning a dict
#   makes that impossible rather than merely unlikely.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from sqlalchemy import select

from src.data.config import BUDGET_SOFT_LIMIT, DAILY_BUDGETS
from src.persistence.db import session_scope
from src.persistence.models import ApiBudget, ResearchRun, utcnow

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
        created_at=run.created_at.isoformat(),
    )


# ── Research runs ───────────────────────────────────────
def record_research_run(state: dict[str, Any], *, thread_id: str, latency_ms: int) -> str:
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


def list_research_runs(*, limit: int = 50) -> list[ResearchRunSummary]:
    """Return recent runs, newest first."""
    with session_scope() as session:
        rows = session.scalars(select(ResearchRun).order_by(ResearchRun.created_at.desc()).limit(limit)).all()
        return [_to_summary(row) for row in rows]


def get_research_run(thread_id: str) -> ResearchRunSummary | None:
    """Return one run by thread id, or None."""
    with session_scope() as session:
        run = session.scalar(select(ResearchRun).where(ResearchRun.thread_id == thread_id))
        return _to_summary(run) if run else None


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
