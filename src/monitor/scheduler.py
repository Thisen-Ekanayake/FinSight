# ═══════════════════════════════════════════════════════
# FinSight — Monitoring Scheduler
# ═══════════════════════════════════════════════════════
#
# Purpose : Run a monitoring cycle automatically on MONITOR_CADENCE_HOURS,
#           inside the API process.
#
# Public API:
#   start_scheduler(checkpointer) -> AsyncIOScheduler | None
#   stop_scheduler(scheduler)
#   next_run_at(scheduler) -> str | None
#
# ══ WHY IN THE API PROCESS, NOT A SEPARATE CRON ══
#   The API already holds the one long-lived resource a scheduled cycle
#   needs — the checkpointer (see main.py's lifespan). A cron invocation of
#   `run_monitor.sh --once` works too (it opens its own connection to the
#   same checkpoint database), but there is no reason to run a second
#   always-on process when the API already is one for everything else here.
#
# ══ WHY OFF BY DEFAULT ══
#   `make api` starting up and silently beginning to hit EDGAR, Finnhub, and
#   yfinance on a timer is exactly the kind of surprise MONITOR_SCHEDULER_ENABLED
#   exists to prevent. Enable it once the watchlist has been warmed up.
#
# ══ WHY A PAUSED CYCLE IS NOT A SCHEDULER PROBLEM ══
#   A HIGH alert pauses run_cycle via interrupt() same as always. The
#   scheduled job does not wait for a human — it logs that a decision is
#   pending and returns; the NEXT scheduled tick starts a fresh cycle
#   regardless, because monitoring the rest of the watchlist must not stall
#   on one ticker's unresolved approval. The paused cycle sits durably in the
#   checkpointer exactly as it would from a CLI or API-triggered run, waiting
#   for `--pending` / `--resume` or the API's resume endpoint.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.monitor.config import MONITOR_CADENCE_HOURS, MONITOR_SCHEDULER_ENABLED

logger = logging.getLogger(__name__)

JOB_ID = "monitor_cycle"


async def _run_scheduled_cycle(checkpointer: Any) -> None:
    """
    One scheduled cycle. Never raises.

    APScheduler drops a job from all FUTURE runs if it raises — a single bad
    cycle (a data-source outage, a transient Qdrant blip) must not silently
    end monitoring until the process restarts.
    """
    from src.monitor.graph import cycle_report, run_cycle

    try:
        state = await asyncio.to_thread(run_cycle, checkpointer=checkpointer)
        logger.info(cycle_report(state))
    except Exception:  # noqa: BLE001
        logger.exception("Scheduled monitoring cycle failed")


def start_scheduler(checkpointer: Any) -> Any | None:
    """
    Start the in-process scheduler, if MONITOR_SCHEDULER_ENABLED.

    Parameters
    ----------
    checkpointer : LangGraph checkpointer
        Passed through to every scheduled run_cycle call — must be the same
        one the API's manual /monitor/cycles endpoint uses, or a HIGH alert
        raised on a scheduled tick would pause into a different connection
        than /monitor/cycles/{id}/resume expects to find it in.

    Returns
    -------
    AsyncIOScheduler or None
        None when disabled — main.py stores whatever this returns and skips
        shutdown when it is None.
    """
    if not MONITOR_SCHEDULER_ENABLED:
        logger.info("Monitor scheduler disabled (set MONITOR_SCHEDULER_ENABLED=true to enable)")
        return None

    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        _run_scheduled_cycle,
        trigger=IntervalTrigger(hours=MONITOR_CADENCE_HOURS),
        args=[checkpointer],
        id=JOB_ID,
        # A cycle still running when the next tick is due must not start a
        # second one on top of it — the two would race on the same dedup
        # index within the same time window. coalesce collapses any ticks
        # missed while one was running into a single catch-up run instead of
        # firing once per missed interval.
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info("Monitor scheduler started: every %d hour(s)", MONITOR_CADENCE_HOURS)
    return scheduler


def stop_scheduler(scheduler: Any | None) -> None:
    """Shut down the scheduler if one is running. A no-op when disabled."""
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("Monitor scheduler stopped")


def next_run_at(scheduler: Any | None) -> str | None:
    """The next scheduled cycle's ISO timestamp, or None if disabled or unknown."""
    if scheduler is None:
        return None
    job = scheduler.get_job(JOB_ID)
    if job is None or job.next_run_time is None:
        return None
    return str(job.next_run_time.isoformat())
