# ═══════════════════════════════════════════════════════
# FinSight — Tests: Monitoring Scheduler
# ═══════════════════════════════════════════════════════
#
# Offline only. APScheduler itself is real (it is in-process and does not
# touch the network); run_cycle is mocked.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

from src.monitor import scheduler as scheduler_module
from src.monitor.scheduler import next_run_at, start_scheduler, stop_scheduler


class TestDisabledByDefault:
    def test_returns_none_when_disabled(self):
        with patch.object(scheduler_module, "MONITOR_SCHEDULER_ENABLED", False):
            assert start_scheduler(checkpointer=object()) is None

    def test_stopping_none_is_a_no_op(self):
        stop_scheduler(None)  # must not raise

    def test_next_run_at_is_none_when_disabled(self):
        assert next_run_at(None) is None


class TestEnabled:
    # AsyncIOScheduler.start() calls asyncio.get_running_loop(), matching how
    # it is actually started — inside main.py's async lifespan — so these
    # need a running loop too, not a synchronous test function.

    async def test_starting_registers_the_job_on_the_configured_cadence(self):
        with (
            patch.object(scheduler_module, "MONITOR_SCHEDULER_ENABLED", True),
            patch.object(scheduler_module, "MONITOR_CADENCE_HOURS", 6),
        ):
            job_scheduler = start_scheduler(checkpointer=object())
        try:
            job = job_scheduler.get_job(scheduler_module.JOB_ID)
            assert job is not None
            assert job.trigger.interval.total_seconds() == 6 * 3600
        finally:
            stop_scheduler(job_scheduler)

    async def test_next_run_at_reports_an_iso_timestamp(self):
        with patch.object(scheduler_module, "MONITOR_SCHEDULER_ENABLED", True):
            job_scheduler = start_scheduler(checkpointer=object())
        try:
            stamp = next_run_at(job_scheduler)
            assert stamp is not None
            assert "T" in stamp  # ISO 8601
        finally:
            stop_scheduler(job_scheduler)

    async def test_stopping_a_running_scheduler_stops_it(self):
        import asyncio

        with patch.object(scheduler_module, "MONITOR_SCHEDULER_ENABLED", True):
            job_scheduler = start_scheduler(checkpointer=object())
        stop_scheduler(job_scheduler)
        # AsyncIOScheduler's shutdown is itself scheduled on the loop rather
        # than applied synchronously — one tick is enough for it to land.
        await asyncio.sleep(0)
        assert job_scheduler.running is False


class TestScheduledCycle:
    async def test_a_failing_cycle_does_not_raise(self):
        """
        APScheduler drops a job from all future runs if it raises. A single
        bad cycle must not silently end monitoring until the process restarts.
        """
        from src.monitor.scheduler import _run_scheduled_cycle

        with patch("src.monitor.graph.run_cycle", side_effect=RuntimeError("qdrant down")):
            await _run_scheduled_cycle(checkpointer=object())  # must not raise

    async def test_a_successful_cycle_logs_the_report(self, caplog):
        import logging

        from src.monitor.scheduler import _run_scheduled_cycle

        state = {"cycle_id": "c1", "fired": [], "merged": [], "suppressed": [], "candidates": []}

        with (
            patch("src.monitor.graph.run_cycle", return_value=state),
            caplog.at_level(logging.INFO),
        ):
            await _run_scheduled_cycle(checkpointer=object())

        assert "c1" in caplog.text
