# ═══════════════════════════════════════════════════════
# FinSight — Tests: Persistence
# ═══════════════════════════════════════════════════════
#
# Every test runs against a temporary SQLite file, so nothing touches
# data/finsight.db and nothing needs the network.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import text

from src.persistence import db as db_module
from src.persistence.db import get_engine, init_db, reset_engine, session_scope
from src.persistence.models import ApiBudget, ResearchRun
from src.persistence.repository import (
    get_budget_status,
    get_research_run,
    list_research_runs,
    record_api_call,
    record_research_run,
)


@pytest.fixture
def temp_db(tmp_path):
    """Point the engine at a throwaway database for one test."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    reset_engine()
    with patch.object(db_module, "DATABASE_URL", url):
        init_db()
        yield url
    reset_engine()


def _state(**overrides):
    state = {
        "query": "What was Apple's revenue?",
        "final_answer": "Revenue was $416.2B [SRC:EDGAR:0000320193-25-000079]",
        "findings": [{"a": 1}, {"b": 2}],
        "tool_calls": [{"c": 3}],
        "errors": [],
        "repair_count": 0,
        "plan": {"selected_agents": ["fundamentals"], "tickers": ["AAPL"]},
        "verification": {
            "citation_coverage": 1.0,
            "passed": True,
            "unsupported_claims": [],
        },
    }
    state.update(overrides)
    return state


class TestEngine:
    def test_wal_is_enabled(self, temp_db):
        """
        WAL is the load-bearing pragma: the API writes runs on the same file
        the scheduler reads budgets from, and the default journal blocks
        readers for the whole write.
        """
        with get_engine().connect() as connection:
            assert connection.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"

    def test_pragmas_apply_to_every_connection_not_just_the_first(self, temp_db):
        # Pragmas are connection state and the pool opens new connections as
        # concurrency grows.
        for _ in range(3):
            with get_engine().connect() as connection:
                assert connection.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"

    def test_init_db_is_idempotent(self, temp_db):
        init_db()
        init_db()
        with session_scope() as session:
            assert session.query(ResearchRun).count() == 0

    def test_session_scope_commits_on_success(self, temp_db):
        with session_scope() as session:
            session.add(ResearchRun(thread_id="t1", query="q"))
        with session_scope() as session:
            assert session.query(ResearchRun).count() == 1

    def test_session_scope_rolls_back_on_error(self, temp_db):
        with pytest.raises(RuntimeError):
            with session_scope() as session:
                session.add(ResearchRun(thread_id="t2", query="q"))
                raise RuntimeError("boom")

        with session_scope() as session:
            assert session.query(ResearchRun).count() == 0


class TestResearchRuns:
    def test_a_run_is_recorded(self, temp_db):
        record_research_run(_state(), thread_id="abc", latency_ms=1200)
        assert get_research_run("abc") is not None

    def test_the_summary_is_detached_from_the_session(self, temp_db):
        """
        Returning an ORM object across the session boundary hands the caller
        something that raises DetachedInstanceError the moment a route
        serialises it.
        """
        record_research_run(_state(), thread_id="abc", latency_ms=1200)
        summary = get_research_run("abc")
        assert isinstance(summary, dict)
        assert summary["query"] == "What was Apple's revenue?"

    def test_verification_metrics_are_captured(self, temp_db):
        record_research_run(_state(), thread_id="abc", latency_ms=1)
        summary = get_research_run("abc")
        assert summary["citation_coverage"] == 1.0
        assert summary["verification_passed"]

    def test_a_failed_verification_is_recorded_as_such(self, temp_db):
        state = _state(
            verification={
                "citation_coverage": 0.5,
                "passed": False,
                "unsupported_claims": [{"claim": "x"}],
            }
        )
        record_research_run(state, thread_id="abc", latency_ms=1)
        summary = get_research_run("abc")
        assert not summary["verification_passed"]
        assert summary["unsupported_count"] == 1

    def test_agents_and_tickers_round_trip_as_lists(self, temp_db):
        state = _state(plan={"selected_agents": ["fundamentals", "macro"], "tickers": ["AAPL", "MSFT"]})
        record_research_run(state, thread_id="abc", latency_ms=1)
        summary = get_research_run("abc")
        assert summary["agents_used"] == ["fundamentals", "macro"]
        assert summary["tickers"] == ["AAPL", "MSFT"]

    def test_an_empty_plan_yields_empty_lists_not_a_blank_entry(self, temp_db):
        # "".split(",") is [""] — one empty agent, which would show up in the UI.
        record_research_run(_state(plan={}), thread_id="abc", latency_ms=1)
        assert get_research_run("abc")["agents_used"] == []

    def test_re_running_a_thread_updates_rather_than_duplicates(self, temp_db):
        record_research_run(_state(), thread_id="abc", latency_ms=1)
        record_research_run(_state(query="a different question"), thread_id="abc", latency_ms=2)

        runs = list_research_runs()
        assert len(runs) == 1
        assert runs[0]["query"] == "a different question"

    def test_missing_run_returns_none(self, temp_db):
        assert get_research_run("nope") is None

    def test_runs_are_listed_newest_first(self, temp_db):
        for i in range(3):
            record_research_run(_state(query=f"q{i}"), thread_id=f"t{i}", latency_ms=1)
        assert len(list_research_runs()) == 3

    def test_the_listing_respects_its_limit(self, temp_db):
        for i in range(5):
            record_research_run(_state(), thread_id=f"t{i}", latency_ms=1)
        assert len(list_research_runs(limit=2)) == 2

    def test_a_draft_answer_is_stored_when_finalize_produced_nothing(self, temp_db):
        state = _state(final_answer="")
        state["draft_answer"] = "the draft"
        record_research_run(state, thread_id="abc", latency_ms=1)
        assert get_research_run("abc")["final_answer"] == "the draft"


class TestBudgets:
    def test_a_call_is_counted(self, temp_db):
        assert record_api_call("fmp") == 1

    def test_calls_accumulate_within_a_day(self, temp_db):
        record_api_call("fmp")
        record_api_call("fmp")
        assert record_api_call("fmp") == 3

    def test_providers_are_counted_separately(self, temp_db):
        record_api_call("fmp", count=5)
        assert record_api_call("finnhub") == 1

    def test_status_lists_every_budgeted_provider_even_at_zero(self, temp_db):
        """
        An empty response reads as "budgets are not being tracked", which is a
        very different thing from "nothing has been spent".
        """
        from src.data.config import DAILY_BUDGETS

        assert {s["provider"] for s in get_budget_status()} == set(DAILY_BUDGETS)

    def test_remaining_reflects_usage(self, temp_db):
        from src.data.config import DAILY_BUDGETS

        record_api_call("fmp", count=10)
        status = next(s for s in get_budget_status() if s["provider"] == "fmp")
        assert status["used"] == 10
        assert status["remaining"] == DAILY_BUDGETS["fmp"] - 10

    def test_the_soft_limit_trips_before_exhaustion(self, temp_db):
        from src.data.config import BUDGET_SOFT_LIMIT, DAILY_BUDGETS

        record_api_call("fmp", count=int(DAILY_BUDGETS["fmp"] * BUDGET_SOFT_LIMIT))
        status = next(s for s in get_budget_status() if s["provider"] == "fmp")
        assert status["soft_limit_reached"]
        assert not status["exhausted"]

    def test_a_zero_limit_provider_reads_as_exhausted_not_unlimited(self, temp_db):
        # Alpha Vantage's 25/day is unusable, so DAILY_BUDGETS sets it to 0.
        # "0 remaining" must not be reported as an untouched budget.
        zero = [s for s in get_budget_status() if s["limit"] == 0]
        assert zero and all(s["exhausted"] for s in zero)

    def test_counters_are_bucketed_by_utc_day(self, temp_db):
        record_api_call("fmp", count=3)
        with session_scope() as session:
            rows = session.query(ApiBudget).all()
            assert len(rows) == 1
            assert len(rows[0].day) == 10  # ISO date


# ═══════════════════════════════════════════════════════
# Monitoring subsystem (Phase 6)
# ═══════════════════════════════════════════════════════


def _alert(**overrides):
    alert = {
        "alert_id": "a1",
        "ticker": "AAPL",
        "alert_type": "PRICE_MOVE",
        "severity": "MED",
        "status": "FIRED",
        "headline": "AAPL fell 5.2%",
        "detail": "Closed at 210.11, down 5.2% on 2.1x average volume.",
        "canonical_text": "AAPL Apple Inc. | PRICE_MOVE | sharp single-day decline on elevated volume",
        "dedup_key": "abc123",
        "evidence": [{"source_type": "YFINANCE", "source_id": "AAPL@2026-08-03"}],
        "metrics": {"change_pct_1d": -5.2},
        "occurrence_count": 1,
        "first_seen_at": "2026-08-03T12:00:00+00:00",
        "last_seen_at": "2026-08-03T12:00:00+00:00",
        "fired_at": "2026-08-03T12:00:00+00:00",
        "parent_alert_id": None,
    }
    alert.update(overrides)
    return alert


class TestWatchlist:
    def test_add_then_list(self, temp_db):
        from src.persistence.repository import add_watch_item, list_watchlist

        add_watch_item("aapl", company_name="Apple Inc.")
        rows = list_watchlist()
        assert [r["ticker"] for r in rows] == ["AAPL"]
        assert rows[0]["company_name"] == "Apple Inc."
        assert not rows[0]["warmed_up"]

    def test_remove_is_a_soft_delete(self, temp_db):
        from src.persistence.repository import add_watch_item, list_watchlist, remove_watch_item

        add_watch_item("AAPL")
        assert remove_watch_item("AAPL")
        assert list_watchlist() == []
        assert [r["ticker"] for r in list_watchlist(active_only=False)] == ["AAPL"]

    def test_removing_twice_reports_false(self, temp_db):
        from src.persistence.repository import add_watch_item, remove_watch_item

        add_watch_item("AAPL")
        assert remove_watch_item("AAPL")
        assert not remove_watch_item("AAPL")

    def test_readding_does_not_reset_warmed_up(self, temp_db):
        """
        A ticker that already warmed up has its events in the dedup index.
        Re-warming would re-upsert the same points and delay real alerts for a
        whole cycle.
        """
        from src.persistence.repository import add_watch_item, mark_warmed_up, remove_watch_item

        add_watch_item("AAPL")
        mark_warmed_up(["AAPL"])
        remove_watch_item("AAPL")

        assert add_watch_item("AAPL")["warmed_up"]

    def test_mark_warmed_up_ignores_unknown_tickers(self, temp_db):
        from src.persistence.repository import add_watch_item, mark_warmed_up

        add_watch_item("AAPL")
        assert mark_warmed_up(["AAPL", "ZZZZ"]) == 1


class TestCheckpoints:
    def test_missing_checkpoint_is_absent_not_zero(self, temp_db):
        """
        A never-checked pair must be MISSING so the monitor falls back to its
        default lookback. A zero timestamp would read as 1970 and pull the
        company's entire filing history as "new".
        """
        from src.persistence.repository import get_checkpoints

        assert get_checkpoints(["AAPL"]) == {}

    def test_set_then_get_round_trips(self, temp_db):
        from datetime import datetime, timezone

        from src.persistence.repository import get_checkpoints, set_checkpoint

        when = datetime(2026, 8, 1, 9, 30, tzinfo=timezone.utc)
        set_checkpoint("aapl", "filing", when=when)

        assert get_checkpoints(["AAPL"]) == {"AAPL:filing": when.isoformat()}

    def test_second_set_advances_rather_than_duplicating(self, temp_db):
        from datetime import datetime, timezone

        from src.persistence.repository import get_checkpoints, set_checkpoint

        set_checkpoint("AAPL", "filing", when=datetime(2026, 8, 1, tzinfo=timezone.utc))
        set_checkpoint("AAPL", "filing", when=datetime(2026, 8, 2, tzinfo=timezone.utc))

        checkpoints = get_checkpoints(["AAPL"])
        assert len(checkpoints) == 1
        assert checkpoints["AAPL:filing"].startswith("2026-08-02")

    def test_keys_are_flat_strings_so_they_survive_checkpoint_serialisation(self, temp_db):
        from src.persistence.repository import get_checkpoints, set_checkpoint

        set_checkpoint("AAPL", "news")
        key = next(iter(get_checkpoints()))
        assert isinstance(key, str) and ":" in key


class TestAlerts:
    def test_record_then_read_back(self, temp_db):
        from src.persistence.repository import get_alert, record_alert

        record_alert(_alert(), cycle_id="c1")
        row = get_alert("a1")

        assert row is not None
        assert row["ticker"] == "AAPL"
        assert row["metrics"]["change_pct_1d"] == -5.2
        assert row["evidence"][0]["source_id"] == "AAPL@2026-08-03"

    def test_recording_the_same_id_upserts(self, temp_db):
        from src.persistence.repository import list_alerts, record_alert

        record_alert(_alert(), cycle_id="c1")
        record_alert(_alert(severity="HIGH"), cycle_id="c2")

        rows = list_alerts()
        assert len(rows) == 1
        assert rows[0]["severity"] == "HIGH"

    def test_bump_increments_rather_than_inserting(self, temp_db):
        from src.persistence.repository import bump_alert_occurrence, get_alert, list_alerts, record_alert

        record_alert(_alert(), cycle_id="c1")
        assert bump_alert_occurrence("a1") == 2
        assert bump_alert_occurrence("a1") == 3

        assert len(list_alerts()) == 1
        assert get_alert("a1")["occurrence_count"] == 3

    def test_bumping_an_unknown_alert_returns_zero_rather_than_raising(self, temp_db):
        """
        A Qdrant point can outlive its SQLite row — a restored database, a
        hand-seeded index. That must not abort the cycle mid-flight.
        """
        from src.persistence.repository import bump_alert_occurrence

        assert bump_alert_occurrence("ghost") == 0

    def test_filters_compose(self, temp_db):
        from src.persistence.repository import list_alerts, record_alert

        record_alert(_alert(alert_id="a1", ticker="AAPL", severity="HIGH"), cycle_id="c1")
        record_alert(_alert(alert_id="a2", ticker="MSFT", severity="LOW"), cycle_id="c1")

        assert len(list_alerts(ticker="AAPL")) == 1
        assert len(list_alerts(severity="HIGH")) == 1
        assert len(list_alerts(ticker="MSFT", severity="HIGH")) == 0


class TestDedupDecisions:
    def test_fires_are_recorded_too_not_only_suppressions(self, temp_db):
        """
        The Phase 7 sweep needs negatives. A log of only suppressions can
        justify the threshold that produced it and nothing else.
        """
        from src.persistence.repository import list_dedup_decisions, record_dedup_decisions

        record_dedup_decisions(
            [
                {"ticker": "AAPL", "decision": "FIRE", "score": 0.0},
                {"ticker": "AAPL", "decision": "SUPPRESS_SEMANTIC", "score": 0.94, "parent_alert_id": "a1"},
            ],
            cycle_id="c1",
        )

        rows = list_dedup_decisions()
        assert {r["decision"] for r in rows} == {"FIRE", "SUPPRESS_SEMANTIC"}
        assert next(r for r in rows if r["decision"] == "SUPPRESS_SEMANTIC")["score"] == 0.94

    def test_empty_input_writes_nothing(self, temp_db):
        from src.persistence.repository import record_dedup_decisions

        assert record_dedup_decisions([], cycle_id="c1") == 0

    def test_filter_by_decision(self, temp_db):
        from src.persistence.repository import list_dedup_decisions, record_dedup_decisions

        record_dedup_decisions(
            [{"ticker": "AAPL", "decision": "FIRE"}, {"ticker": "MSFT", "decision": "MERGE"}],
            cycle_id="c1",
        )
        assert len(list_dedup_decisions(decision="merge")) == 1


class TestCycles:
    def test_record_summarises_the_state(self, temp_db):
        from src.persistence.repository import get_cycle, record_cycle

        record_cycle(
            {
                "cycle_id": "c1",
                "started_at": "2026-08-03T12:00:00+00:00",
                "warmup": False,
                "watchlist": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
                "candidates": [1, 2, 3],
                "fired": [1],
                "suppressed": [1, 1],
                "merged": [],
                "monitor_errors": ["boom"],
                "api_calls": [1, 1, 1, 1],
            },
            duration_ms=4200,
        )

        row = get_cycle("c1")
        assert row is not None
        assert row["tickers"] == ["AAPL", "MSFT"]
        assert (row["candidate_count"], row["fired_count"], row["suppressed_count"]) == (3, 1, 2)
        assert row["error_count"] == 1
        assert row["duration_ms"] == 4200

    def test_rerecording_a_cycle_upserts(self, temp_db):
        from src.persistence.repository import list_cycles, record_cycle

        state = {"cycle_id": "c1", "watchlist": [], "fired": []}
        record_cycle(state, duration_ms=1)
        record_cycle({**state, "fired": [1, 2]}, duration_ms=2)

        rows = list_cycles()
        assert len(rows) == 1
        assert rows[0]["fired_count"] == 2

    def test_status_defaults_to_complete(self, temp_db):
        from src.persistence.repository import get_cycle, record_cycle

        record_cycle({"cycle_id": "c1", "watchlist": [], "fired": []}, duration_ms=1)
        assert get_cycle("c1")["status"] == "COMPLETE"

    def test_a_paused_cycle_is_recorded_pending_approval_then_flips_on_resume(self, temp_db):
        """
        The exact sequence run_cycle / resume_cycle produces: one row, upserted
        twice, whose status is the queryable pointer into what the checkpointer
        actually holds.
        """
        from src.persistence.repository import get_cycle, list_cycles, record_cycle

        state = {"cycle_id": "c1", "watchlist": [], "fired": [1]}
        record_cycle(state, duration_ms=50, status="PENDING_APPROVAL")

        assert get_cycle("c1")["status"] == "PENDING_APPROVAL"
        assert len(list_cycles(status="PENDING_APPROVAL")) == 1
        assert len(list_cycles(status="COMPLETE")) == 0

        record_cycle(state, duration_ms=5, status="COMPLETE")

        assert get_cycle("c1")["status"] == "COMPLETE"
        assert len(list_cycles(status="PENDING_APPROVAL")) == 0
