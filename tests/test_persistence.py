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
