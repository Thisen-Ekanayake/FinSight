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
from sqlalchemy import Column, Integer, String, Table, inspect, text

from src.persistence import db as db_module
from src.persistence.db import get_engine, init_db, reset_engine, session_scope
from src.persistence.models import Base, ApiBudget, ResearchRun
from src.persistence.repository import (
    LEGACY_SUBJECT,
    can_use_thread,
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


# The research_runs schema as it stood before ownership existed, written out
# by hand. A create_all() database already has every column and index, so it
# would exercise none of _sync_additive_schema — the only way to test a
# migration is to build the thing being migrated FROM.
_PRE_OWNERSHIP_DDL = """
CREATE TABLE research_runs (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    thread_id VARCHAR(64) NOT NULL UNIQUE,
    query TEXT NOT NULL,
    final_answer TEXT NOT NULL,
    citation_coverage FLOAT NOT NULL,
    verification_passed BOOLEAN NOT NULL,
    unsupported_count INTEGER NOT NULL,
    repair_count INTEGER NOT NULL,
    agents_used VARCHAR(256) NOT NULL,
    tickers VARCHAR(128) NOT NULL,
    finding_count INTEGER NOT NULL,
    tool_call_count INTEGER NOT NULL,
    error_count INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    created_at DATETIME NOT NULL
)
"""

_LEGACY_ROW = (
    "INSERT INTO research_runs VALUES "
    "(1,'research:old','a question from before','an answer',1.0,1,0,0,'','',0,0,0,5,"
    "'2026-01-01 00:00:00')"
)


@pytest.fixture
def legacy_db(tmp_path):
    """A database at the schema this change migrates FROM, holding one row."""
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    reset_engine()
    with patch.object(db_module, "DATABASE_URL", url):
        with get_engine().begin() as connection:
            connection.execute(text(_PRE_OWNERSHIP_DDL))
            connection.execute(text(_LEGACY_ROW))
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


# Every run in TestResearchRuns belongs to one account, so the ownership rule
# is never what is under test there — TestRunOwnership below is where that
# lives. `unlimited=False` is deliberate: these read back as the plain owner,
# with no legacy-row visibility to mask a scoping mistake.
ALICE = "google-sub-alice"
BOB = "google-sub-bob"


def _mine(thread_id, subject=ALICE):
    return get_research_run(thread_id, subject=subject, unlimited=False)


class TestAdditiveSchema:
    """
    init_db() bringing a LIVE table up to the models' schema.

    create_all() skips any table that already exists — including that table's
    declared indexes — so before _sync_additive_schema every one of these was
    a hand-written migration.
    """

    def test_a_missing_column_is_added_to_a_live_table(self, legacy_db):
        init_db()
        columns = {c["name"] for c in inspect(get_engine()).get_columns("research_runs")}
        assert {"subject", "owner_email"} <= columns

    def test_a_legacy_row_backfills_to_the_empty_subject(self, legacy_db):
        """
        NOT NULL, not NULL. If subject were nullable this row would come back
        NULL and every `subject IN (...)` predicate would silently stop
        matching it — the row would vanish for everyone rather than becoming
        the operator's.
        """
        init_db()
        with get_engine().connect() as connection:
            subject, kind = connection.execute(text("SELECT subject, typeof(subject) FROM research_runs")).one()

        assert subject == LEGACY_SUBJECT
        assert kind == "text"

    def test_the_legacy_row_survives_intact(self, legacy_db):
        """Additive means additive — the migration must not touch the data."""
        init_db()
        assert get_research_run("research:old", subject=ALICE, unlimited=True)["query"] == ("a question from before")

    def test_a_missing_index_is_created_on_a_live_table(self, legacy_db):
        """
        The failure mode nobody looks for: create_all() would leave a live
        table permanently without any index added after its creation.
        """
        init_db()
        names = {i["name"] for i in inspect(get_engine()).get_indexes("research_runs")}
        assert "ix_research_runs_subject_created" in names

    def test_other_tables_are_still_created(self, legacy_db):
        """The helper runs after create_all, not instead of it."""
        init_db()
        assert "free_query_quotas" in inspect(get_engine()).get_table_names()

    def test_it_is_a_no_op_on_a_fresh_database(self, temp_db):
        init_db()
        init_db()
        columns = [c["name"] for c in inspect(get_engine()).get_columns("research_runs")]
        assert columns.count("subject") == 1

    def test_running_it_twice_on_a_migrated_database_is_a_no_op(self, legacy_db):
        init_db()
        init_db()
        columns = [c["name"] for c in inspect(get_engine()).get_columns("research_runs")]
        assert columns.count("subject") == 1

    def test_a_not_null_column_with_no_server_default_is_refused(self, temp_db):
        """
        SQLite cannot ADD COLUMN a NOT NULL with nothing to backfill from.
        Failing here turns that into a test failure at review time rather than
        a broken deploy on the VM.
        """
        table = Table(
            "additive_probe",
            Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("ok", String(8), nullable=True),
        )
        try:
            init_db()  # creates additive_probe with just id/ok
            table.append_column(Column("bad", String(8), nullable=False))
            with pytest.raises(RuntimeError, match="server_default"):
                init_db()
        finally:
            Base.metadata.remove(table)

    def test_a_unique_column_is_refused(self, temp_db):
        """ADD COLUMN cannot create a key. That genuinely needs a migration."""
        table = Table(
            "additive_probe_unique",
            Base.metadata,
            Column("id", Integer, primary_key=True),
            Column("ok", String(8), nullable=True),
        )
        try:
            init_db()
            table.append_column(Column("tag", String(8), unique=True))
            with pytest.raises(RuntimeError, match="key column"):
                init_db()
        finally:
            Base.metadata.remove(table)


class TestResearchRuns:
    def test_a_run_is_recorded(self, temp_db):
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1200)
        assert _mine("abc") is not None

    def test_the_summary_is_detached_from_the_session(self, temp_db):
        """
        Returning an ORM object across the session boundary hands the caller
        something that raises DetachedInstanceError the moment a route
        serialises it.
        """
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1200)
        summary = _mine("abc")
        assert isinstance(summary, dict)
        assert summary["query"] == "What was Apple's revenue?"

    def test_verification_metrics_are_captured(self, temp_db):
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1)
        summary = _mine("abc")
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
        record_research_run(state, thread_id="abc", subject=ALICE, latency_ms=1)
        summary = _mine("abc")
        assert not summary["verification_passed"]
        assert summary["unsupported_count"] == 1

    def test_agents_and_tickers_round_trip_as_lists(self, temp_db):
        state = _state(plan={"selected_agents": ["fundamentals", "macro"], "tickers": ["AAPL", "MSFT"]})
        record_research_run(state, thread_id="abc", subject=ALICE, latency_ms=1)
        summary = _mine("abc")
        assert summary["agents_used"] == ["fundamentals", "macro"]
        assert summary["tickers"] == ["AAPL", "MSFT"]

    def test_an_empty_plan_yields_empty_lists_not_a_blank_entry(self, temp_db):
        # "".split(",") is [""] — one empty agent, which would show up in the UI.
        record_research_run(_state(plan={}), thread_id="abc", subject=ALICE, latency_ms=1)
        assert _mine("abc")["agents_used"] == []

    def test_re_running_a_thread_updates_rather_than_duplicates(self, temp_db):
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1)
        record_research_run(_state(query="a different question"), thread_id="abc", subject=ALICE, latency_ms=2)

        runs = list_research_runs(subject=ALICE, unlimited=False)
        assert len(runs) == 1
        assert runs[0]["query"] == "a different question"

    def test_missing_run_returns_none(self, temp_db):
        assert _mine("nope") is None

    def test_runs_are_listed_newest_first(self, temp_db):
        for i in range(3):
            record_research_run(_state(query=f"q{i}"), thread_id=f"t{i}", subject=ALICE, latency_ms=1)
        assert len(list_research_runs(subject=ALICE, unlimited=False)) == 3

    def test_the_listing_respects_its_limit(self, temp_db):
        for i in range(5):
            record_research_run(_state(), thread_id=f"t{i}", subject=ALICE, latency_ms=1)
        assert len(list_research_runs(subject=ALICE, unlimited=False, limit=2)) == 2

    def test_a_draft_answer_is_stored_when_finalize_produced_nothing(self, temp_db):
        state = _state(final_answer="")
        state["draft_answer"] = "the draft"
        record_research_run(state, thread_id="abc", subject=ALICE, latency_ms=1)
        assert _mine("abc")["final_answer"] == "the draft"

    def test_the_owner_email_is_stored_but_never_serialised(self, temp_db):
        """A label for whoever reads the table, not a field any client receives."""
        record_research_run(_state(), thread_id="abc", subject=ALICE, email="alice@example.com", latency_ms=1)

        with session_scope() as session:
            assert session.query(ResearchRun).one().owner_email == "alice@example.com"
        assert "owner_email" not in _mine("abc")


class TestRunOwnership:
    """Who can see which runs — the rule, from the storage side."""

    def test_a_run_is_invisible_to_another_account(self, temp_db):
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1)

        assert list_research_runs(subject=BOB, unlimited=False) == []
        assert get_research_run("abc", subject=BOB, unlimited=False) is None

    def test_the_unlimited_tier_does_not_see_another_accounts_runs(self, temp_db):
        """
        Own + unattributable, NOT everything. The operator is exempt from the
        query meter, not from other people's privacy.
        """
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1)

        assert list_research_runs(subject=BOB, unlimited=True) == []

    def test_a_legacy_row_is_visible_only_to_the_unlimited_tier(self, temp_db):
        """Rows from before ownership existed, and CLI runs. Nobody owns them."""
        record_research_run(_state(), thread_id="old", subject=LEGACY_SUBJECT, latency_ms=1)

        assert list_research_runs(subject=ALICE, unlimited=False) == []
        assert len(list_research_runs(subject=ALICE, unlimited=True)) == 1

    def test_can_use_thread_treats_an_unrecorded_thread_as_legacy(self, temp_db):
        """
        A thread with no row is unattributable, not denied — it may have
        crashed before its summary was written. Same treatment as a legacy
        row, which keeps it fail-closed for free accounts.
        """
        assert can_use_thread("research:never-seen", subject=ALICE, unlimited=False) is False
        assert can_use_thread("research:never-seen", subject=ALICE, unlimited=True) is True

    def test_can_use_thread_admits_the_owner_and_refuses_everyone_else(self, temp_db):
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1)

        assert can_use_thread("abc", subject=ALICE, unlimited=False) is True
        assert can_use_thread("abc", subject=BOB, unlimited=False) is False
        # Not even the operator: a row with a real owner is that account's.
        assert can_use_thread("abc", subject=BOB, unlimited=True) is False

    def test_recording_does_not_overwrite_a_run_owned_by_someone_else(self, temp_db):
        """
        The last line of defence behind _resolve_thread. Unreachable through
        the API, so if this ever fires something else has already failed.
        """
        record_research_run(_state(), thread_id="abc", subject=ALICE, latency_ms=1)
        record_research_run(_state(query="hijacked"), thread_id="abc", subject=BOB, latency_ms=1)

        summary = _mine("abc")
        assert summary["query"] == "What was Apple's revenue?"
        assert list_research_runs(subject=BOB, unlimited=False) == []

    def test_a_re_run_claims_a_legacy_row_rather_than_orphaning_it(self, temp_db):
        """Re-asking something from before ownership existed makes it yours."""
        record_research_run(_state(), thread_id="abc", subject=LEGACY_SUBJECT, latency_ms=1)
        record_research_run(_state(query="asked again"), thread_id="abc", subject=ALICE, latency_ms=1)

        runs = list_research_runs(subject=ALICE, unlimited=False)
        assert len(runs) == 1 and runs[0]["query"] == "asked again"

    def test_an_empty_subject_is_refused(self, temp_db):
        """
        An empty subject would match every legacy row — the exact failure the
        whole change exists to prevent. It cannot arrive from require_identity,
        so this is a tripwire rather than a code path.
        """
        with pytest.raises(ValueError):
            list_research_runs(subject="", unlimited=False)
        with pytest.raises(ValueError):
            can_use_thread("abc", subject="", unlimited=True)


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
