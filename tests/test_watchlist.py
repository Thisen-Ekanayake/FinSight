# ═══════════════════════════════════════════════════════
# FinSight — Tests: Watchlist and Watermarks
# ═══════════════════════════════════════════════════════
#
# The lookback bounds are the interesting part. A monitor with no floor
# reports a decade of filings as new on its first run; one with no ceiling
# wakes up after a week of downtime and asks EDGAR for the whole gap across
# every ticker at once.
#
# Temporary SQLite, no network.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.monitor.config import DEFAULT_LOOKBACK_DAYS, MAX_LOOKBACK_DAYS
from src.monitor.watchlist import (
    add_ticker,
    current_watchlist,
    ensure_seeded,
    load_watchlist_node,
    lookback_for,
    remove_ticker,
)
from src.persistence import db as db_module
from src.persistence.db import init_db, reset_engine

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def temp_db(tmp_path):
    """Throwaway database, with the EDGAR name lookup stubbed out."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    reset_engine()
    with patch.object(db_module, "DATABASE_URL", url):
        init_db()
        with patch("src.data.edgar.resolve_company_name", side_effect=lambda t: f"{t.upper()} Corp"):
            yield url
    reset_engine()


class TestSeeding:
    def test_an_empty_watchlist_is_seeded_from_env(self, temp_db):
        from src.monitor.config import DEFAULT_WATCHLIST

        assert ensure_seeded() == len(DEFAULT_WATCHLIST)
        assert {item["ticker"] for item in current_watchlist()} == set(DEFAULT_WATCHLIST)

    def test_seeding_is_a_no_op_once_populated(self, temp_db):
        ensure_seeded()
        assert ensure_seeded() == 0

    def test_a_deliberately_removed_ticker_does_not_come_back(self, temp_db):
        """
        MONITOR_WATCHLIST is a starting point, not a floor. Re-seeding a
        removed ticker on every start would make removal impossible without
        editing .env.
        """
        ensure_seeded()
        removed = current_watchlist()[0]["ticker"]
        remove_ticker(removed)

        ensure_seeded()
        assert removed not in {item["ticker"] for item in current_watchlist()}


class TestAddRemove:
    def test_add_resolves_the_company_name(self, temp_db):
        assert add_ticker("tsla")["company_name"] == "TSLA Corp"

    def test_an_explicit_name_wins_over_the_lookup(self, temp_db):
        assert add_ticker("TSLA", company_name="Tesla, Inc.")["company_name"] == "Tesla, Inc."

    def test_a_failed_name_lookup_falls_back_to_the_ticker(self, temp_db):
        # A cosmetic lookup must never stop a ticker being watched.
        with patch("src.data.edgar.resolve_company_name", return_value=""):
            assert add_ticker("ZZZZ")["company_name"] == "ZZZZ"

    def test_removing_an_unwatched_ticker_reports_false(self, temp_db):
        assert not remove_ticker("NOPE")


class TestLookbackBounds:
    def test_a_missing_watermark_uses_the_default_not_the_epoch(self):
        """
        The floor. Without it a first-ever check reports every filing the
        company has ever made as new.
        """
        since = lookback_for("AAPL", "filing_monitor", {}, now=NOW)
        assert since == NOW - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    def test_a_recent_watermark_is_used_as_is(self):
        stamp = (NOW - timedelta(hours=6)).isoformat()
        assert lookback_for("AAPL", "filing_monitor", {"AAPL:filing": stamp}, now=NOW) == NOW - timedelta(hours=6)

    def test_a_very_stale_watermark_is_clamped(self):
        """
        The ceiling. A process down for a month must not wake up and ask EDGAR
        for a month of history across the whole watchlist in one burst.
        """
        ancient = (NOW - timedelta(days=90)).isoformat()
        since = lookback_for("AAPL", "filing_monitor", {"AAPL:filing": ancient}, now=NOW)
        assert since == NOW - timedelta(days=MAX_LOOKBACK_DAYS)

    def test_a_future_watermark_is_clamped_to_now(self):
        """
        Clock skew or a hand-edited row would otherwise make since > now, and
        the monitor would return nothing forever without ever erroring.
        """
        future = (NOW + timedelta(days=3)).isoformat()
        assert lookback_for("AAPL", "filing_monitor", {"AAPL:filing": future}, now=NOW) == NOW

    def test_an_unparseable_watermark_falls_back_rather_than_raising(self):
        since = lookback_for("AAPL", "filing_monitor", {"AAPL:filing": "not a date"}, now=NOW)
        assert since == NOW - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    def test_a_naive_watermark_is_read_as_utc(self):
        # SQLite drops the offset; comparing naive to aware raises TypeError.
        naive = (NOW - timedelta(hours=2)).replace(tzinfo=None).isoformat()
        assert lookback_for("AAPL", "filing_monitor", {"AAPL:filing": naive}, now=NOW) == NOW - timedelta(hours=2)

    def test_monitors_have_independent_watermarks(self):
        watermarks = {"AAPL:filing": (NOW - timedelta(hours=1)).isoformat()}

        assert lookback_for("AAPL", "filing_monitor", watermarks, now=NOW) == NOW - timedelta(hours=1)
        assert lookback_for("AAPL", "news_monitor", watermarks, now=NOW) == NOW - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    def test_tickers_have_independent_watermarks(self):
        watermarks = {"AAPL:filing": (NOW - timedelta(hours=1)).isoformat()}

        assert lookback_for("MSFT", "filing_monitor", watermarks, now=NOW) == NOW - timedelta(
            days=DEFAULT_LOOKBACK_DAYS
        )


class TestLoadNode:
    def test_the_node_loads_the_stored_watchlist(self, temp_db):
        add_ticker("AAPL")
        add_ticker("MSFT")

        result = load_watchlist_node({"cycle_id": "c1"})
        assert [item["ticker"] for item in result["watchlist"]] == ["AAPL", "MSFT"]

    def test_an_injected_watchlist_wins(self, temp_db):
        """
        How the CLI runs a single-ticker cycle, and how a replay eval uses a
        frozen watchlist without touching the database.
        """
        add_ticker("AAPL")
        injected = [{"ticker": "NVDA", "company_name": "NVIDIA", "warmed_up": False}]

        result = load_watchlist_node({"cycle_id": "c1", "watchlist": injected})
        assert [item["ticker"] for item in result["watchlist"]] == ["NVDA"]

    def test_an_empty_watchlist_produces_empty_state_not_an_error(self, temp_db):
        result = load_watchlist_node({"cycle_id": "c1"})
        assert result == {"watchlist": [], "last_checked": {}}

    def test_watermarks_are_loaded_for_the_watchlist(self, temp_db):
        from src.persistence.repository import set_checkpoint

        add_ticker("AAPL")
        set_checkpoint("AAPL", "filing", when=NOW)

        result = load_watchlist_node({"cycle_id": "c1"})
        assert result["last_checked"]["AAPL:filing"].startswith("2026-08-03")
