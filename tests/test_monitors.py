# ═══════════════════════════════════════════════════════
# FinSight — Tests: The Four Monitors
# ═══════════════════════════════════════════════════════
#
# Data sources are mocked at the function each monitor imports, not at the
# HTTP layer: these tests are about what a monitor DOES with what it got —
# which candidates it emits, what natural key it assigns, and whether one
# failing branch can take the cycle down with it.
#
# No network, no LLM, no database.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.monitor.monitors._common import bucket
from src.monitor.monitors.filings import filing_monitor_node
from src.monitor.monitors.macro import detect_crossing, macro_monitor_node
from src.monitor.monitors.news import count_independent_sources, news_monitor_node
from src.monitor.monitors.price import price_monitor_node, price_natural_key

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def indicators(ticker="AAPL", **overrides):
    base = {
        "ticker": ticker,
        "as_of": "2026-08-03",
        "last_close": 210.11,
        "change_pct_1d": -5.2,
        "change_pct_5d": -6.0,
        "change_pct_20d": -8.0,
        "rsi_14": 28.0,
        "macd": None,
        "macd_signal": None,
        "ma_20": 220.0,
        "ma_50": None,
        "ma_200": None,
        "bb_upper": None,
        "bb_lower": None,
        "volume": 1e8,
        "avg_volume_20": 5e7,
        "volume_ratio": 2.0,
        "vol_zscore": -2.4,
    }
    base.update(overrides)
    return base


def filing(**overrides):
    base = {
        "ticker": "AAPL",
        "cik": "0000320193",
        "accession_no": "0000320193-26-000010",
        "form_type": "8-K",
        "filing_date": "2026-08-03",
        "period_of_report": None,
        "primary_document": "a.htm",
        "url": "https://sec.gov/x",
        "items": ["4.02"],
    }
    base.update(overrides)
    return base


def article(**overrides):
    base = {
        "ticker": "AAPL",
        "headline": "Apple faces a DOJ antitrust probe",
        "summary": "Regulators opened an inquiry into App Store practices.",
        "source": "reuters",
        "url": "https://www.reuters.com/tech/apple-probe",
        "published_at": "2026-08-03T09:00:00+00:00",
        "article_id": "abc123",
        "sentiment": -0.7,
    }
    base.update(overrides)
    return base


def series(series_id="DFF", values=(4.33, 4.58), **overrides):
    base = {
        "series_id": series_id,
        "title": "Federal Funds Effective Rate",
        "units": "Percent",
        "frequency": "Daily",
        "observations": [{"date": "2026-07-30", "value": values[0]}, {"date": "2026-08-01", "value": values[1]}],
        "latest_value": values[-1],
        "latest_date": "2026-08-01",
        "url": "https://fred.stlouisfed.org/series/DFF",
    }
    base.update(overrides)
    return base


class TestPriceMonitor:
    def test_a_material_move_becomes_a_candidate(self):
        with patch("src.data.prices.get_indicators", return_value={"AAPL": indicators()}):
            result = price_monitor_node({"tickers": ["AAPL"], "companies": {"AAPL": "Apple Inc."}})

        assert len(result["candidates"]) == 1
        candidate = result["candidates"][0]
        assert candidate["alert_type"] == "PRICE_MOVE"
        assert candidate["metrics"]["change_pct_1d"] == -5.2

    def test_noise_is_not_a_candidate(self):
        """
        A 1% day is not an event. Emitting it would put an embedding and a
        dedup decision to work on nothing.
        """
        with patch("src.data.prices.get_indicators", return_value={"AAPL": indicators(change_pct_1d=-1.0)}):
            result = price_monitor_node({"tickers": ["AAPL"], "companies": {}})

        assert result["candidates"] == []

    def test_the_whole_watchlist_costs_one_call(self):
        """
        The batching that makes a ten-ticker cycle 26 calls rather than 40. If
        this ever becomes one call per ticker it will still work, and quietly
        cost five times as much.
        """
        calls = []

        def record(tickers, **kwargs):
            calls.append(list(tickers))
            return {t: indicators(t) for t in tickers}

        with patch("src.data.prices.get_indicators", side_effect=record):
            price_monitor_node({"tickers": ["AAPL", "MSFT", "NVDA"], "companies": {}})

        assert calls == [["AAPL", "MSFT", "NVDA"]]

    def test_a_ticker_with_no_history_is_skipped_not_fatal(self):
        with patch("src.data.prices.get_indicators", return_value={"AAPL": indicators()}):
            result = price_monitor_node({"tickers": ["AAPL", "NEWCO"], "companies": {}})

        assert len(result["candidates"]) == 1
        assert result["monitor_errors"] == []

    def test_a_failing_fetch_does_not_advance_the_watermark(self):
        """
        A watermark advanced past an outage skips whatever was published
        during it, permanently and silently.
        """
        with patch("src.data.prices.get_indicators", side_effect=RuntimeError("yahoo is down")):
            result = price_monitor_node({"tickers": ["AAPL"], "companies": {}})

        assert result["checked"] == []
        assert len(result["monitor_errors"]) == 1
        assert "yahoo is down" in result["monitor_errors"][0]

    def test_a_successful_fetch_does_advance_it(self):
        with patch("src.data.prices.get_indicators", return_value={"AAPL": indicators()}):
            result = price_monitor_node({"tickers": ["AAPL", "MSFT"], "companies": {}})

        assert result["checked"] == ["AAPL:price", "MSFT:price"]

    def test_the_detail_mentions_volume_and_the_moving_average(self):
        with patch("src.data.prices.get_indicators", return_value={"AAPL": indicators()}):
            detail = price_monitor_node({"tickers": ["AAPL"], "companies": {}})["candidates"][0]["detail"]

        assert "average volume" in detail
        assert "20-day moving average" in detail


class TestPriceNaturalKey:
    def test_the_same_move_remeasured_keeps_its_identity(self):
        """
        The whole reason the magnitude is bucketed. Hashing the raw percentage
        would hand every intraday re-check a fresh identity and defeat the free
        exact-match path.
        """
        assert price_natural_key("AAPL", "2026-08-03", -5.2) == price_natural_key("AAPL", "2026-08-03", -5.4)

    def test_direction_is_part_of_the_identity(self):
        assert price_natural_key("AAPL", "2026-08-03", -5.2) != price_natural_key("AAPL", "2026-08-03", 5.2)

    def test_a_different_magnitude_band_is_a_different_event(self):
        assert price_natural_key("AAPL", "2026-08-03", -3.1) != price_natural_key("AAPL", "2026-08-03", -8.4)

    def test_tomorrow_is_a_new_event(self):
        assert price_natural_key("AAPL", "2026-08-03", -5.2) != price_natural_key("AAPL", "2026-08-04", -5.2)

    def test_bucket_rejects_a_zero_width(self):
        with pytest.raises(ValueError):
            bucket(5.0, 0)


class TestFilingMonitor:
    def test_a_new_filing_becomes_a_candidate_keyed_on_its_accession(self):
        with patch("src.data.edgar.get_filing_index", return_value=[filing()]):
            result = filing_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        candidate = result["candidates"][0]
        assert candidate["natural_key"] == "0000320193-26-000010"
        assert candidate["metrics"]["items"] == ["4.02"]

    def test_the_watermark_is_passed_through_to_edgar(self):
        """
        This is what makes it a monitor rather than a poller: the submissions
        arrays are newest-first, so EDGAR stops reading at the watermark.
        """
        captured = {}

        def record(ticker, **kwargs):
            captured.update(kwargs)
            return []

        with patch("src.data.edgar.get_filing_index", side_effect=record):
            filing_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {"AAPL": "2026-08-01T00:00:00+00:00"}})

        assert captured["since"].isoformat() == "2026-08-01"

    def test_the_headline_leads_with_the_most_serious_item(self):
        with patch("src.data.edgar.get_filing_index", return_value=[filing(items=["8.01", "4.02"])]):
            headline = filing_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})["candidates"][0][
                "headline"
            ]

        assert "non-reliance" in headline

    def test_item_codes_are_glossed_in_the_detail(self):
        with patch("src.data.edgar.get_filing_index", return_value=[filing(items=["5.02"])]):
            detail = filing_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})["candidates"][0]["detail"]

        assert "departure or appointment of principal officers" in detail

    def test_an_unknown_item_code_still_appears(self):
        with patch("src.data.edgar.get_filing_index", return_value=[filing(items=["9.99"])]):
            detail = filing_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})["candidates"][0]["detail"]

        assert "Item 9.99" in detail

    def test_one_failing_ticker_loses_only_that_branch(self):
        # Each ticker is its own Send, so a failure here is already isolated.
        with patch("src.data.edgar.get_filing_index", side_effect=RuntimeError("SEC 403")):
            result = filing_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        assert result["candidates"] == []
        assert result["checked"] == []


class TestNewsMonitor:
    def test_each_significant_article_is_its_own_candidate(self):
        """
        Three outlets covering one story must reach the dedup engine as three
        candidates. Collapsing them here would mean the engine never sees the
        case it exists for.
        """
        articles = [
            article(article_id="a1", url="https://reuters.com/x"),
            article(article_id="a2", url="https://bloomberg.com/y"),
            article(article_id="a3", url="https://ft.com/z"),
        ]
        with patch("src.data.news.get_company_news", return_value=articles):
            result = news_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        assert len(result["candidates"]) == 3
        assert {c["natural_key"] for c in result["candidates"]} == {"a1", "a2", "a3"}

    def test_corroboration_is_attached_to_every_candidate(self):
        articles = [
            article(article_id="a1", url="https://reuters.com/x"),
            article(article_id="a2", url="https://bloomberg.com/y"),
        ]
        with patch("src.data.news.get_company_news", return_value=articles):
            result = news_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        assert all(c["metrics"]["source_count"] == 2 for c in result["candidates"])

    def test_mild_sentiment_is_not_a_candidate(self):
        with patch("src.data.news.get_company_news", return_value=[article(sentiment=-0.05)]):
            result = news_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        assert result["candidates"] == []

    def test_unscored_articles_are_skipped(self):
        with patch("src.data.news.get_company_news", return_value=[article(sentiment=None)]):
            result = news_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        assert result["candidates"] == []

    def test_a_flood_is_capped_at_the_most_negative(self):
        from src.monitor.config import NEWS_MAX_CANDIDATES_PER_TICKER

        articles = [
            article(article_id=f"a{i}", sentiment=-0.3 - i * 0.05, url=f"https://x{i}.com/a") for i in range(12)
        ]
        with patch("src.data.news.get_company_news", return_value=articles):
            result = news_monitor_node({"tickers": ["AAPL"], "companies": {}, "since": {}})

        assert len(result["candidates"]) == NEWS_MAX_CANDIDATES_PER_TICKER
        # Most negative kept: a11 has sentiment -0.85.
        assert result["candidates"][0]["metrics"]["sentiment"] == pytest.approx(-0.85)


class TestIndependentSources:
    def test_counted_by_domain_not_by_label(self):
        """
        The same outlet arrives as "Reuters", "reuters", and "Reuters News"
        from different feeds. Three spellings of one outlet must not satisfy a
        rule that exists to require three OUTLETS.
        """
        items = [
            {"url": "https://www.reuters.com/a", "source": "Reuters"},
            {"url": "https://reuters.com/b", "source": "reuters"},
            {"url": "https://REUTERS.com/c", "source": "Reuters News"},
        ]
        assert count_independent_sources(items) == 1

    def test_distinct_domains_count_separately(self):
        items = [
            {"url": "https://reuters.com/a", "source": "Reuters"},
            {"url": "https://bloomberg.com/b", "source": "Bloomberg"},
        ]
        assert count_independent_sources(items) == 2

    def test_a_missing_url_falls_back_to_the_label(self):
        assert count_independent_sources([{"url": "", "source": "Reuters"}]) == 1

    def test_an_empty_batch_is_zero(self):
        assert count_independent_sources([]) == 0


class TestMacroMonitor:
    def test_a_material_move_becomes_a_candidate(self):
        with patch("src.data.fred.get_series", side_effect=lambda sid, **kw: series(sid, (4.33, 4.58))):
            result = macro_monitor_node({"tickers": []})

        dff = [c for c in result["candidates"] if c["metrics"]["series_id"] == "DFF"]
        assert dff
        assert dff[0]["ticker"] == ""  # economy-wide, and that is the correct scope

    def test_a_routine_reading_is_not_a_candidate(self):
        with patch("src.data.fred.get_series", side_effect=lambda sid, **kw: series(sid, (4.33, 4.34))):
            result = macro_monitor_node({"tickers": []})

        assert [c for c in result["candidates"] if c["metrics"]["series_id"] == "DFF"] == []

    def test_the_natural_key_is_series_plus_release_date(self):
        with patch("src.data.fred.get_series", side_effect=lambda sid, **kw: series(sid, (4.33, 4.58))):
            result = macro_monitor_node({"tickers": []})

        dff = next(c for c in result["candidates"] if c["metrics"]["series_id"] == "DFF")
        assert dff["natural_key"] == "DFF:2026-08-01"

    def test_one_failing_series_does_not_lose_the_others(self):
        def flaky(series_id, **kwargs):
            if series_id == "DFF":
                raise RuntimeError("FRED timeout")
            return series(series_id, (4.33, 4.58))

        with patch("src.data.fred.get_series", side_effect=flaky):
            result = macro_monitor_node({"tickers": []})

        assert result["monitor_errors"] == []  # handled inside, not a branch failure
        assert any(c["metrics"]["series_id"] != "DFF" for c in result["candidates"])
        assert any(not call["ok"] for call in result["api_calls"])

    def test_a_series_with_one_observation_is_skipped(self):
        thin = series("DFF")
        thin["observations"] = [{"date": "2026-08-01", "value": 4.5}]

        with patch("src.data.fred.get_series", side_effect=lambda sid, **kw: thin):
            result = macro_monitor_node({"tickers": []})

        assert result["candidates"] == []


class TestCrossingDetection:
    def test_a_transition_is_reported(self):
        assert detect_crossing("T10Y2Y", 0.02, -0.01) == "below 0.0"
        assert detect_crossing("T10Y2Y", -0.02, 0.01) == "above 0.0"

    def test_staying_below_is_not_a_crossing(self):
        """
        "The curve is inverted" is a STATE. Alerting on it would fire every
        cycle for however many months it persists.
        """
        assert detect_crossing("T10Y2Y", -0.20, -0.25) == ""

    def test_staying_above_is_not_a_crossing(self):
        assert detect_crossing("T10Y2Y", 0.20, 0.25) == ""

    def test_series_without_a_watched_level_never_cross(self):
        assert detect_crossing("CPIAUCSL", 300.0, -1.0) == ""
