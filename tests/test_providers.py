# ═══════════════════════════════════════════════════════
# FinSight — Tests: Provider Fallback Chains
# ═══════════════════════════════════════════════════════
#
# The chain is what stops a silent fallback from producing a wrong number
# wearing a confident citation. These tests pin that behaviour.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import pytest

from src.core.errors import BudgetExhausted, DataSourceError
from src.data.providers import run_chain


class TestChainOrdering:
    def test_first_provider_wins_when_it_succeeds(self):
        result = run_chain(
            "prices",
            {"yfinance": lambda: "from-yf", "finnhub": lambda: "from-fh"},
        )
        assert result.value == "from-yf"
        assert result.provider == "yfinance"
        assert result.degraded is False

    def test_falls_through_to_the_next_provider(self):
        def broken():
            raise DataSourceError("yfinance", "Yahoo changed its endpoints again")

        result = run_chain("prices", {"yfinance": broken, "finnhub": lambda: "from-fh"})
        assert result.value == "from-fh"
        assert result.provider == "finnhub"

    def test_configured_order_is_used_not_dict_order(self):
        # dict order deliberately reversed relative to PROVIDER_CHAINS.
        result = run_chain("prices", {"finnhub": lambda: "fh", "yfinance": lambda: "yf"})
        assert result.provider == "yfinance"

    def test_fundamentals_chain_tries_edgar_first(self):
        # EDGAR is authoritative and self-citing; it must lead.
        result = run_chain(
            "fundamentals",
            {"yfinance": lambda: "yf", "edgar_xbrl": lambda: "edgar"},
        )
        assert result.provider == "edgar_xbrl"

    def test_explicit_order_overrides_configuration(self):
        result = run_chain(
            "prices",
            {"yfinance": lambda: "yf", "finnhub": lambda: "fh"},
            order=["finnhub", "yfinance"],
        )
        assert result.provider == "finnhub"

    def test_providers_absent_from_the_dict_are_skipped(self):
        result = run_chain("prices", {"fmp": lambda: "fmp-only"})
        assert result.provider == "fmp"


class TestDegradedFlag:
    """A degraded result means the most-trusted source was unavailable."""

    def test_not_degraded_on_first_choice(self):
        assert run_chain("prices", {"yfinance": lambda: 1}).degraded is False

    def test_degraded_after_a_fallback(self):
        def broken():
            raise DataSourceError("yfinance", "down")

        assert run_chain("prices", {"yfinance": broken, "finnhub": lambda: 2}).degraded is True


class TestFailureHandling:
    def test_budget_exhaustion_falls_through(self):
        def spent():
            raise BudgetExhausted("fmp", "daily soft limit reached")

        result = run_chain("prices", {"fmp": spent, "yfinance": lambda: "yf"})
        assert result.provider == "yfinance"

    def test_unexpected_exception_does_not_kill_the_chain(self):
        # A provider raising something unforeseen must not take down the
        # request — that is the whole point of having a chain.
        def exploding():
            raise ValueError("something entirely unexpected")

        result = run_chain("prices", {"yfinance": exploding, "finnhub": lambda: "fh"})
        assert result.provider == "finnhub"

    def test_default_is_returned_when_everything_fails(self):
        def broken():
            raise DataSourceError("x", "down")

        result = run_chain("prices", {"yfinance": broken, "finnhub": broken}, default=[])
        assert result.value == []
        assert result.provider == "none"
        assert result.ok is False

    def test_raises_when_all_fail_and_no_default(self):
        def broken():
            raise DataSourceError("x", "down")

        with pytest.raises(DataSourceError, match="all providers failed"):
            run_chain("prices", {"yfinance": broken, "finnhub": broken})

    def test_raises_when_no_provider_is_usable(self):
        with pytest.raises(DataSourceError, match="no usable providers"):
            run_chain("prices", {"not_in_any_chain": lambda: 1}, order=["yfinance"])


class TestAuditTrail:
    """Every attempt is recorded — this is what the audit trail is built from."""

    def test_successful_first_attempt_is_recorded(self):
        result = run_chain("prices", {"yfinance": lambda: 1})
        assert len(result.attempts) == 1
        assert result.attempts[0].ok is True

    def test_failures_are_recorded_with_their_error(self):
        def broken():
            raise DataSourceError("yfinance", "endpoint moved")

        result = run_chain("prices", {"yfinance": broken, "finnhub": lambda: 2})
        assert len(result.attempts) == 2
        assert result.attempts[0].ok is False
        assert "endpoint moved" in (result.attempts[0].error or "")
        assert result.attempts[1].ok is True

    def test_attempts_preserve_chain_order(self):
        def broken():
            raise DataSourceError("x", "down")

        result = run_chain("prices", {"yfinance": broken, "finnhub": broken, "fmp": lambda: 3})
        assert [a.provider for a in result.attempts] == ["yfinance", "finnhub", "fmp"]

    def test_latency_is_captured(self):
        result = run_chain("prices", {"yfinance": lambda: 1})
        assert result.attempts[0].latency_ms >= 0
