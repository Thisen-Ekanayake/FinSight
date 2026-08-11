# ═══════════════════════════════════════════════════════
# FinSight — Tests: Specialist Agents
# ═══════════════════════════════════════════════════════
#
# Selection logic and the failure contract are tested offline. Live data
# access is marked integration.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.data.schemas import IndicatorSet
from src.research.agents import AGENT_NODES
from src.research.agents._common import specialist
from src.research.agents.filings_rag import select_sections
from src.research.agents.fundamentals import select_metrics
from src.research.agents.macro import select_series
from src.research.agents.technical import RSI_OVERBOUGHT, RSI_OVERSOLD, interpret
from src.research.state import AGENT_NAMES


class TestAgentRegistry:
    def test_every_routed_agent_has_a_node(self):
        # A router selecting an agent with no node would Send() into nothing.
        assert set(AGENT_NODES) == set(AGENT_NAMES)

    def test_nodes_are_callable(self):
        assert all(callable(node) for node in AGENT_NODES.values())


class TestFailureContract:
    """
    Branches run concurrently in ONE superstep. An unhandled exception would
    abort the superstep and discard the other branches' work, so failures must
    become data rather than propagate.
    """

    def test_exception_becomes_an_error_entry(self):
        @specialist("boom")
        def failing(payload, ticker):
            raise RuntimeError("provider exploded")

        result = failing({"ticker": "AAPL"})
        assert result["errors"]
        assert "provider exploded" in result["errors"][0]

    def test_failure_still_returns_every_fan_in_key(self):
        # A partial state missing a reducer key would break the merge.
        @specialist("boom")
        def failing(payload, ticker):
            raise ValueError("nope")

        result = failing({"ticker": "AAPL"})
        assert set(result) == {"findings", "citations", "errors", "tool_calls"}

    def test_failure_records_a_failed_tool_call(self):
        @specialist("boom")
        def failing(payload, ticker):
            raise ValueError("nope")

        assert failing({"ticker": "AAPL"})["tool_calls"][0]["ok"] is False

    def test_success_returns_no_errors(self):
        @specialist("ok")
        def working(payload, ticker):
            return [], [], []

        assert working({"ticker": "AAPL"})["errors"] == []


class TestFundamentalsMetricSelection:
    """Keyword lookup, not an LLM call — this runs in every fan-out branch."""

    def test_revenue_question_selects_revenue(self):
        assert "revenue" in select_metrics("what was total revenue?")

    def test_margin_question_pulls_both_numerator_and_denominator(self):
        # A margin cannot be computed from gross profit alone.
        metrics = select_metrics("how did gross margin trend?")
        assert "gross_profit" in metrics
        assert "revenue" in metrics

    def test_balance_sheet_question_selects_balance_items(self):
        metrics = select_metrics("what do the assets look like?")
        assert "total_assets" in metrics

    def test_unmatched_question_falls_back_to_defaults(self):
        assert select_metrics("tell me about the company") != []

    def test_never_returns_empty(self):
        assert select_metrics("") != []

    def test_only_returns_known_metrics(self):
        from src.data.fundamentals import METRIC_CONCEPTS

        for question in ("revenue", "margin", "cash flow", "debt", ""):
            assert set(select_metrics(question)) <= set(METRIC_CONCEPTS)


class TestFilingsSectionSelection:
    def test_risk_question_targets_item_1a(self):
        assert select_sections("what risks does the company face?") == ["1A"]

    def test_legal_question_targets_item_3(self):
        assert select_sections("any ongoing litigation?") == ["3"]

    def test_mdna_question_includes_both_form_numberings(self):
        # MD&A is Item 7 in a 10-K but Item 2 in a 10-Q, and the collection
        # holds both.
        sections = select_sections("what did management say about performance?")
        assert "7" in sections
        assert "2" in sections

    def test_vague_question_returns_none_to_search_everything(self):
        assert select_sections("tell me something interesting") is None


class TestMacroSeriesSelection:
    def test_inflation_question_selects_cpi(self):
        assert "CPIAUCSL" in select_series("what is inflation doing?")

    def test_rate_question_selects_fed_funds(self):
        assert "DFF" in select_series("where is the fed funds rate?")

    def test_yield_curve_question_selects_the_spread(self):
        assert "T10Y2Y" in select_series("is the yield curve inverted?")

    def test_vague_question_falls_back_to_a_broad_set(self):
        assert len(select_series("how is the economy?")) >= 3

    def test_never_returns_empty(self):
        assert select_series("") != []


class TestTechnicalInterpretation:
    """Deterministic rules — RSI 74 is overbought by definition, not opinion."""

    def _indicators(self, **overrides) -> IndicatorSet:
        base = dict(
            ticker="TEST",
            as_of="2026-01-01",
            last_close=100.0,
            change_pct_1d=0.0,
            change_pct_5d=0.0,
            change_pct_20d=0.0,
            rsi_14=50.0,
            macd=0.0,
            macd_signal=0.0,
            ma_20=100.0,
            ma_50=100.0,
            ma_200=100.0,
            bb_upper=110.0,
            bb_lower=90.0,
            volume=1e6,
            avg_volume_20=1e6,
            volume_ratio=1.0,
            vol_zscore=0.0,
        )
        base.update(overrides)
        return IndicatorSet(**base)  # type: ignore[typeddict-item]

    def test_high_rsi_reads_overbought(self):
        readings = interpret(self._indicators(rsi_14=RSI_OVERBOUGHT + 5))
        assert any("overbought" in r for r in readings)

    def test_low_rsi_reads_oversold(self):
        readings = interpret(self._indicators(rsi_14=RSI_OVERSOLD - 5))
        assert any("oversold" in r for r in readings)

    def test_mid_rsi_reads_neutral(self):
        assert any("neutral" in r for r in interpret(self._indicators(rsi_14=50.0)))

    def test_golden_cross_reads_uptrend(self):
        readings = interpret(self._indicators(ma_50=120.0, ma_200=100.0))
        assert any("uptrend" in r for r in readings)

    def test_death_cross_reads_downtrend(self):
        readings = interpret(self._indicators(ma_50=90.0, ma_200=100.0))
        assert any("downtrend" in r for r in readings)

    def test_large_zscore_is_called_unusual(self):
        readings = interpret(self._indicators(vol_zscore=-3.7))
        assert any("sigma" in r and "unusual" in r for r in readings)

    def test_ordinary_zscore_is_not_reported(self):
        # Manufacturing significance from an unremarkable move is worse than
        # saying nothing.
        readings = interpret(self._indicators(vol_zscore=0.4))
        assert not any("sigma" in r for r in readings)

    def test_volume_spike_is_reported(self):
        readings = interpret(self._indicators(volume_ratio=3.1))
        assert any("volume" in r.lower() for r in readings)

    def test_missing_indicators_are_skipped_not_guessed(self):
        readings = interpret(self._indicators(rsi_14=None, ma_50=None, ma_200=None, vol_zscore=None))
        assert not any("RSI" in r for r in readings)

    def test_readings_are_deterministic(self):
        ind = self._indicators(rsi_14=75.0)
        assert interpret(ind) == interpret(ind)


class TestFundamentalsNode:
    def test_empty_result_yields_no_findings(self):
        with patch("src.research.agents.fundamentals.get_fundamentals_history", return_value={}):
            result = AGENT_NODES["fundamentals"]({"ticker": "AAPL", "sub_question": "revenue?"})
        assert result["findings"] == []
        assert result["tool_calls"][0]["ok"] is False

    def test_finding_value_is_the_raw_number(self):
        # The citation verifier matches answer numbers against this, so a
        # formatted string here would break grounding.
        fake = {
            "revenue": [
                {
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "value": 416161000000.0,
                    "unit": "USD",
                    "period": "2025 FY",
                    "as_of": "2025-09-27",
                    "provider": "edgar_xbrl",
                    "source_id": "0000320193-25-000079",
                    "source_url": "https://www.sec.gov/x",
                }
            ]
        }
        with patch("src.research.agents.fundamentals.get_fundamentals_history", return_value=fake):
            result = AGENT_NODES["fundamentals"]({"ticker": "AAPL", "sub_question": "revenue?"})

        assert result["findings"][0]["value"] == 416161000000.0
        assert result["findings"][0]["citations"][0]["source_id"] == "0000320193-25-000079"

    def test_a_per_share_figure_keeps_its_decimals(self):
        # Found by the Phase 5 eval. ",.0f" rendered EPS of 20.02 as "20", and
        # because the claim TEXT is itself evidence for the citation verifier,
        # the answer then grounded "20 USD" perfectly while being wrong by two
        # cents a share. Every citation check passed on a wrong number.
        fake = {
            "eps_diluted": [
                {
                    "ticker": "JPM",
                    "metric": "eps_diluted",
                    "value": 20.02,
                    "unit": "USD/shares",
                    "period": "2025 FY",
                    "as_of": "2025-12-31",
                    "provider": "edgar_xbrl",
                    "source_id": "0001628280-26-008131",
                    "source_url": "https://www.sec.gov/x",
                }
            ]
        }
        with patch("src.research.agents.fundamentals.get_fundamentals_history", return_value=fake):
            result = AGENT_NODES["fundamentals"]({"ticker": "JPM", "sub_question": "eps?"})

        assert "20.02" in result["findings"][0]["claim"]

    def test_a_large_figure_is_still_rendered_without_decimals(self):
        from src.research.agents.fundamentals import format_value

        assert format_value(416161000000.0) == "416,161,000,000"
        assert format_value(20.02) == "20.02"
        assert format_value(-3.5) == "-3.50"

    def test_fallback_provider_lowers_confidence(self):
        fake = {
            "revenue": [
                {
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "value": 1.0,
                    "unit": "USD",
                    "period": "TTM",
                    "as_of": "2026-01-01",
                    "provider": "yfinance",
                    "source_id": "AAPL@2026-01-01",
                    "source_url": "https://finance.yahoo.com/quote/AAPL",
                }
            ]
        }
        with patch("src.research.agents.fundamentals.get_fundamentals_history", return_value=fake):
            result = AGENT_NODES["fundamentals"]({"ticker": "AAPL", "sub_question": "revenue?"})

        assert result["findings"][0]["confidence"] < 1.0


@pytest.mark.integration
@pytest.mark.slow
class TestSpecialistsAgainstLiveData:
    """Real APIs and the real Qdrant collection."""

    def test_fundamentals_returns_cited_figures(self):
        result = AGENT_NODES["fundamentals"]({"ticker": "AAPL", "sub_question": "what was revenue?"})
        assert result["findings"]
        assert all(f["citations"] for f in result["findings"])

    def test_filings_rag_returns_cited_narrative(self):
        result = AGENT_NODES["filings_rag"]({"ticker": "AAPL", "sub_question": "what supply chain risks were flagged?"})
        if not result["findings"]:
            pytest.skip("no filings ingested; run ./shell_scripts/run_ingest.sh")
        assert all(f["citations"][0]["source_id"] for f in result["findings"])

    def test_macro_ignores_the_ticker(self):
        result = AGENT_NODES["macro"]({"ticker": "AAPL", "sub_question": "what is the fed funds rate?"})
        assert result["findings"]
        assert all(f["ticker"] is None for f in result["findings"])

    def test_technical_returns_price_and_readings(self):
        result = AGENT_NODES["technical"]({"ticker": "AAPL", "sub_question": "overbought?"})
        assert len(result["findings"]) >= 2
        assert result["findings"][0]["value"] > 0
