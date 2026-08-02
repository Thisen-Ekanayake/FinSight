# ═══════════════════════════════════════════════════════
# FinSight — Tests: Prices & Technical Indicators
# ═══════════════════════════════════════════════════════
# Offline: indicators are computed from synthetic bars.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from src.core.errors import DataSourceError
from src.data.prices import MIN_BARS_FOR_INDICATORS, compute_indicators
from src.data.schemas import PriceBar


def _bars(closes: list[float], *, volume: float = 1_000_000.0) -> list[PriceBar]:
    """Build a synthetic oldest-first price history from closing prices."""
    start = date(2024, 1, 1)
    return [
        PriceBar(
            date=(start + timedelta(days=i)).isoformat(),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=volume,
        )
        for i, close in enumerate(closes)
    ]


class TestComputeIndicators:
    def test_rejects_insufficient_history(self):
        with pytest.raises(DataSourceError, match="need"):
            compute_indicators(_bars([100.0] * 5), "TEST")

    def test_minimum_history_is_enough(self):
        assert compute_indicators(_bars([100.0] * MIN_BARS_FOR_INDICATORS), "TEST")

    def test_last_close_and_as_of_come_from_the_final_bar(self):
        bars = _bars([100.0] * 24 + [123.45])
        ind = compute_indicators(bars, "TEST")
        assert ind["last_close"] == pytest.approx(123.45)
        assert ind["as_of"] == bars[-1]["date"]

    def test_ticker_is_uppercased(self):
        assert compute_indicators(_bars([100.0] * 25), "aapl")["ticker"] == "AAPL"

    def test_one_day_change_is_correct(self):
        ind = compute_indicators(_bars([100.0] * 24 + [110.0]), "TEST")
        assert ind["change_pct_1d"] == pytest.approx(10.0)

    def test_five_day_change_is_correct(self):
        ind = compute_indicators(_bars([100.0] * 20 + [100, 100, 100, 100, 200.0]), "TEST")
        assert ind["change_pct_5d"] == pytest.approx(100.0)

    def test_moving_average_of_a_flat_series_equals_the_price(self):
        ind = compute_indicators(_bars([50.0] * 60), "TEST")
        assert ind["ma_20"] == pytest.approx(50.0)
        assert ind["ma_50"] == pytest.approx(50.0)

    def test_long_window_indicators_are_none_when_history_is_short(self):
        # Better an explicit None than a confident number computed from a
        # window too short to mean anything.
        ind = compute_indicators(_bars([100.0] * 30), "TEST")
        assert ind["ma_50"] is None
        assert ind["ma_200"] is None

    def test_long_window_indicators_populate_with_enough_history(self):
        ind = compute_indicators(_bars([100.0 + i * 0.1 for i in range(250)]), "TEST")
        assert ind["ma_200"] is not None
        assert ind["rsi_14"] is not None
        assert ind["macd"] is not None

    def test_rsi_is_high_after_a_sustained_rally(self):
        ind = compute_indicators(_bars([100.0 + i for i in range(60)]), "TEST")
        assert ind["rsi_14"] is not None and ind["rsi_14"] > 70

    def test_rsi_is_low_after_a_sustained_selloff(self):
        ind = compute_indicators(_bars([200.0 - i for i in range(60)]), "TEST")
        assert ind["rsi_14"] is not None and ind["rsi_14"] < 30

    def test_bollinger_bands_straddle_the_price(self):
        closes = [100.0 + (5 if i % 2 else -5) for i in range(40)]
        ind = compute_indicators(_bars(closes), "TEST")
        assert ind["bb_lower"] < ind["last_close"] < ind["bb_upper"]  # type: ignore[operator]

    def test_volume_ratio_detects_a_spike(self):
        bars = _bars([100.0] * 40)
        bars[-1]["volume"] = 5_000_000.0
        ind = compute_indicators(bars, "TEST")
        assert ind["volume_ratio"] is not None and ind["volume_ratio"] > 3.0


class TestVolatilityZScore:
    """
    The z-score is what separates 'NVDA moved 4%' (normal) from 'JNJ moved 4%'
    (unusual). It drives severity in the Phase 6 price monitor.
    """

    def test_none_without_enough_history(self):
        assert compute_indicators(_bars([100.0] * 40), "TEST")["vol_zscore"] is None

    def test_large_move_after_calm_scores_high(self):
        # 80 flat days, then a 10% jump — should be many sigma out.
        closes = [100.0 + (i % 2) * 0.05 for i in range(80)] + [110.0]
        ind = compute_indicators(_bars(closes), "TEST")
        assert ind["vol_zscore"] is not None
        assert abs(ind["vol_zscore"]) > 3.0

    def test_ordinary_move_in_a_volatile_series_scores_low(self):
        closes = [100.0 + (5 if i % 2 else -5) for i in range(80)]
        ind = compute_indicators(_bars(closes), "TEST")
        assert ind["vol_zscore"] is not None
        assert abs(ind["vol_zscore"]) < 3.0

    def test_result_is_always_finite_or_none(self):
        # A zero-variance series must not yield inf/nan.
        ind = compute_indicators(_bars([100.0] * 80), "TEST")
        assert ind["vol_zscore"] is None or math.isfinite(ind["vol_zscore"])


class TestExtractSymbolFrame:
    """
    Regression guard for a real bug: yfinance returns a MultiIndex even for a
    SINGLE ticker under group_by="ticker", so assuming "one ticker means flat
    columns" silently produced zero bars. Shape must be detected, not assumed.
    """

    def _frame(self, columns):
        import pandas as pd

        return pd.DataFrame([[1.0] * len(columns)], columns=columns)

    def test_multiindex_ticker_first_single_symbol(self):
        import pandas as pd

        from src.data.prices import _extract_symbol_frame

        cols = pd.MultiIndex.from_product([["AAPL"], ["Open", "High", "Low", "Close", "Volume"]])
        assert _extract_symbol_frame(self._frame(cols), "AAPL") is not None

    def test_multiindex_ticker_first_multiple_symbols(self):
        import pandas as pd

        from src.data.prices import _extract_symbol_frame

        cols = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "Close"]])
        frame = self._frame(cols)
        assert _extract_symbol_frame(frame, "AAPL") is not None
        assert _extract_symbol_frame(frame, "MSFT") is not None

    def test_multiindex_field_first_is_also_handled(self):
        # Some yfinance versions nest (field, ticker) instead.
        import pandas as pd

        from src.data.prices import _extract_symbol_frame

        cols = pd.MultiIndex.from_product([["Open", "Close"], ["AAPL", "MSFT"]])
        assert _extract_symbol_frame(self._frame(cols), "AAPL") is not None

    def test_flat_ohlcv_columns_are_accepted(self):
        from src.data.prices import _extract_symbol_frame

        frame = self._frame(["Open", "High", "Low", "Close", "Volume"])
        assert _extract_symbol_frame(frame, "AAPL") is not None

    def test_absent_symbol_returns_none(self):
        import pandas as pd

        from src.data.prices import _extract_symbol_frame

        cols = pd.MultiIndex.from_product([["AAPL"], ["Open", "Close"]])
        assert _extract_symbol_frame(self._frame(cols), "TSLA") is None

    def test_unrecognised_flat_frame_returns_none(self):
        from src.data.prices import _extract_symbol_frame

        assert _extract_symbol_frame(self._frame(["foo", "bar"]), "AAPL") is None


@pytest.mark.integration
class TestLivePrices:
    """Against live yfinance. This is the canary for Yahoo endpoint changes."""

    def test_batched_download_returns_all_tickers(self):
        from src.data.prices import get_prices

        data = get_prices(["AAPL", "MSFT"], period="6mo")
        assert set(data) == {"AAPL", "MSFT"}
        assert all(len(bars) > 100 for bars in data.values())

    def test_indicators_are_plausible(self):
        from src.data.prices import get_indicators

        ind = get_indicators(["AAPL"], period="2y")["AAPL"]
        assert ind["last_close"] > 0
        assert 0 <= ind["rsi_14"] <= 100  # type: ignore[operator]
