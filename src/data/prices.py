# ═══════════════════════════════════════════════════════
# FinSight — Price History & Technical Indicators
# ═══════════════════════════════════════════════════════
#
# Purpose : OHLCV history and derived indicators. This is the ONLY module that
#           may import yfinance — everything else goes through get_prices(),
#           so when Yahoo changes its endpoints (roughly twice a year) the
#           blast radius is one file.
#
# Public API:
#   get_prices(tickers, ...)        BATCHED download -> {ticker: [PriceBar]}
#   compute_indicators(bars, ...)   RSI/MACD/MA/Bollinger/volume/vol z-score
#   get_indicators(tickers, ...)    both, in one call
#
# Batching note:
#   get_prices takes a LIST and issues ONE yfinance call for all symbols.
#   A 10-ticker monitoring cycle costs 1 request here, not 10. That batching
#   is why the cycle budget is ~26 calls instead of ~250.
#
# Usage:
#   python -m src.data.prices --ticker AAPL --ticker MSFT
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
from typing import Any

from src.core.errors import DataSourceError
from src.data.rate_limit import guard
from src.data.schemas import IndicatorSet, PriceBar

logger = logging.getLogger(__name__)

PROVIDER = "yfinance"

# Enough history for a 200-day moving average plus warm-up.
DEFAULT_PERIOD = "2y"
MIN_BARS_FOR_INDICATORS = 20


def _bars_from_frame(frame: Any) -> list[PriceBar]:
    """Convert a per-ticker OHLCV DataFrame into PriceBar records."""
    import pandas as pd

    bars: list[PriceBar] = []
    for index, row in frame.iterrows():
        close = row.get("Close")
        if close is None or pd.isna(close):
            continue
        bars.append(
            PriceBar(
                date=index.strftime("%Y-%m-%d") if hasattr(index, "strftime") else str(index),
                open=float(row.get("Open", close) or close),
                high=float(row.get("High", close) or close),
                low=float(row.get("Low", close) or close),
                close=float(close),
                volume=float(row.get("Volume", 0) or 0),
            )
        )
    return bars


def _extract_symbol_frame(raw: Any, symbol: str) -> Any | None:
    """
    Pull one symbol's OHLCV columns out of a yfinance download.

    Shape-detecting rather than count-assuming, because yfinance is not
    consistent about it: with ``group_by="ticker"`` it returns a MultiIndex
    even for a SINGLE symbol, and that behaviour has changed across versions.
    Assuming "one ticker means flat columns" silently yields zero bars.

    Returns
    -------
    DataFrame or None
        None when the symbol is absent from the response.
    """
    import pandas as pd

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if symbol in level0:
            return raw[symbol]
        # Some versions nest the other way round: (field, ticker).
        level1 = set(raw.columns.get_level_values(1))
        if symbol in level1:
            return raw.xs(symbol, axis=1, level=1)
        return None

    # Flat columns: the whole frame is this symbol, provided it looks like OHLCV.
    return raw if "Close" in raw.columns else None


def get_prices(
    tickers: list[str],
    *,
    period: str = DEFAULT_PERIOD,
    interval: str = "1d",
) -> dict[str, list[PriceBar]]:
    """
    Download OHLCV history for several tickers in ONE request.

    Parameters
    ----------
    tickers : list of str
        Symbols. Passing a list is the point — do not loop over this function.
    period : str, default "2y"
        yfinance period string. Two years covers a 200-day MA with warm-up.
    interval : str, default "1d"
        Bar interval.

    Returns
    -------
    dict
        ``{ticker: [PriceBar, ...]}`` oldest-first. Tickers that returned no
        data are omitted rather than mapped to an empty list.

    Raises
    ------
    DataSourceError
        If the download fails outright.
    """
    if not tickers:
        return {}

    import yfinance as yf

    symbols = [t.upper() for t in tickers]
    guard(PROVIDER)

    try:
        raw = yf.download(
            tickers=" ".join(symbols),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:  # yfinance raises a menagerie of exception types
        raise DataSourceError(PROVIDER, f"download failed for {symbols}: {exc}") from exc

    if raw is None or raw.empty:
        raise DataSourceError(PROVIDER, f"no data returned for {symbols}")

    results: dict[str, list[PriceBar]] = {}
    for symbol in symbols:
        frame = _extract_symbol_frame(raw, symbol)
        if frame is None:
            logger.warning("yfinance returned no frame for %s", symbol)
            continue
        bars = _bars_from_frame(frame.dropna(how="all"))
        if bars:
            results[symbol] = bars

    logger.info("Prices: %d/%d tickers in one batched call", len(results), len(symbols))
    return results


def compute_indicators(bars: list[PriceBar], ticker: str) -> IndicatorSet:
    """
    Compute technical indicators from a price history.

    Indicators requiring more history than is available come back as None
    rather than being computed from a short window, which would produce
    confident-looking but meaningless numbers.

    Parameters
    ----------
    bars : list of PriceBar
        Oldest-first history. At least 20 bars.
    ticker : str
        Symbol, for the returned record.

    Returns
    -------
    IndicatorSet

    Raises
    ------
    DataSourceError
        If there is too little history to compute anything.
    """
    import numpy as np
    import pandas as pd
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands

    if len(bars) < MIN_BARS_FOR_INDICATORS:
        raise DataSourceError(PROVIDER, f"{ticker}: only {len(bars)} bars, need {MIN_BARS_FOR_INDICATORS}")

    frame = pd.DataFrame(bars)
    close = frame["close"]
    volume = frame["volume"]
    n = len(frame)

    def at(series: Any, minimum: int) -> float | None:
        """Last value of a series, or None if history is too short."""
        if n < minimum:
            return None
        value = series.iloc[-1]
        return None if pd.isna(value) else float(value)

    def change_over(periods: int) -> float:
        if n <= periods:
            return 0.0
        past = close.iloc[-1 - periods]
        return 0.0 if past == 0 else float((close.iloc[-1] - past) / past * 100.0)

    macd = MACD(close) if n >= 26 else None
    bollinger = BollingerBands(close) if n >= 20 else None

    # Daily-return z-score against 60-day realised volatility. This is what
    # separates "NVDA moved 4%" (normal) from "JNJ moved 4%" (unusual), and it
    # drives severity in the Phase 6 price monitor.
    vol_z: float | None = None
    if n >= 61:
        returns = close.pct_change().dropna()
        window = returns.iloc[-60:]
        sigma = float(window.std())
        if sigma > 0:
            vol_z = float((returns.iloc[-1] - window.mean()) / sigma)

    avg_volume_20 = at(volume.rolling(20).mean(), 20)
    last_volume = float(volume.iloc[-1])

    return IndicatorSet(
        ticker=ticker.upper(),
        as_of=frame["date"].iloc[-1],
        last_close=float(close.iloc[-1]),
        change_pct_1d=change_over(1),
        change_pct_5d=change_over(5),
        change_pct_20d=change_over(20),
        rsi_14=at(RSIIndicator(close).rsi(), 15) if n >= 15 else None,
        macd=at(macd.macd(), 26) if macd else None,
        macd_signal=at(macd.macd_signal(), 26) if macd else None,
        ma_20=at(close.rolling(20).mean(), 20),
        ma_50=at(close.rolling(50).mean(), 50),
        ma_200=at(close.rolling(200).mean(), 200),
        bb_upper=at(bollinger.bollinger_hband(), 20) if bollinger else None,
        bb_lower=at(bollinger.bollinger_lband(), 20) if bollinger else None,
        volume=last_volume,
        avg_volume_20=avg_volume_20,
        volume_ratio=(last_volume / avg_volume_20 if avg_volume_20 else None),
        vol_zscore=vol_z if vol_z is None or np.isfinite(vol_z) else None,
    )


def get_indicators(tickers: list[str], *, period: str = DEFAULT_PERIOD) -> dict[str, IndicatorSet]:
    """
    Fetch prices and compute indicators for several tickers in one batch.

    Returns
    -------
    dict
        ``{ticker: IndicatorSet}``. Tickers with insufficient history are
        logged and omitted rather than failing the batch.
    """
    price_data = get_prices(tickers, period=period)

    results: dict[str, IndicatorSet] = {}
    for ticker, bars in price_data.items():
        try:
            results[ticker] = compute_indicators(bars, ticker)
        except DataSourceError as exc:
            logger.warning("Indicators: skipping %s — %s", ticker, exc)
    return results


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Print indicators for one or more tickers."""
    parser = argparse.ArgumentParser(description="Fetch prices and technical indicators")
    parser.add_argument("--ticker", action="append", required=True, help="symbol (repeatable)")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    for ticker, ind in get_indicators(args.ticker, period=args.period).items():
        print(f"\n{ticker} — as of {ind['as_of']}")
        print(f"  close        {ind['last_close']:>12,.2f}")
        print(f"  1d / 5d /20d {ind['change_pct_1d']:>+7.2f}% {ind['change_pct_5d']:>+7.2f}% " f"{ind['change_pct_20d']:>+7.2f}%")
        for label, key in (("RSI(14)", "rsi_14"), ("MACD", "macd"), ("MA20", "ma_20"), ("MA50", "ma_50"), ("MA200", "ma_200")):
            value = ind[key]  # type: ignore[literal-required]
            print(f"  {label:12s} {value:>12,.2f}" if value is not None else f"  {label:12s} {'n/a':>12s}")
        ratio = ind["volume_ratio"]
        zscore = ind["vol_zscore"]
        print(f"  vol ratio    {ratio:>12,.2f}" if ratio else f"  vol ratio    {'n/a':>12s}")
        print(f"  vol z-score  {zscore:>12,.2f}" if zscore is not None else f"  vol z-score  {'n/a':>12s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
