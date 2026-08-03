# ═══════════════════════════════════════════════════════
# FinSight — Price Monitor
# ═══════════════════════════════════════════════════════
#
# Purpose : Notice unusual single-day price moves across the whole watchlist.
#
# Public API:
#   price_monitor_node(payload)
#   price_natural_key(ticker, as_of, change_pct)
#
# ══ BATCHED: ONE BRANCH FOR EVERY TICKER ══
#   yfinance takes a list of symbols and returns them in one request, so this
#   monitor receives the entire watchlist in a single Send() rather than one
#   per ticker. Ten tickers cost ONE call here and ten at the filing monitor —
#   which is why a ten-ticker cycle is 26 external calls and not 40.
#
#   The asymmetry lives in graph.py, where it is visible as topology.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time

from src.core.schemas import make_citation
from src.data.schemas import IndicatorSet
from src.monitor.config import PRICE_MIN_PCT, PRICE_NOTABLE_VOLUME_RATIO
from src.monitor.monitors._common import bucket, candidate, monitor
from src.monitor.state import CandidateAlert

logger = logging.getLogger(__name__)

MONITOR_NAME = "price_monitor"

# Band width for the magnitude component of the natural key, in percentage
# points. One point is coarse enough that an intraday re-check collides with
# the morning's reading, and fine enough that a 3% day and an 8% day never do.
PRICE_BUCKET_PCT: float = 1.0

# Six months of daily bars — enough for a 60-day volatility estimate and the
# 200-day moving average, without pulling years nobody reads.
PRICE_PERIOD: str = "6mo"


def price_natural_key(ticker: str, as_of: str, change_pct: float) -> str:
    """
    Identity of a price event: which day, which direction, roughly how big.

    Rounding the magnitude is what makes this stable. The same move re-measured
    an hour later is a different percentage and the same event, so hashing the
    raw number would hand every intraday re-check a fresh identity and defeat
    the free exact-match path entirely.
    """
    direction = "down" if change_pct < 0 else "up"
    return f"{ticker.upper()}:{as_of}:{direction}:{bucket(change_pct, PRICE_BUCKET_PCT)}"


def _describe(indicators: IndicatorSet) -> str:
    """One line of human-readable detail for the alert body."""
    parts = [f"closed at {indicators['last_close']:,.2f}, {indicators['change_pct_1d']:+.2f}% on the day"]

    ratio = indicators.get("volume_ratio")
    if ratio and ratio >= PRICE_NOTABLE_VOLUME_RATIO:
        parts.append(f"on {ratio:.1f}x average volume")

    zscore = indicators.get("vol_zscore")
    if zscore is not None:
        parts.append(f"a {abs(zscore):.1f}-sigma move against 60-day realised volatility")

    ma20 = indicators.get("ma_20")
    if ma20 and indicators["last_close"] < ma20 and indicators["change_pct_1d"] < 0:
        parts.append("breaking below the 20-day moving average")

    rsi = indicators.get("rsi_14")
    if rsi is not None and (rsi <= 30 or rsi >= 70):
        parts.append(f"RSI {rsi:.0f} ({'oversold' if rsi <= 30 else 'overbought'})")

    return "; ".join(parts).capitalize() + "."


@monitor(MONITOR_NAME)
def price_monitor_node(payload: dict) -> tuple[list[CandidateAlert], list]:
    """
    Check every watched ticker's last close in one batched request.

    Emits a candidate only when the one-day move clears PRICE_MIN_PCT. A 1%
    day is not an event, and emitting it would put the dedup engine — and an
    embedding — to work on noise.

    Returns
    -------
    tuple
        ``(candidates, api_calls)``.
    """
    from src.data.prices import get_indicators
    from src.research.agents._common import tool_record

    tickers = [t.upper() for t in payload.get("tickers") or []]
    companies = payload.get("companies") or {}
    if not tickers:
        return [], []

    started = time.monotonic()
    indicator_sets = get_indicators(tickers, period=PRICE_PERIOD)
    calls = [
        tool_record(
            MONITOR_NAME,
            "get_indicators",
            args={"tickers": tickers, "period": PRICE_PERIOD},
            provider="prices",
            latency_ms=int((time.monotonic() - started) * 1000),
            ok=bool(indicator_sets),
        )
    ]

    missing = sorted(set(tickers) - set(indicator_sets))
    if missing:
        # Not an error: a ticker with too little history is skipped by
        # get_indicators by design. Worth saying out loud, because "no alert"
        # and "no data" look identical from the outside.
        logger.info("%s: no indicators for %s", MONITOR_NAME, ", ".join(missing))

    candidates: list[CandidateAlert] = []

    for ticker, indicators in indicator_sets.items():
        change = float(indicators["change_pct_1d"])
        if abs(change) < PRICE_MIN_PCT:
            continue

        as_of = indicators["as_of"]
        citation = make_citation("YFINANCE", f"{ticker}@{as_of}", as_of=as_of, ticker=ticker)

        candidates.append(
            candidate(
                ticker,
                "PRICE_MOVE",
                monitor_name=MONITOR_NAME,
                company_name=companies.get(ticker, ""),
                headline=f"{ticker} {'fell' if change < 0 else 'rose'} {abs(change):.1f}%",
                detail=_describe(indicators),
                natural_key=price_natural_key(ticker, as_of, change),
                metrics={
                    "last_close": indicators["last_close"],
                    "change_pct_1d": change,
                    "vol_zscore": indicators.get("vol_zscore"),
                    "volume_ratio": indicators.get("volume_ratio"),
                    "rsi_14": indicators.get("rsi_14"),
                    "ma_20": indicators.get("ma_20"),
                    "as_of": as_of,
                },
                evidence=[citation],
                observed_at=as_of,
            )
        )

    return candidates, calls
