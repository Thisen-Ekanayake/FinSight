# ═══════════════════════════════════════════════════════
# FinSight — Technical Specialist
# ═══════════════════════════════════════════════════════
#
# Purpose : Price action and technical indicators.
#
# Public API:
#   technical_node(payload)
#   interpret(indicators)
#
# Interpretation is RULES, not an LLM call. "RSI 74" means overbought by
# definition, not by opinion — so the thresholds live in code where they are
# testable, and the model cannot invent a different reading of the same number.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time
from datetime import date

from src.core.schemas import make_citation
from src.data.prices import get_indicators
from src.data.schemas import IndicatorSet
from src.research.agents._common import finding, specialist, tool_record

logger = logging.getLogger(__name__)

AGENT = "technical"

# Conventional interpretation thresholds.
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
VOLUME_SPIKE_RATIO = 2.0
VOL_ZSCORE_UNUSUAL = 2.0


def interpret(ind: IndicatorSet) -> list[str]:
    """
    Turn indicator values into plain-language readings.

    Deterministic by design: the same numbers always produce the same
    statements, which makes them testable and stops the synthesizer from
    inventing its own reading of an RSI value.

    Parameters
    ----------
    ind : IndicatorSet
        Computed indicators for one ticker.

    Returns
    -------
    list of str
        Readings, most significant first. May be empty for an unremarkable
        chart — saying nothing is better than manufacturing significance.
    """
    readings: list[str] = []
    rsi = ind["rsi_14"]

    if rsi is not None:
        if rsi >= RSI_OVERBOUGHT:
            readings.append(f"RSI(14) at {rsi:.1f} is in overbought territory (above {RSI_OVERBOUGHT:.0f})")
        elif rsi <= RSI_OVERSOLD:
            readings.append(f"RSI(14) at {rsi:.1f} is in oversold territory (below {RSI_OVERSOLD:.0f})")
        else:
            readings.append(f"RSI(14) at {rsi:.1f} is neutral")

    close, ma_50, ma_200 = ind["last_close"], ind["ma_50"], ind["ma_200"]
    if ma_50 is not None and ma_200 is not None:
        # Only report the golden/death-cross relationship, which is the part
        # practitioners actually read, rather than every MA crossing.
        if ma_50 > ma_200:
            readings.append(f"the 50-day average ({ma_50:,.2f}) is above the 200-day ({ma_200:,.2f}), an uptrend")
        else:
            readings.append(f"the 50-day average ({ma_50:,.2f}) is below the 200-day ({ma_200:,.2f}), a downtrend")

    if ma_200 is not None:
        side = "above" if close > ma_200 else "below"
        readings.append(f"price ({close:,.2f}) is {side} its 200-day average")

    zscore = ind["vol_zscore"]
    if zscore is not None and abs(zscore) >= VOL_ZSCORE_UNUSUAL:
        direction = "decline" if zscore < 0 else "advance"
        readings.append(
            f"the latest move is a {abs(zscore):.1f}-sigma {direction} against 60-day realised volatility, "
            f"statistically unusual for this name"
        )

    ratio = ind["volume_ratio"]
    if ratio is not None and ratio >= VOLUME_SPIKE_RATIO:
        readings.append(f"volume is {ratio:.1f}x its 20-day average, an elevated-participation session")

    return readings


@specialist(AGENT)
def technical_node(payload: dict, ticker: str) -> tuple[list, list, list]:
    """
    Fetch price history and derive technical readings for one ticker.

    Returns
    -------
    tuple
        ``(findings, citations, tool_calls)``.
    """
    started = time.monotonic()
    indicators = get_indicators([ticker])
    elapsed = int((time.monotonic() - started) * 1000)

    calls = [
        tool_record(
            AGENT,
            "get_indicators",
            args={"ticker": ticker},
            provider="yfinance",
            latency_ms=elapsed,
            ok=bool(indicators),
        )
    ]

    ind = indicators.get(ticker)
    if ind is None:
        return [], [], calls

    citation = make_citation(
        "YFINANCE",
        f"{ticker}@{ind['as_of']}",
        as_of=ind["as_of"] or date.today().isoformat(),
        ticker=ticker,
    )
    citations = [citation]

    findings = [
        finding(
            AGENT,
            f"{ticker} closed at {ind['last_close']:,.2f} on {ind['as_of']}, "
            f"{ind['change_pct_1d']:+.2f}% on the day, "
            f"{ind['change_pct_5d']:+.2f}% over 5 sessions and {ind['change_pct_20d']:+.2f}% over 20",
            ticker=ticker,
            metric="last_close",
            value=ind["last_close"],
            unit="USD",
            citations=[citation],
        )
    ]

    for reading in interpret(ind):
        findings.append(
            finding(
                AGENT,
                f"{ticker}: {reading}",
                ticker=ticker,
                # No `value`: these are interpretations of numbers already
                # reported above, so grounding them numerically would
                # double-count the same figure.
                citations=[citation],
            )
        )

    return findings, citations, calls
