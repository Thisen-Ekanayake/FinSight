# ═══════════════════════════════════════════════════════
# FinSight — Data Layer Schemas
# ═══════════════════════════════════════════════════════
#
# Purpose : Return types for every data-source wrapper. All of them carry
#           their own source_id and source_url.
#
# Public API:
#   FilingRef, XBRLFact, SeriesPoint, MacroSeries
#   PriceBar, IndicatorSet, NewsItem, FundamentalMetric
#
# Design note:
#   THE CITATION IS BORN HERE, not bolted on by the LLM later. Every wrapper
#   returns the identifier needed to cite it — an accession number, a FRED
#   series id, a ticker+date. The citation verifier in Phase 4 can then match
#   numbers in the answer against values the tools actually returned, because
#   provenance travelled with the value the whole way.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import TypedDict


class FilingRef(TypedDict):
    """
    A single SEC filing, identified by its accession number.

    ``accession_no`` is the citation ID and is globally unique. A filing is
    immutable once accepted, so this reference is stable forever.
    """

    ticker: str
    cik: str
    accession_no: str
    form_type: str
    filing_date: str
    period_of_report: str | None
    primary_document: str
    url: str
    items: list[str]


class XBRLFact(TypedDict):
    """
    One reported financial fact from XBRL companyfacts.

    Exact by construction — no HTML table parsing, no LLM reading a number out
    of mangled markup. ``accession_no`` comes straight from the SEC's own
    ``accn`` field, so every fact cites itself.
    """

    concept: str
    label: str | None
    value: float
    unit: str
    fiscal_year: int
    fiscal_period: str
    period_start: str | None
    period_end: str
    form_type: str
    accession_no: str
    filed_date: str


class SeriesPoint(TypedDict):
    """One observation in a macroeconomic time series."""

    date: str
    value: float


class MacroSeries(TypedDict):
    """A FRED series with its observations and metadata."""

    series_id: str
    title: str
    units: str
    frequency: str
    observations: list[SeriesPoint]
    latest_value: float | None
    latest_date: str | None
    url: str


class PriceBar(TypedDict):
    """One OHLCV bar."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorSet(TypedDict):
    """
    Technical indicators for one ticker, computed from its price history.

    ``vol_zscore`` is the current move measured against 60-day realised
    volatility — the monitoring subsystem uses it to distinguish a genuinely
    unusual move from ordinary noise in a volatile name.
    """

    ticker: str
    as_of: str
    last_close: float
    change_pct_1d: float
    change_pct_5d: float
    change_pct_20d: float
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    ma_20: float | None
    ma_50: float | None
    ma_200: float | None
    bb_upper: float | None
    bb_lower: float | None
    volume: float
    avg_volume_20: float | None
    volume_ratio: float | None
    vol_zscore: float | None


class NewsItem(TypedDict):
    """A news article about a ticker."""

    ticker: str
    headline: str
    summary: str
    source: str
    url: str
    published_at: str
    article_id: str
    sentiment: float | None


class FundamentalMetric(TypedDict):
    """
    A single fundamental metric with the provider that actually served it.

    ``provider`` matters: the fundamentals chain tries EDGAR XBRL first and
    falls through to yfinance or FMP. An answer citing a fallback should say
    so, and the aggregator's source-trust ranking needs to know.
    """

    ticker: str
    metric: str
    value: float
    unit: str
    period: str
    as_of: str
    provider: str
    source_id: str
    source_url: str
