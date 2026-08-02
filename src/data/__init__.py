# ═══════════════════════════════════════════════════════
# FinSight — Data Layer Package
# ═══════════════════════════════════════════════════════
#
# Every external data source, wrapped, cached, rate-limited, and
# fallback-chained. Nothing above this layer talks to an API directly.
#
# Public API:
#   edgar        get_filing_index, get_company_facts, resolve_cik, ...
#   fred         get_series, get_series_batch, get_latest_value
#   prices       get_prices (BATCHED), compute_indicators, get_indicators
#   news         get_company_news
#   fundamentals get_fundamentals, get_metric
#   providers    run_chain, ChainResult
#   rate_limit   guard, budget_status
#
# THE CITATION IS BORN HERE. Every wrapper returns the identifier needed to
# cite its value — accession number, FRED series id, ticker@date — so
# grounding is structural rather than something the LLM is asked to remember.
#
# Only lightweight symbols are re-exported: importing `src.data` must not drag
# pandas, yfinance, and feedparser into a process that needs one source.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from src.data.providers import ChainResult, run_chain
from src.data.rate_limit import budget_status, guard
from src.data.schemas import (
    FilingRef,
    FundamentalMetric,
    IndicatorSet,
    MacroSeries,
    NewsItem,
    PriceBar,
    SeriesPoint,
    XBRLFact,
)

__all__ = [
    "run_chain",
    "ChainResult",
    "guard",
    "budget_status",
    "FilingRef",
    "XBRLFact",
    "SeriesPoint",
    "MacroSeries",
    "PriceBar",
    "IndicatorSet",
    "NewsItem",
    "FundamentalMetric",
]
