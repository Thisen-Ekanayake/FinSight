# ═══════════════════════════════════════════════════════
# FinSight — Data Layer Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : Rate limits, cache TTLs, provider fallback chains, and the SEC
#           User-Agent. Every external call routes through these constants.
#
# Public API:
#   SEC_USER_AGENT, SEC_* endpoints
#   RATE_LIMITS, DAILY_BUDGETS, BUDGET_SOFT_LIMIT
#   CACHE_TTL, PROVIDER_CHAINS
#   FRED_API_KEY, FINNHUB_API_KEY, FMP_API_KEY
#
# Design note:
#   Batch and cache at the MONITOR level, not per-agent. A 10-ticker cycle
#   costs 26 calls when price/macro are batched, and ~10x that if every agent
#   fetches per ticker. The numbers below assume the batched design.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import os

from src.core.config import EDGAR_CACHE_DIR, HTTP_CACHE_PATH  # noqa: F401  (re-exported for convenience)

# ── SEC EDGAR ───────────────────────────────────────────
# MANDATORY. The SEC returns 403 to any client without a descriptive
# User-Agent carrying a real contact address. src/data/cache.py enforces this
# at the session factory, so it is impossible to call EDGAR without one.
SEC_USER_AGENT: str = os.getenv("SEC_USER_AGENT", "")

SEC_DATA_BASE: str = "https://data.sec.gov"
SEC_WWW_BASE: str = "https://www.sec.gov"
SEC_ARCHIVES_BASE: str = "https://www.sec.gov/Archives/edgar/data"
SEC_COMPANY_TICKERS_URL: str = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL: str = SEC_DATA_BASE + "/submissions/CIK{cik10}.json"
SEC_COMPANYFACTS_URL: str = SEC_DATA_BASE + "/api/xbrl/companyfacts/CIK{cik10}.json"

# Placeholder addresses that must never reach the SEC.
_PLACEHOLDER_EMAILS: frozenset[str] = frozenset(
    {"your.email@example.com", "you@example.com", "email@example.com", "user@example.com"}
)

# ── API keys ────────────────────────────────────────────
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")
FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")

FRED_BASE: str = "https://api.stlouisfed.org/fred"
FINNHUB_BASE: str = "https://finnhub.io/api/v1"
FMP_BASE: str = "https://financialmodelingprep.com/api/v3"

# ── Rate limits (requests per second, token bucket) ─────
# Deliberately below each provider's published ceiling so bursts have headroom.
RATE_LIMITS: dict[str, float] = {
    "sec": 8.0,  # published limit is 10/s — 8 leaves room and SEC bans hard
    "fred": 1.8,  # ~120/min
    "finnhub": 0.9,  # 60/min
    "fmp": 1.0,
    "yfinance": 2.0,  # unofficial; self-imposed politeness
    "rss": 2.0,
}

# ── Daily budgets (free-tier hard caps) ─────────────────
# 0 means the provider is disabled entirely.
DAILY_BUDGETS: dict[str, int] = {
    "fmp": 250,
    "alphavantage": 0,  # 25/day is a toy — rejected outright
}

# At this fraction of the daily budget, stop using the provider and log a
# WARNING. Leaves headroom rather than failing mid-cycle at exactly 250.
BUDGET_SOFT_LIMIT: float = 0.8

# ── Cache TTLs (seconds; None = cache forever) ──────────
CACHE_TTL: dict[str, int | None] = {
    # A filing is immutable once accepted — its accession number will never
    # point at different bytes. Cache forever; re-ingest then costs nothing.
    "edgar_filing_doc": None,
    "edgar_submissions": 6 * 3600,
    "edgar_companyfacts": 24 * 3600,
    "edgar_tickers": 7 * 86400,
    "fred_series": 12 * 3600,
    "prices_daily": 24 * 3600,
    "prices_intraday": 15 * 60,
    "news": 30 * 60,
    "fundamentals": 24 * 3600,
}

DEFAULT_CACHE_TTL: int = 3600

# ── Provider fallback chains ────────────────────────────
# Tried in order; each falls through on DataSourceError. The provider that
# actually served a value is stamped onto its Citation — silently ingesting a
# fallback is how you end up with wrong numbers and confident citations.
PROVIDER_CHAINS: dict[str, list[str]] = {
    "prices": ["yfinance", "finnhub", "fmp"],
    # EDGAR first: authoritative, free, and self-citing via accession number.
    "fundamentals": ["edgar_xbrl", "yfinance", "fmp"],
    "news": ["finnhub", "rss"],
    "macro": ["fred"],
}

# ── HTTP behaviour ──────────────────────────────────────
HTTP_TIMEOUT: int = 30
HTTP_MAX_RETRIES: int = 3
HTTP_BACKOFF_FACTOR: float = 0.5
HTTP_RETRY_STATUSES: tuple[int, ...] = (429, 500, 502, 503, 504)

# ── Macro series watched by the monitoring subsystem ────
# Shared across all tickers — these are series, not per-symbol lookups.
WATCHED_FRED_SERIES: dict[str, str] = {
    "DFF": "Federal Funds Effective Rate",
    "CPIAUCSL": "Consumer Price Index (All Urban Consumers)",
    "UNRATE": "Unemployment Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
}


def validate_sec_user_agent() -> str:
    """
    Return the SEC User-Agent, refusing placeholders and empty values.

    Returns
    -------
    str
        A usable User-Agent header value.

    Raises
    ------
    MissingCredentialError
        If unset, or still carrying an example address.
    """
    from src.core.errors import MissingCredentialError

    ua = SEC_USER_AGENT.strip()
    if not ua:
        raise MissingCredentialError(
            "SEC_USER_AGENT is not set. The SEC returns 403 without a descriptive "
            'User-Agent. Set it in .env, e.g. "FinSight/0.1 (you@yourdomain.com)".'
        )

    if any(placeholder in ua.lower() for placeholder in _PLACEHOLDER_EMAILS):
        raise MissingCredentialError(
            f"SEC_USER_AGENT still contains a placeholder address: {ua!r}\n"
            "  Put your real contact email in it — the SEC blocks clients they cannot reach."
        )

    if "@" not in ua:
        raise MissingCredentialError(
            f"SEC_USER_AGENT should include a contact email address, got {ua!r}\n"
            '  Expected something like: "FinSight/0.1 (you@yourdomain.com)"'
        )

    return ua
