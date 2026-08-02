# ═══════════════════════════════════════════════════════
# FinSight — Company News
# ═══════════════════════════════════════════════════════
#
# Purpose : Recent news per ticker, from Finnhub with an RSS fallback.
#
# Public API:
#   get_company_news(ticker, since=..., provider=...)
#   get_news_finnhub(ticker, since, until)
#   get_news_rss(ticker, since)
#
# Design note:
#   Every NewsItem carries a stable ``article_id``. The Phase 6 dedup engine
#   hashes the canonical URL as its exact-match fast path, which catches the
#   ~90% of duplicate news alerts that come from the same story being re-fetched
#   across cycles — at zero embedding or LLM cost.
#
# Usage:
#   python -m src.data.news --ticker AAPL --days 3
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timedelta, timezone

from src.core.errors import DataSourceError, MissingCredentialError
from src.data.cache import fetch_json, fetch_text
from src.data.config import FINNHUB_API_KEY, FINNHUB_BASE
from src.data.rate_limit import guard
from src.data.schemas import NewsItem

logger = logging.getLogger(__name__)

COMPANY_NEWS_URL = f"{FINNHUB_BASE}/company-news"

# Yahoo Finance per-ticker RSS. Unauthenticated and unofficial, but it needs
# no key, which makes it a genuine fallback when Finnhub's quota is spent.
RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


def _article_id(url: str, headline: str) -> str:
    """
    Stable id for an article, used as the dedup fast path.

    Prefers the URL; falls back to the headline when a feed omits the link.
    """
    basis = url.strip() or headline.strip()
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def get_news_finnhub(ticker: str, *, since: datetime, until: datetime | None = None) -> list[NewsItem]:
    """
    Fetch company news from Finnhub.

    Parameters
    ----------
    ticker : str
        Symbol.
    since : datetime
        Earliest publication time.
    until : datetime, optional
        Latest publication time. Defaults to now.

    Returns
    -------
    list of NewsItem
        Newest first.

    Raises
    ------
    MissingCredentialError
        If FINNHUB_API_KEY is unset.
    DataSourceError
        On transport or format failure.
    """
    if not FINNHUB_API_KEY:
        raise MissingCredentialError("FINNHUB_API_KEY is not set. Get one free at https://finnhub.io/register")

    until = until or datetime.now(timezone.utc)
    payload = fetch_json(
        "finnhub",
        COMPANY_NEWS_URL,
        params={
            "symbol": ticker.upper(),
            "from": since.strftime("%Y-%m-%d"),
            "to": until.strftime("%Y-%m-%d"),
            "token": FINNHUB_API_KEY,
        },
        ttl_key="news",
    )

    if not isinstance(payload, list):
        raise DataSourceError("finnhub", f"expected a list of articles, got {type(payload).__name__}")

    since_ts = since.timestamp()
    items: list[NewsItem] = []

    for row in payload:
        published = row.get("datetime")
        if not published or published < since_ts:
            continue

        url = row.get("url", "")
        headline = row.get("headline", "")
        if not headline:
            continue

        items.append(
            NewsItem(
                ticker=ticker.upper(),
                headline=headline,
                summary=row.get("summary", ""),
                source=row.get("source", "finnhub"),
                url=url,
                published_at=datetime.fromtimestamp(published, tz=timezone.utc).isoformat(),
                article_id=_article_id(url, headline),
                sentiment=None,
            )
        )

    items.sort(key=lambda item: item["published_at"], reverse=True)
    logger.info("Finnhub: %s -> %d articles since %s", ticker, len(items), since.date())
    return items


def get_news_rss(ticker: str, *, since: datetime) -> list[NewsItem]:
    """
    Fetch company news from Yahoo Finance RSS. No API key required.

    The fallback when Finnhub is unavailable or its quota is spent.
    """
    import feedparser

    guard("rss")
    raw = fetch_text("rss", RSS_TEMPLATE.format(ticker=ticker.upper()), ttl_key="news")
    feed = feedparser.parse(raw)

    items: list[NewsItem] = []
    for entry in feed.entries:
        parsed = entry.get("published_parsed")
        if parsed:
            published = datetime(*parsed[:6], tzinfo=timezone.utc)
        else:
            published = datetime.now(timezone.utc)

        if published < since:
            continue

        url = entry.get("link", "")
        headline = entry.get("title", "")
        if not headline:
            continue

        items.append(
            NewsItem(
                ticker=ticker.upper(),
                headline=headline,
                summary=entry.get("summary", ""),
                source=entry.get("source", {}).get("title", "Yahoo Finance"),
                url=url,
                published_at=published.isoformat(),
                article_id=_article_id(url, headline),
                sentiment=None,
            )
        )

    items.sort(key=lambda item: item["published_at"], reverse=True)
    logger.info("RSS: %s -> %d articles since %s", ticker, len(items), since.date())
    return items


def get_company_news(
    ticker: str,
    *,
    since: datetime | None = None,
    days: int = 3,
    provider: str | None = None,
) -> list[NewsItem]:
    """
    Fetch company news through the configured fallback chain.

    Parameters
    ----------
    ticker : str
        Symbol.
    since : datetime, optional
        Earliest publication time. Defaults to ``days`` ago.
    days : int, default 3
        Lookback window when ``since`` is not given.
    provider : str, optional
        Force a single provider instead of running the chain.

    Returns
    -------
    list of NewsItem
        Newest first. Empty if every provider in the chain fails.
    """
    since = since or datetime.now(timezone.utc) - timedelta(days=days)

    if provider == "finnhub":
        return get_news_finnhub(ticker, since=since)
    if provider == "rss":
        return get_news_rss(ticker, since=since)

    from src.data.providers import run_chain

    result = run_chain(
        "news",
        {
            "finnhub": lambda: get_news_finnhub(ticker, since=since),
            "rss": lambda: get_news_rss(ticker, since=since),
        },
        default=[],
    )
    return result.value


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Print recent news for a ticker."""
    parser = argparse.ArgumentParser(description="Fetch company news")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--provider", choices=["finnhub", "rss"], help="force one provider")
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    items = get_company_news(args.ticker, days=args.days, provider=args.provider)
    print(f"\n{len(items)} articles for {args.ticker.upper()} (last {args.days}d):\n")
    for item in items[:20]:
        print(f"  {item['published_at'][:16]}  [{item['source']}]")
        print(f"    {item['headline'][:110]}")
        print(f"    id={item['article_id']}  {item['url'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
