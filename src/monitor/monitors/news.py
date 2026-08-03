# ═══════════════════════════════════════════════════════
# FinSight — News Monitor
# ═══════════════════════════════════════════════════════
#
# Purpose : Notice materially negative coverage of a watched ticker.
#
# Public API:
#   news_monitor_node(payload)
#   count_independent_sources(items)
#
# ══ ONE CANDIDATE PER ARTICLE, NOT PER TICKER ══
#   The obvious design — one aggregate "AAPL sentiment is negative" candidate
#   per cycle — throws away the thing that makes the dedup engine worth
#   building. Three outlets covering the same lawsuit are three articles, three
#   URLs, three natural keys, and ONE event; collapsing them upstream means the
#   engine never sees the case it exists for.
#
#   So each significant article is its own candidate, keyed on its URL hash,
#   and the semantic layer collapses them. That is also the honest arrangement:
#   whether two stories are the same story is a judgement about meaning, and
#   the vector index is the part of this system that makes such judgements.
#
# ══ CORROBORATION IS A SEVERITY INPUT, NOT A FILTER ══
#   The count of independent outlets covering the ticker in this window is
#   attached to every candidate's metrics, because HIGH severity requires more
#   than one. Provider sentiment scores are wrong often enough that a single
#   badly-scored headline must not be able to page anyone.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time
from datetime import datetime
from urllib.parse import urlparse

from src.data.schemas import NewsItem
from src.monitor.config import NEWS_MAX_CANDIDATES_PER_TICKER, NEWS_MIN_ABS_SENTIMENT
from src.monitor.monitors._common import candidate, monitor
from src.monitor.state import CandidateAlert

logger = logging.getLogger(__name__)

MONITOR_NAME = "news_monitor"

# Cap on the detail excerpt. The full summary goes nowhere useful — the alert
# body is read in a notification, not a reader.
MAX_SUMMARY_CHARS: int = 300


def count_independent_sources(items: list[NewsItem]) -> int:
    """
    Count distinct publishers in a batch of articles.

    Counted by registrable domain rather than by the provider's ``source``
    label, which is inconsistent: the same outlet arrives as "Reuters",
    "reuters", and "Reuters News" from different feeds, and three spellings of
    one outlet would satisfy a corroboration rule that exists precisely to
    require three OUTLETS.

    Parameters
    ----------
    items : list of NewsItem
        Articles fetched for one ticker in this window.

    Returns
    -------
    int
        Distinct domains. Zero for an empty batch.
    """
    domains: set[str] = set()

    for item in items:
        host = urlparse(item.get("url") or "").netloc.lower()
        host = host[4:] if host.startswith("www.") else host
        if host:
            domains.add(host)
        elif item.get("source"):
            # No URL to normalise — fall back to the label, lower-cased.
            domains.add(str(item["source"]).strip().lower())

    return len(domains)


@monitor(MONITOR_NAME)
def news_monitor_node(payload: dict) -> tuple[list[CandidateAlert], list]:
    """
    Check one ticker's news since the watermark.

    Returns
    -------
    tuple
        ``(candidates, api_calls)``.
    """
    from src.core.schemas import make_citation
    from src.data.news import get_company_news
    from src.research.agents._common import tool_record

    tickers = [t.upper() for t in payload.get("tickers") or []]
    companies = payload.get("companies") or {}
    since_map = payload.get("since") or {}
    if not tickers:
        return [], []

    candidates: list[CandidateAlert] = []
    calls = []

    for ticker in tickers:
        raw_since = since_map.get(ticker)
        since = datetime.fromisoformat(raw_since) if raw_since else None

        started = time.monotonic()
        articles = get_company_news(ticker, since=since)
        calls.append(
            tool_record(
                MONITOR_NAME,
                "get_company_news",
                args={"ticker": ticker, "since": str(since)},
                provider="news",
                latency_ms=int((time.monotonic() - started) * 1000),
                ok=True,
            )
        )

        # Corroboration is measured across EVERYTHING published about the
        # ticker in the window, not only the articles that clear the sentiment
        # bar — the question is how widely the story is being covered.
        source_count = count_independent_sources(articles)

        scored = [item for item in articles if item.get("sentiment") is not None]
        significant = [item for item in scored if abs(float(item["sentiment"] or 0.0)) >= NEWS_MIN_ABS_SENTIMENT]
        significant.sort(key=lambda item: float(item["sentiment"] or 0.0))

        if len(significant) > NEWS_MAX_CANDIDATES_PER_TICKER:
            logger.info(
                "%s(%s): %d significant articles, keeping the %d most negative",
                MONITOR_NAME,
                ticker,
                len(significant),
                NEWS_MAX_CANDIDATES_PER_TICKER,
            )

        for item in significant[:NEWS_MAX_CANDIDATES_PER_TICKER]:
            sentiment = float(item["sentiment"] or 0.0)
            citation = make_citation(
                "FINNHUB" if item.get("source", "").lower() == "finnhub" else "RSS",
                item["article_id"],
                as_of=item["published_at"][:10],
                url=item["url"],
                excerpt=item.get("summary", "")[:MAX_SUMMARY_CHARS],
            )

            candidates.append(
                candidate(
                    ticker,
                    "NEWS_SENTIMENT",
                    monitor_name=MONITOR_NAME,
                    company_name=companies.get(ticker, ""),
                    headline=item["headline"],
                    detail=(item.get("summary") or item["headline"])[:MAX_SUMMARY_CHARS],
                    # article_id is already a hash of the canonical URL — see
                    # src/data/news.py. One article, one key.
                    natural_key=item["article_id"],
                    metrics={
                        "sentiment": sentiment,
                        "source_count": source_count,
                        "source": item.get("source", ""),
                        "published_at": item["published_at"],
                        "url": item["url"],
                    },
                    evidence=[citation],
                    observed_at=item["published_at"],
                )
            )

    return candidates, calls
