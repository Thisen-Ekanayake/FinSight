# ═══════════════════════════════════════════════════════
# FinSight — Filing Monitor
# ═══════════════════════════════════════════════════════
#
# Purpose : Notice new SEC filings since this ticker was last checked.
#
# Public API:
#   filing_monitor_node(payload)
#   ITEM_DESCRIPTIONS
#
# ══ PER-TICKER, AND WATERMARKED ══
#   EDGAR's submissions endpoint is per-CIK, so this monitor gets one Send()
#   per ticker — unlike price and macro, which batch. That is a property of
#   the API, not a choice, and graph.py encodes it directly.
#
#   `since` is what makes this a monitor. The submissions arrays are
#   newest-first, so get_filing_index stops reading as soon as it passes the
#   watermark: a quiet ticker costs one small cached request and returns
#   nothing.
#
# ══ THE ITEM CODES CARRY THE INFORMATION ══
#   An 8-K only says "something happened that shareholders must be told about
#   promptly". Its item codes say WHAT. Item 4.02 — the auditor has told the
#   company its previously issued financials should not be relied upon — is
#   among the most serious things a public company ever files, and it arrives
#   in exactly the same envelope as a routine press release under Item 8.01.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time
from datetime import datetime

from src.data.schemas import FilingRef
from src.monitor.config import FILING_WATCHED_FORMS
from src.monitor.monitors._common import candidate, monitor
from src.monitor.state import CandidateAlert

logger = logging.getLogger(__name__)

MONITOR_NAME = "filing_monitor"

# Ceiling per ticker per cycle. A company that filed twenty 8-Ks since the last
# check has had a remarkable week, and reporting all twenty is not more useful
# than reporting the most recent few plus the count.
MAX_FILINGS_PER_CYCLE: int = 8

# Plain-English gloss for the item codes that drive severity, so an alert says
# what happened rather than making the reader look up a number.
ITEM_DESCRIPTIONS: dict[str, str] = {
    "1.01": "entry into a material definitive agreement",
    "1.03": "bankruptcy or receivership",
    "2.02": "results of operations and financial condition",
    "2.06": "material impairment",
    "3.01": "notice of delisting or failure to satisfy a listing rule",
    "4.01": "change in the registrant's certifying accountant",
    "4.02": "non-reliance on previously issued financial statements",
    "5.02": "departure or appointment of principal officers",
    "5.07": "submission of matters to a vote of security holders",
    "8.01": "other events",
}


def _describe(filing: FilingRef) -> str:
    """Readable detail for a filing, naming its item codes where known."""
    form = filing["form_type"]
    parts = [f"{form} filed {filing['filing_date']}"]

    if filing.get("period_of_report"):
        parts.append(f"for the period ending {filing['period_of_report']}")

    items = filing.get("items") or []
    if items:
        described = [
            f"Item {code} ({ITEM_DESCRIPTIONS[code]})" if code in ITEM_DESCRIPTIONS else f"Item {code}"
            for code in items
        ]
        parts.append("carrying " + ", ".join(described))

    return "; ".join(parts).capitalize() + f". Accession {filing['accession_no']}."


def _headline(filing: FilingRef) -> str:
    """A one-line headline that leads with the most serious item code."""
    ticker = filing["ticker"]
    form = filing["form_type"]

    for code in filing.get("items") or []:
        if code in ITEM_DESCRIPTIONS:
            return f"{ticker} filed an {form}: {ITEM_DESCRIPTIONS[code]}"

    return f"{ticker} filed a {form}"


@monitor(MONITOR_NAME)
def filing_monitor_node(payload: dict) -> tuple[list[CandidateAlert], list]:
    """
    Check one ticker for filings accepted since the watermark.

    Returns
    -------
    tuple
        ``(candidates, api_calls)``.
    """
    from src.core.schemas import make_citation
    from src.data.edgar import get_filing_index
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
        since_date = datetime.fromisoformat(raw_since).date() if raw_since else None

        started = time.monotonic()
        filings = get_filing_index(
            ticker,
            forms=FILING_WATCHED_FORMS,
            since=since_date,
            limit=MAX_FILINGS_PER_CYCLE,
        )
        calls.append(
            tool_record(
                MONITOR_NAME,
                "get_filing_index",
                args={"ticker": ticker, "forms": FILING_WATCHED_FORMS, "since": str(since_date)},
                provider="sec",
                latency_ms=int((time.monotonic() - started) * 1000),
                ok=True,
            )
        )

        for filing in filings:
            citation = make_citation(
                "EDGAR",
                filing["accession_no"],
                as_of=filing["filing_date"],
                url=filing["url"],
            )
            candidates.append(
                candidate(
                    ticker,
                    "NEW_FILING",
                    monitor_name=MONITOR_NAME,
                    company_name=companies.get(ticker, ""),
                    headline=_headline(filing),
                    detail=_describe(filing),
                    # The accession number IS the event identity — globally
                    # unique, assigned by the SEC, and immutable once accepted.
                    # No other alert type gets a key this good.
                    natural_key=filing["accession_no"],
                    metrics={
                        "form_type": filing["form_type"],
                        "items": filing.get("items") or [],
                        "filing_date": filing["filing_date"],
                        "period_of_report": filing.get("period_of_report"),
                        "accession_no": filing["accession_no"],
                    },
                    evidence=[citation],
                    observed_at=filing["filing_date"],
                )
            )

    return candidates, calls
