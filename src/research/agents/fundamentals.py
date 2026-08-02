# ═══════════════════════════════════════════════════════
# FinSight — Fundamentals Specialist
# ═══════════════════════════════════════════════════════
#
# Purpose : Filed financial figures. The authoritative numeric path.
#
# Public API:
#   fundamentals_node(payload)
#   select_metrics(sub_question)
#
# Every value here originates in SEC XBRL companyfacts — exact as reported,
# carrying the accession number of the filing it came from. No HTML table is
# ever parsed for a number, and the LLM is never asked to recall one.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time

from src.core.schemas import make_citation
from src.data.fundamentals import METRIC_CONCEPTS, get_fundamentals
from src.research.agents._common import finding, specialist, tool_record

logger = logging.getLogger(__name__)

AGENT = "fundamentals"

# Keyword -> metrics, so a focused sub-question fetches only what it needs
# rather than every concept. Order matters: longer phrases are checked first.
_METRIC_KEYWORDS: list[tuple[tuple[str, ...], list[str]]] = [
    (("gross margin", "gross profit"), ["gross_profit", "revenue"]),
    (("operating margin", "operating income"), ["operating_income", "revenue"]),
    (("net margin", "profit margin"), ["net_income", "revenue"]),
    (("net income", "profit", "earnings"), ["net_income"]),
    (("revenue", "sales", "top line"), ["revenue"]),
    (("eps", "per share"), ["eps_diluted"]),
    (("cash flow", "operating cash"), ["operating_cash_flow"]),
    (("r&d", "research and development"), ["rd_expense"]),
    (("assets", "balance sheet"), ["total_assets", "total_liabilities", "stockholders_equity"]),
    (("cash", "liquidity"), ["cash"]),
    (("equity", "book value"), ["stockholders_equity"]),
    (("debt", "liabilities", "leverage"), ["total_liabilities", "stockholders_equity"]),
]

# Fallback when nothing matches — the figures most questions actually want.
_DEFAULT_METRICS = ["revenue", "net_income", "gross_profit", "operating_income"]

_UNIT_LABEL = {"USD": "USD", "USD/shares": "USD per share", "shares": "shares"}


def select_metrics(sub_question: str) -> list[str]:
    """
    Choose which metrics to fetch from a sub-question.

    Keyword matching rather than an LLM call: this runs inside every fan-out
    branch, and a deterministic lookup is faster, free, and reproducible.

    Parameters
    ----------
    sub_question : str
        The focused question routed to this specialist.

    Returns
    -------
    list of str
        Metric names from METRIC_CONCEPTS. Never empty.
    """
    lowered = sub_question.lower()
    selected: list[str] = []

    for keywords, metrics in _METRIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            selected.extend(m for m in metrics if m not in selected)

    valid = [m for m in selected if m in METRIC_CONCEPTS]
    return valid or list(_DEFAULT_METRICS)


@specialist(AGENT)
def fundamentals_node(payload: dict, ticker: str) -> tuple[list, list, list]:
    """
    Fetch filed financial metrics for one ticker.

    Returns
    -------
    tuple
        ``(findings, citations, tool_calls)`` — merged by the fan-in reducers.
    """
    sub_question = payload.get("sub_question", "")
    metrics = select_metrics(sub_question)

    started = time.monotonic()
    data = get_fundamentals(ticker, metrics=metrics, annual_only=True)
    elapsed = int((time.monotonic() - started) * 1000)

    calls = [
        tool_record(
            AGENT,
            "get_fundamentals",
            args={"ticker": ticker, "metrics": metrics},
            provider=next(iter(data.values()))["provider"] if data else "none",
            latency_ms=elapsed,
            ok=bool(data),
        )
    ]

    if not data:
        return [], [], calls

    findings = []
    citations = []

    for name, record in sorted(data.items()):
        citation = make_citation(
            "EDGAR" if record["provider"] == "edgar_xbrl" else record["provider"].upper(),  # type: ignore[arg-type]
            record["source_id"],
            as_of=record["as_of"],
            url=record["source_url"],
        )
        citations.append(citation)

        unit = _UNIT_LABEL.get(record["unit"], record["unit"])
        readable = name.replace("_", " ")
        findings.append(
            finding(
                AGENT,
                f"{ticker} reported {readable} of {record['value']:,.0f} {unit} for {record['period']}",
                ticker=ticker,
                metric=name,
                # Raw figure, not a formatted string — the citation verifier
                # matches numbers in the answer against exactly this.
                value=record["value"],
                unit=record["unit"],
                citations=[citation],
                # A fallback provider is less trustworthy than a filed figure,
                # and the aggregator ranks on this.
                confidence=1.0 if record["provider"] == "edgar_xbrl" else 0.7,
            )
        )

    return findings, citations, calls
