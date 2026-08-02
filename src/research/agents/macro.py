# ═══════════════════════════════════════════════════════
# FinSight — Macro Specialist
# ═══════════════════════════════════════════════════════
#
# Purpose : US macroeconomic context from FRED.
#
# Public API:
#   macro_node(payload)
#   select_series(sub_question)
#
# ══ THIS AGENT IGNORES TICKERS ══
#   FRED series are economy-wide, not company-scoped. The router therefore
#   dispatches macro ONCE per query rather than once per ticker — fanning it
#   out per ticker would issue N identical FRED calls for the same answer.
#   graph.py encodes that asymmetry directly in the Send() topology.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time

from src.core.schemas import make_citation
from src.data.config import WATCHED_FRED_SERIES
from src.data.fred import get_series, pct_change
from src.research.agents._common import finding, specialist, tool_record

logger = logging.getLogger(__name__)

AGENT = "macro"

# Keyword -> FRED series id.
_SERIES_KEYWORDS: list[tuple[tuple[str, ...], list[str]]] = [
    (("inflation", "cpi", "price level", "cost of living"), ["CPIAUCSL"]),
    (("fed funds", "federal funds", "policy rate", "interest rate", "rate cut", "rate hike"), ["DFF"]),
    (("unemployment", "jobs", "labor market", "labour market", "employment"), ["UNRATE"]),
    (("treasury", "10-year", "10 year", "bond yield", "long rate"), ["DGS10"]),
    (("yield curve", "inversion", "spread", "recession signal"), ["T10Y2Y"]),
]

# When the question is vaguely macro, these three give the broadest picture.
_DEFAULT_SERIES = ["DFF", "CPIAUCSL", "UNRATE"]

# Two years of history — enough to describe a trend without pulling decades.
_OBSERVATION_LIMIT = 24


def select_series(sub_question: str) -> list[str]:
    """
    Choose which FRED series to fetch.

    Parameters
    ----------
    sub_question : str
        The focused question routed to this specialist.

    Returns
    -------
    list of str
        FRED series ids. Never empty.
    """
    lowered = sub_question.lower()
    selected: list[str] = []

    for keywords, series in _SERIES_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            selected.extend(s for s in series if s not in selected)

    return selected or list(_DEFAULT_SERIES)


@specialist(AGENT)
def macro_node(payload: dict, ticker: str) -> tuple[list, list, list]:
    """
    Fetch macroeconomic series relevant to the sub-question.

    ``ticker`` is accepted for signature symmetry with the other specialists
    and deliberately unused — see the module docstring.

    Returns
    -------
    tuple
        ``(findings, citations, tool_calls)``.
    """
    sub_question = payload.get("sub_question") or payload.get("query", "")
    series_ids = select_series(sub_question)

    findings = []
    citations = []
    calls = []

    for series_id in series_ids:
        started = time.monotonic()
        try:
            series = get_series(series_id, limit=_OBSERVATION_LIMIT)
        except Exception as exc:  # noqa: BLE001 - one bad series must not lose the others
            calls.append(
                tool_record(
                    AGENT,
                    "get_series",
                    args={"series_id": series_id},
                    provider="fred",
                    latency_ms=int((time.monotonic() - started) * 1000),
                    ok=False,
                )
            )
            logger.warning("%s: series %s failed — %s", AGENT, series_id, exc)
            continue

        calls.append(
            tool_record(
                AGENT,
                "get_series",
                args={"series_id": series_id},
                provider="fred",
                latency_ms=int((time.monotonic() - started) * 1000),
                ok=True,
            )
        )

        if series["latest_value"] is None:
            continue

        citation = make_citation(
            "FRED",
            series_id,
            as_of=series["latest_date"] or "",
            url=series["url"],
        )
        citations.append(citation)

        change = pct_change(series)
        trend = f", {change:+.2f}% versus the prior observation" if change is not None else ""
        findings.append(
            finding(
                AGENT,
                f"{series['title']} ({series_id}) was {series['latest_value']} {series['units']} "
                f"as of {series['latest_date']}{trend}",
                ticker=None,
                metric=series_id,
                value=series["latest_value"],
                unit=series["units"],
                citations=[citation],
            )
        )

    return findings, citations, calls


# Exposed for the monitoring subsystem in Phase 6, which watches the same set.
WATCHED_SERIES = WATCHED_FRED_SERIES
