# ═══════════════════════════════════════════════════════
# FinSight — Shared Schemas
# ═══════════════════════════════════════════════════════
#
# Purpose : Types crossing subsystem boundaries. Graph-local state lives in
#           src/research/state.py and src/monitor/state.py instead.
#
# Public API:
#   SourceType, Severity, AlertType
#   Citation, AgentFinding, ToolCallRecord, Conflict
#   SOURCE_TRUST, make_citation()
#
# Design note:
#   Citations are created by the DATA LAYER, not by the LLM. Every data-source
#   wrapper returns its source_id/source_url alongside the value, so grounding
#   is structural rather than something the model is asked to remember.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Literal, TypedDict

# ── Enumerations ────────────────────────────────────────
SourceType = Literal["EDGAR", "FRED", "YFINANCE", "FMP", "FINNHUB", "RSS"]
Severity = Literal["LOW", "MED", "HIGH"]
AlertType = Literal["PRICE_MOVE", "NEW_FILING", "NEWS_SENTIMENT", "MACRO_EVENT"]

# Trust ranking used by the aggregator when two sources disagree. EDGAR and
# FRED are authoritative primary sources; the rest are convenience APIs that
# re-publish (and sometimes mangle) the same figures.
SOURCE_TRUST: dict[str, int] = {
    "EDGAR": 100,
    "FRED": 100,
    "FMP": 60,
    "FINNHUB": 55,
    "YFINANCE": 40,
    "RSS": 20,
}

# Two values agreeing within this relative tolerance are treated as the same
# number (absorbs rounding: 391_035_000_000 vs "$391.0B").
CONFLICT_REL_TOLERANCE: float = 0.01


# ── Core records ────────────────────────────────────────
class Citation(TypedDict):
    """
    A pointer back to the primary source of a single claim.

    ``source_id`` is the citable identifier for its type:
      EDGAR    -> accession number, e.g. "0000320193-24-000123"
      FRED     -> series id, e.g. "CPIAUCSL"
      YFINANCE -> f"{ticker}@{date}"
    """

    source_type: SourceType
    source_id: str
    url: str
    as_of: str
    excerpt: str | None


class AgentFinding(TypedDict):
    """
    One grounded claim produced by one specialist agent.

    Specialists return these; the aggregator merges them; the citation verifier
    checks every number in the final answer against ``value`` here.
    """

    agent: str
    ticker: str | None
    claim: str
    metric: str | None
    value: float | str | None
    unit: str | None
    citations: list[Citation]
    confidence: float
    error: str | None


class ToolCallRecord(TypedDict):
    """
    One external call, recorded for the audit trail.

    ``provider_used`` matters: a provider chain may silently fall through to a
    lower-trust source, and an answer citing a fallback should say so.
    """

    node: str
    tool: str
    args: dict
    provider_used: str
    cache_hit: bool
    latency_ms: int
    ok: bool


class Conflict(TypedDict):
    """
    Two sources disagreeing about the same metric beyond tolerance.

    Surfaced in the answer rather than silently resolved — saying "EDGAR reports
    $391.0B, yfinance reports $383.3B; using the filed figure" is the difference
    between a demo and a system.
    """

    metric: str
    ticker: str | None
    values: list[tuple[str, float]]
    chosen_source: str
    chosen_value: float
    rel_difference: float


# ── Helpers ─────────────────────────────────────────────
_URL_TEMPLATES: dict[str, str] = {
    "FRED": "https://fred.stlouisfed.org/series/{source_id}",
    "YFINANCE": "https://finance.yahoo.com/quote/{ticker}",
}


def make_citation(
    source_type: SourceType,
    source_id: str,
    *,
    as_of: str,
    url: str | None = None,
    excerpt: str | None = None,
    ticker: str | None = None,
) -> Citation:
    """
    Build a Citation, deriving a canonical URL when one is not supplied.

    Parameters
    ----------
    source_type : SourceType
        Which provider the value came from.
    source_id : str
        Citable identifier (accession number, FRED series id, ...).
    as_of : str
        ISO date the value refers to — not the date it was fetched.
    url : str, optional
        Explicit URL; derived from a template when omitted.
    excerpt : str, optional
        Supporting text, for narrative claims retrieved from filings.
    ticker : str, optional
        Needed only to build a yfinance URL.

    Returns
    -------
    Citation
    """
    if url is None:
        template = _URL_TEMPLATES.get(source_type, "")
        url = template.format(source_id=source_id, ticker=ticker or "") if template else ""

    return Citation(
        source_type=source_type,
        source_id=source_id,
        url=url,
        as_of=as_of,
        excerpt=excerpt,
    )
