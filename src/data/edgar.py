# ═══════════════════════════════════════════════════════
# FinSight — SEC EDGAR Client
# ═══════════════════════════════════════════════════════
#
# Purpose : Authoritative filings and financial facts. EDGAR is the primary
#           source in FinSight: free, unlimited, and self-citing via accession
#           numbers.
#
# Public API:
#   resolve_cik(ticker)                     ticker -> zero-padded CIK
#   resolve_company_name(ticker)            ticker -> registrant name
#   get_filing_index(ticker, ...)           recent filings, filterable
#   get_company_facts(cik, ...)             XBRL facts by concept
#   get_latest_fact(cik, concept)           most recent value for one concept
#   download_filing_document(filing)        primary doc -> disk (forever)
#   filing_url(filing)                      canonical EDGAR URL
#
# ══ THE LOAD-BEARING DECISION ══
#   NUMBERS COME FROM XBRL. NARRATIVE COMES FROM RAG.
#
#   A 10-K's financial statements become unreadable soup under naive HTML text
#   extraction, and that is exactly where hallucinated figures originate. So
#   the LLM is never asked to read a number out of a table. companyfacts gives
#   exact values AND an accession number for free; the filing text is used
#   only for narrative sections (Items 1, 1A, 3, 7, 7A).
#
# Usage:
#   python -m src.data.edgar --ticker AAPL --form 10-K
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Any

from src.core.errors import DataSourceError
from src.data.cache import cached_file, fetch_json
from src.data.config import (
    EDGAR_CACHE_DIR,
    SEC_ARCHIVES_BASE,
    SEC_COMPANY_TICKERS_URL,
    SEC_COMPANYFACTS_URL,
    SEC_SUBMISSIONS_URL,
)
from src.data.schemas import FilingRef, XBRLFact

logger = logging.getLogger(__name__)

PROVIDER = "sec"

# Concepts the fundamentals agent asks for most. US-GAAP taxonomy names.
COMMON_CONCEPTS: dict[str, str] = {
    "revenue": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "revenue_alt": "Revenues",
    "net_income": "NetIncomeLoss",
    "gross_profit": "GrossProfit",
    "operating_income": "OperatingIncomeLoss",
    "total_assets": "Assets",
    "total_liabilities": "Liabilities",
    "stockholders_equity": "StockholdersEquity",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
    "eps_diluted": "EarningsPerShareDiluted",
    "shares_diluted": "WeightedAverageNumberOfDilutedSharesOutstanding",
    "rd_expense": "ResearchAndDevelopmentExpense",
    "operating_cash_flow": "NetCashProvidedByUsedInOperatingActivities",
}

_TICKER_MAP_CACHE: dict[str, str] = {}
_TITLE_MAP_CACHE: dict[str, str] = {}


# ── CIK resolution ──────────────────────────────────────
def _load_ticker_map() -> dict[str, str]:
    """Fetch and memoise the SEC's ticker -> CIK mapping, and its titles."""
    global _TICKER_MAP_CACHE, _TITLE_MAP_CACHE
    if _TICKER_MAP_CACHE:
        return _TICKER_MAP_CACHE

    payload = fetch_json(PROVIDER, SEC_COMPANY_TICKERS_URL, ttl_key="edgar_tickers")
    # Shape is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    entries = [entry for entry in payload.values() if isinstance(entry, dict) and entry.get("ticker")]

    _TICKER_MAP_CACHE = {entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in entries}
    _TITLE_MAP_CACHE = {entry["ticker"].upper(): str(entry.get("title") or "") for entry in entries}

    logger.info("Loaded SEC ticker map (%d symbols)", len(_TICKER_MAP_CACHE))
    return _TICKER_MAP_CACHE


def resolve_company_name(ticker: str) -> str:
    """
    Resolve a ticker to its registered company name.

    Comes from the same cached ``company_tickers.json`` as the CIK map, so it
    costs no extra request. Used by the monitoring subsystem, where the company
    name is part of the canonical alert text.

    Parameters
    ----------
    ticker : str
        US-listed symbol.

    Returns
    -------
    str
        The registrant's name, or ``""`` if the ticker is unknown. Deliberately
        does NOT raise: a missing display name must not stop a cycle, and every
        caller has the ticker to fall back on.
    """
    try:
        _load_ticker_map()
    except Exception as exc:  # noqa: BLE001 - a cosmetic lookup must not be fatal
        logger.warning("Company-name lookup unavailable: %s", exc)
        return ""
    return _TITLE_MAP_CACHE.get(ticker.upper(), "")


def resolve_cik(ticker: str) -> str:
    """
    Resolve a ticker to its zero-padded 10-digit CIK.

    Parameters
    ----------
    ticker : str
        US-listed symbol, e.g. ``"AAPL"``. Case-insensitive.

    Returns
    -------
    str
        Zero-padded CIK, e.g. ``"0000320193"``.

    Raises
    ------
    DataSourceError
        If the ticker is not in the SEC's mapping — commonly because it is not
        US-listed, in which case EDGAR has nothing for it at all.
    """
    mapping = _load_ticker_map()
    cik = mapping.get(ticker.upper())
    if cik is None:
        raise DataSourceError(
            PROVIDER,
            f"ticker {ticker!r} not found in SEC company_tickers.json — " f"EDGAR only covers US-listed registrants.",
        )
    return cik


# ── Filings ─────────────────────────────────────────────
def filing_url(filing: FilingRef) -> str:
    """Canonical EDGAR URL for a filing's primary document."""
    cik_int = str(int(filing["cik"]))
    accession_nodash = filing["accession_no"].replace("-", "")
    return f"{SEC_ARCHIVES_BASE}/{cik_int}/{accession_nodash}/{filing['primary_document']}"


def get_filing_index(
    ticker: str,
    *,
    forms: list[str] | None = None,
    since: date | None = None,
    limit: int = 40,
) -> list[FilingRef]:
    """
    List a company's recent filings, most recent first.

    Parameters
    ----------
    ticker : str
        US-listed symbol.
    forms : list of str, optional
        Restrict to these form types, e.g. ``["10-K", "10-Q"]``. All if None.
    since : date, optional
        Only filings on or after this date. This is what makes the Phase 6
        filing monitor a monitor rather than a poller.
    limit : int, default 40
        Maximum filings to return.

    Returns
    -------
    list of FilingRef
        Newest first.

    Raises
    ------
    DataSourceError
        If the ticker cannot be resolved or the SEC response is malformed.
    """
    cik = resolve_cik(ticker)
    payload = fetch_json(PROVIDER, SEC_SUBMISSIONS_URL.format(cik10=cik), ttl_key="edgar_submissions")

    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return []

    # The SEC returns parallel arrays, not records — zip them back together.
    accessions = recent.get("accessionNumber", [])
    columns = {
        "form": recent.get("form", []),
        "filingDate": recent.get("filingDate", []),
        "reportDate": recent.get("reportDate", []),
        "primaryDocument": recent.get("primaryDocument", []),
        "items": recent.get("items", []),
    }

    wanted = {f.upper() for f in forms} if forms else None
    results: list[FilingRef] = []

    for i, accession in enumerate(accessions):
        form = columns["form"][i] if i < len(columns["form"]) else ""
        if wanted and form.upper() not in wanted:
            continue

        filed = columns["filingDate"][i] if i < len(columns["filingDate"]) else ""
        if since and filed and date.fromisoformat(filed) < since:
            # Arrays are newest-first, so everything past here is older too.
            break

        period = columns["reportDate"][i] if i < len(columns["reportDate"]) else ""
        raw_items = columns["items"][i] if i < len(columns["items"]) else ""

        filing = FilingRef(
            ticker=ticker.upper(),
            cik=cik,
            accession_no=accession,
            form_type=form,
            filing_date=filed,
            period_of_report=period or None,
            primary_document=(columns["primaryDocument"][i] if i < len(columns["primaryDocument"]) else ""),
            url="",
            # 8-K item codes drive severity in Phase 6 — 4.02 is an automatic HIGH.
            items=[item.strip() for item in raw_items.split(",") if item.strip()],
        )
        filing["url"] = filing_url(filing)
        results.append(filing)

        if len(results) >= limit:
            break

    logger.info("EDGAR: %s -> %d filings (forms=%s, since=%s)", ticker, len(results), forms or "all", since)
    return results


def download_filing_document(filing: FilingRef) -> Any:
    """
    Download a filing's primary document to disk, cached permanently.

    Layout: ``data/edgar/{cik}/{accession_no}/{primary_document}``

    Filings are immutable once accepted, so this is cached forever — re-ingest
    and reproducibility cost zero live requests.

    Returns
    -------
    Path
        Local path to the document.
    """
    dest = EDGAR_CACHE_DIR / filing["cik"] / filing["accession_no"] / filing["primary_document"]
    return cached_file(PROVIDER, filing["url"], dest)


# ── XBRL company facts ──────────────────────────────────
def _extract_facts(payload: dict, concept: str, taxonomy: str = "us-gaap") -> list[XBRLFact]:
    """Flatten one concept's unit-keyed entries into XBRLFact records."""
    node = payload.get("facts", {}).get(taxonomy, {}).get(concept)
    if not node:
        return []

    label = node.get("label")
    facts: list[XBRLFact] = []

    for unit, entries in node.get("units", {}).items():
        for entry in entries:
            # Skip entries without an accession number: they cannot be cited,
            # and an uncitable number has no place in this system.
            accession = entry.get("accn")
            if not accession or entry.get("val") is None:
                continue

            facts.append(
                XBRLFact(
                    concept=concept,
                    label=label,
                    value=float(entry["val"]),
                    unit=unit,
                    fiscal_year=int(entry.get("fy") or 0),
                    fiscal_period=entry.get("fp") or "",
                    period_start=entry.get("start"),
                    period_end=entry.get("end", ""),
                    form_type=entry.get("form", ""),
                    accession_no=accession,
                    filed_date=entry.get("filed", ""),
                )
            )

    facts.sort(key=lambda f: (f["period_end"], f["filed_date"]))
    return facts


def get_company_facts(
    cik: str,
    *,
    concepts: list[str] | None = None,
    taxonomy: str = "us-gaap",
) -> dict[str, list[XBRLFact]]:
    """
    Fetch XBRL facts for a company.

    This is where every number in FinSight should come from. Values are exact
    as reported, and each carries the accession number of the filing it came
    from — so citations are structural rather than remembered.

    Parameters
    ----------
    cik : str
        Zero-padded CIK, or a ticker (resolved automatically).
    concepts : list of str, optional
        US-GAAP concept names. Defaults to COMMON_CONCEPTS. Missing concepts
        are omitted rather than raising — coverage varies by filer.
    taxonomy : str, default "us-gaap"
        XBRL taxonomy.

    Returns
    -------
    dict
        ``{concept: [XBRLFact, ...]}``, each list oldest-first.
    """
    if not cik.isdigit():
        cik = resolve_cik(cik)

    payload = fetch_json(PROVIDER, SEC_COMPANYFACTS_URL.format(cik10=cik), ttl_key="edgar_companyfacts")

    wanted = concepts if concepts is not None else list(COMMON_CONCEPTS.values())
    facts = {}
    for concept in wanted:
        extracted = _extract_facts(payload, concept, taxonomy)
        if extracted:
            facts[concept] = extracted

    logger.info("EDGAR XBRL: CIK %s -> %d/%d concepts populated", cik, len(facts), len(wanted))
    return facts


def get_latest_fact(cik: str, concept: str, *, form_type: str | None = None) -> XBRLFact | None:
    """
    Return the most recently reported value for a single concept.

    Parameters
    ----------
    cik : str
        Zero-padded CIK or ticker.
    concept : str
        US-GAAP concept name.
    form_type : str, optional
        Restrict to a form, e.g. ``"10-K"`` for annual figures only.

    Returns
    -------
    XBRLFact or None
        None if the filer does not report this concept.
    """
    facts = get_company_facts(cik, concepts=[concept]).get(concept, [])
    if form_type:
        facts = [f for f in facts if f["form_type"] == form_type]
    return facts[-1] if facts else None


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Inspect EDGAR from the shell. Second run is served from cache."""
    parser = argparse.ArgumentParser(description="Query SEC EDGAR")
    parser.add_argument("--ticker", required=True, help="US-listed symbol, e.g. AAPL")
    parser.add_argument("--form", action="append", help="form type filter (repeatable), e.g. 10-K")
    parser.add_argument("--since", help="only filings on/after this ISO date")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--facts", action="store_true", help="show XBRL facts instead of filings")
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    if args.facts:
        facts = get_company_facts(args.ticker)
        print(f"\nXBRL facts for {args.ticker.upper()}:\n")
        for concept, values in sorted(facts.items()):
            latest = values[-1]
            print(f"  {concept:55s} {latest['value']:>20,.0f} {latest['unit']:8s}")
            print(f"  {'':55s} {latest['fiscal_year']} {latest['fiscal_period']:4s} " f"acc {latest['accession_no']}")
        return 0

    filings = get_filing_index(
        args.ticker,
        forms=args.form,
        since=date.fromisoformat(args.since) if args.since else None,
        limit=args.limit,
    )
    print(f"\n{len(filings)} filings for {args.ticker.upper()}:\n")
    for f in filings:
        items = f" items={','.join(f['items'])}" if f["items"] else ""
        print(f"  {f['filing_date']}  {f['form_type']:8s}  {f['accession_no']}{items}")
        print(f"    {f['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
