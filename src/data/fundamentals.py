# ═══════════════════════════════════════════════════════
# FinSight — Company Fundamentals
# ═══════════════════════════════════════════════════════
#
# Purpose : Financial metrics with provenance, via the chain
#           edgar_xbrl -> yfinance -> fmp.
#
# Public API:
#   get_fundamentals(ticker, metrics=...)   -> {metric: FundamentalMetric}
#   get_metric(ticker, metric)              -> FundamentalMetric | None
#   METRIC_CONCEPTS
#
# ══ WHY EDGAR IS FIRST, ALWAYS ══
#   EDGAR XBRL is authoritative, free, and self-citing: every fact carries the
#   accession number of the filing it was reported in. yfinance and FMP
#   re-publish the same figures with their own normalisation, and they do not
#   always agree with the filed number.
#
#   So EDGAR leads the chain, and when a fallback serves a value the provider
#   is stamped onto the record. The Phase 3 aggregator ranks by SOURCE_TRUST
#   and SURFACES disagreements instead of silently picking one:
#     "EDGAR (10-K, accession 0000320193-24-000123) reports $391.0B;
#      yfinance reports $383.3B. Using the filed figure."
#
# Usage:
#   python -m src.data.fundamentals --ticker AAPL
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging

from src.core.errors import DataSourceError
from src.data.edgar import COMMON_CONCEPTS, get_company_facts, resolve_cik
from src.data.providers import ChainResult, run_chain
from src.data.rate_limit import guard
from src.data.schemas import FundamentalMetric, XBRLFact

logger = logging.getLogger(__name__)

# Friendly metric name -> US-GAAP concept. Some metrics have alternates
# because filers differ in which concept they report revenue under.
METRIC_CONCEPTS: dict[str, list[str]] = {
    "revenue": [COMMON_CONCEPTS["revenue"], COMMON_CONCEPTS["revenue_alt"]],
    "net_income": [COMMON_CONCEPTS["net_income"]],
    "gross_profit": [COMMON_CONCEPTS["gross_profit"]],
    "operating_income": [COMMON_CONCEPTS["operating_income"]],
    "total_assets": [COMMON_CONCEPTS["total_assets"]],
    "total_liabilities": [COMMON_CONCEPTS["total_liabilities"]],
    "stockholders_equity": [COMMON_CONCEPTS["stockholders_equity"]],
    "cash": [COMMON_CONCEPTS["cash"]],
    "eps_diluted": [COMMON_CONCEPTS["eps_diluted"]],
    "rd_expense": [COMMON_CONCEPTS["rd_expense"]],
    "operating_cash_flow": [COMMON_CONCEPTS["operating_cash_flow"]],
}

# yfinance .info keys for the same metrics, used only on fallback.
_YF_KEYS: dict[str, str] = {
    "revenue": "totalRevenue",
    "net_income": "netIncomeToCommon",
    "gross_profit": "grossProfits",
    "total_assets": "totalAssets",
    "cash": "totalCash",
    "eps_diluted": "trailingEps",
}

EDGAR_ACCESSION_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}"


def _from_edgar(
    ticker: str,
    metrics: list[str],
    *,
    annual_only: bool,
    periods: int = 1,
) -> dict[str, list[FundamentalMetric]]:
    """
    Pull metrics from XBRL companyfacts. The authoritative path.

    Returns up to ``periods`` reporting periods per metric, newest last. More
    than one is what makes trend questions ("how did margin trend?") and
    specific-year questions ("fiscal 2024") answerable at all — with only the
    latest value there is nothing to compare against.
    """
    cik = resolve_cik(ticker)

    wanted_concepts = [concept for metric in metrics for concept in METRIC_CONCEPTS.get(metric, [])]
    facts = get_company_facts(cik, concepts=wanted_concepts)

    results: dict[str, list[FundamentalMetric]] = {}
    for metric in metrics:
        for concept in METRIC_CONCEPTS.get(metric, []):
            candidates = facts.get(concept, [])
            if annual_only:
                candidates = [f for f in candidates if f["form_type"] == "10-K"]
            if not candidates:
                continue

            # Companyfacts repeats a period across amended and later filings;
            # keep one entry per fiscal period so "3 periods" means 3 years,
            # not 3 restatements of the same year.
            by_period: dict[str, XBRLFact] = {}
            for fact in candidates:
                by_period[f"{fact['fiscal_year']}-{fact['fiscal_period']}"] = fact

            selected = sorted(by_period.values(), key=lambda f: (f["fiscal_year"], f["period_end"]))[-periods:]

            results[metric] = [
                FundamentalMetric(
                    ticker=ticker.upper(),
                    metric=metric,
                    value=fact["value"],
                    unit=fact["unit"],
                    period=f"{fact['fiscal_year']} {fact['fiscal_period']}".strip(),
                    as_of=fact["period_end"],
                    provider="edgar_xbrl",
                    # The accession number IS the citation. No LLM involved.
                    source_id=fact["accession_no"],
                    source_url=EDGAR_ACCESSION_URL.format(cik=cik, form=fact["form_type"]),
                )
                for fact in selected
            ]
            break  # first concept that has data wins

    if not results:
        raise DataSourceError("edgar_xbrl", f"{ticker}: no requested metrics present in XBRL facts")

    logger.info(
        "Fundamentals: %s -> %d/%d metrics from EDGAR XBRL (%d period(s) each)",
        ticker,
        len(results),
        len(metrics),
        periods,
    )
    return results


def _from_yfinance(ticker: str, metrics: list[str]) -> dict[str, list[FundamentalMetric]]:
    """Fallback path. Lower trust: normalised figures, no accession number."""
    import yfinance as yf

    guard("yfinance")
    try:
        info = yf.Ticker(ticker.upper()).info
    except Exception as exc:
        raise DataSourceError("yfinance", f"{ticker}: info lookup failed: {exc}") from exc

    if not info:
        raise DataSourceError("yfinance", f"{ticker}: empty info payload")

    from datetime import date

    today = date.today().isoformat()
    results: dict[str, list[FundamentalMetric]] = {}

    for metric in metrics:
        key = _YF_KEYS.get(metric)
        value = info.get(key) if key else None
        if value is None:
            continue

        # yfinance .info exposes only a trailing-twelve-month snapshot, so the
        # fallback path is always single-period regardless of what was asked.
        results[metric] = [
            FundamentalMetric(
                ticker=ticker.upper(),
                metric=metric,
                value=float(value),
                unit="USD",
                period="TTM",
                as_of=today,
                provider="yfinance",
                source_id=f"{ticker.upper()}@{today}",
                source_url=f"https://finance.yahoo.com/quote/{ticker.upper()}",
            )
        ]

    if not results:
        raise DataSourceError("yfinance", f"{ticker}: none of the requested metrics available")

    logger.info("Fundamentals: %s -> %d/%d metrics from yfinance (FALLBACK)", ticker, len(results), len(metrics))
    return results


def get_fundamentals_history(
    ticker: str,
    *,
    metrics: list[str] | None = None,
    annual_only: bool = True,
    periods: int = 3,
) -> dict[str, list[FundamentalMetric]]:
    """
    Fetch several reporting periods per metric, through the provider chain.

    Use this rather than get_fundamentals whenever the question involves a
    trend or a specific fiscal year. A single latest value cannot answer
    "how did margin trend?" — there is nothing to compare — and cannot answer
    "in fiscal 2024" for a company whose latest 10-K is a different year.

    Parameters
    ----------
    ticker : str
        US-listed symbol.
    metrics : list of str, optional
        Metric names from METRIC_CONCEPTS. Defaults to all.
    annual_only : bool, default True
        Restrict EDGAR facts to 10-K filings, so periods are like-for-like
        annual figures rather than a mix of annual and quarterly.
    periods : int, default 3
        Reporting periods per metric, newest last.

    Returns
    -------
    dict
        ``{metric: [FundamentalMetric, ...]}``, oldest first. Empty if every
        provider failed. Note the yfinance fallback is always single-period.
    """
    metrics = metrics or list(METRIC_CONCEPTS)

    # Annotate the empty default: without it the generic binds T to
    # dict[Never, Never] and the provider callables no longer typecheck.
    empty: dict[str, list[FundamentalMetric]] = {}

    result: ChainResult[dict[str, list[FundamentalMetric]]] = run_chain(
        "fundamentals",
        {
            "edgar_xbrl": lambda: _from_edgar(ticker, metrics, annual_only=annual_only, periods=periods),
            "yfinance": lambda: _from_yfinance(ticker, metrics),
        },
        default=empty,
    )

    if result.degraded and result.ok:
        logger.warning(
            "Fundamentals for %s came from %s, NOT the filed XBRL figures — "
            "downstream citations must reflect that",
            ticker,
            result.provider,
        )

    return result.value


def get_fundamentals(
    ticker: str,
    *,
    metrics: list[str] | None = None,
    annual_only: bool = False,
) -> dict[str, FundamentalMetric]:
    """
    Fetch the most recent value of each fundamental metric.

    Convenience wrapper over get_fundamentals_history for callers that only
    need the latest figure.

    Returns
    -------
    dict
        ``{metric: FundamentalMetric}``. Each record names the provider that
        served it — check ``provider`` before treating a value as filed.
    """
    history = get_fundamentals_history(ticker, metrics=metrics, annual_only=annual_only, periods=1)
    return {metric: records[-1] for metric, records in history.items() if records}


def get_metric(ticker: str, metric: str) -> FundamentalMetric | None:
    """Fetch a single fundamental metric, or None if unavailable."""
    return get_fundamentals(ticker, metrics=[metric]).get(metric)


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Print fundamentals for a ticker, showing which provider served each."""
    parser = argparse.ArgumentParser(description="Fetch company fundamentals")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--metric", action="append", help="metric name (repeatable)")
    parser.add_argument("--annual", action="store_true", help="10-K figures only")
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    data = get_fundamentals(args.ticker, metrics=args.metric, annual_only=args.annual)
    print(f"\nFundamentals for {args.ticker.upper()}:\n")
    for name, record in sorted(data.items()):
        print(f"  {name:22s} {record['value']:>20,.0f} {record['unit']:8s} {record['period']:10s}")
        print(f"  {'':22s} via {record['provider']}  src={record['source_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
