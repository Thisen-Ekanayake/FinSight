# ═══════════════════════════════════════════════════════
# FinSight — FRED Macroeconomic Data Client
# ═══════════════════════════════════════════════════════
#
# Purpose : Macro context from the St. Louis Fed. Like EDGAR, FRED is an
#           authoritative primary source: free, generous (~120 req/min, no
#           daily cap), and self-citing via series IDs.
#
# Public API:
#   get_series(series_id, ...)          observations + metadata
#   get_series_batch(series_ids, ...)   several series in one call sequence
#   get_latest_value(series_id)         most recent observation
#   pct_change(series, periods)         simple change helper
#
# Usage:
#   python -m src.data.fred --series CPIAUCSL --start 2024-01-01
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
from datetime import date

from src.core.errors import DataSourceError, MissingCredentialError
from src.data.cache import fetch_json
from src.data.config import FRED_API_KEY, FRED_BASE, WATCHED_FRED_SERIES
from src.data.schemas import MacroSeries, SeriesPoint

logger = logging.getLogger(__name__)

PROVIDER = "fred"

SERIES_URL = f"{FRED_BASE}/series"
OBSERVATIONS_URL = f"{FRED_BASE}/series/observations"
SERIES_PAGE = "https://fred.stlouisfed.org/series/{series_id}"


def _require_key() -> str:
    """Return the FRED API key, failing with an actionable message."""
    if not FRED_API_KEY:
        raise MissingCredentialError(
            "FRED_API_KEY is not set. Get one free (instant, no approval) at "
            "https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return FRED_API_KEY


def get_series(
    series_id: str,
    *,
    start: date | None = None,
    end: date | None = None,
    limit: int | None = None,
) -> MacroSeries:
    """
    Fetch a FRED series with its observations and metadata.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. ``"CPIAUCSL"``. This is the citation ID.
    start, end : date, optional
        Observation date bounds.
    limit : int, optional
        Cap on observations returned (most recent are kept).

    Returns
    -------
    MacroSeries
        Observations oldest-first, plus title/units/frequency and the latest
        value for convenience.

    Raises
    ------
    DataSourceError
        If the series does not exist or the response is malformed.
    MissingCredentialError
        If FRED_API_KEY is unset.
    """
    key = _require_key()
    base_params = {"series_id": series_id, "api_key": key, "file_type": "json"}

    meta_payload = fetch_json(PROVIDER, SERIES_URL, params=base_params, ttl_key="fred_series")
    series_list = meta_payload.get("seriess") or []
    if not series_list:
        raise DataSourceError(PROVIDER, f"series {series_id!r} not found")
    meta = series_list[0]

    obs_params = dict(base_params)
    if start:
        obs_params["observation_start"] = start.isoformat()
    if end:
        obs_params["observation_end"] = end.isoformat()
    if limit:
        obs_params["limit"] = str(limit)
        obs_params["sort_order"] = "desc"

    obs_payload = fetch_json(PROVIDER, OBSERVATIONS_URL, params=obs_params, ttl_key="fred_series")

    observations: list[SeriesPoint] = []
    for row in obs_payload.get("observations", []):
        raw = row.get("value", ".")
        # FRED encodes missing observations as "." — skip rather than zero-fill,
        # which would silently invent data points.
        if raw == "." or raw is None:
            continue
        try:
            observations.append(SeriesPoint(date=row["date"], value=float(raw)))
        except (ValueError, KeyError):
            continue

    observations.sort(key=lambda p: p["date"])

    latest = observations[-1] if observations else None
    series = MacroSeries(
        series_id=series_id,
        title=meta.get("title", series_id),
        units=meta.get("units", ""),
        frequency=meta.get("frequency", ""),
        observations=observations,
        latest_value=latest["value"] if latest else None,
        latest_date=latest["date"] if latest else None,
        url=SERIES_PAGE.format(series_id=series_id),
    )

    logger.info("FRED: %s -> %d observations (latest %s)", series_id, len(observations), series["latest_date"])
    return series


def get_series_batch(
    series_ids: list[str],
    *,
    start: date | None = None,
    limit: int | None = None,
) -> dict[str, MacroSeries]:
    """
    Fetch several series, tolerating individual failures.

    Used by the Phase 6 macro monitor, which watches a fixed set of series
    shared across every ticker — these are series lookups, not per-symbol
    ones, so this is a handful of calls per cycle regardless of watchlist size.

    Returns
    -------
    dict
        ``{series_id: MacroSeries}``. Failed series are logged and omitted
        rather than aborting the whole batch.
    """
    results: dict[str, MacroSeries] = {}
    for series_id in series_ids:
        try:
            results[series_id] = get_series(series_id, start=start, limit=limit)
        except DataSourceError as exc:
            logger.warning("FRED: skipping %s — %s", series_id, exc)
    return results


def get_latest_value(series_id: str) -> SeriesPoint | None:
    """Return only the most recent observation for a series."""
    series = get_series(series_id, limit=1)
    obs = series["observations"]
    return obs[-1] if obs else None


def pct_change(series: MacroSeries, *, periods: int = 1) -> float | None:
    """
    Percent change over the last ``periods`` observations.

    Returns
    -------
    float or None
        None if there is insufficient history or the base value is zero.
    """
    obs = series["observations"]
    if len(obs) <= periods:
        return None

    current = obs[-1]["value"]
    previous = obs[-1 - periods]["value"]
    if previous == 0:
        return None
    return (current - previous) / abs(previous) * 100.0


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Inspect FRED series from the shell."""
    parser = argparse.ArgumentParser(description="Query FRED")
    parser.add_argument("--series", action="append", help="series id (repeatable); defaults to the watched set")
    parser.add_argument("--start", help="ISO start date")
    parser.add_argument("--limit", type=int, help="max observations")
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    series_ids = args.series or list(WATCHED_FRED_SERIES)
    start = date.fromisoformat(args.start) if args.start else None

    for series_id in series_ids:
        series = get_series(series_id, start=start, limit=args.limit)
        change = pct_change(series)
        print(f"\n{series['series_id']} — {series['title']}")
        print(f"  units      {series['units']}  ({series['frequency']})")
        print(f"  latest     {series['latest_value']} on {series['latest_date']}")
        if change is not None:
            print(f"  change     {change:+.2f}% vs prior observation")
        print(f"  n obs      {len(series['observations'])}")
        print(f"  {series['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
