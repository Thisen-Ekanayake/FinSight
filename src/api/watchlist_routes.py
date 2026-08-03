# ═══════════════════════════════════════════════════════
# FinSight — Watchlist Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : Manage what the monitoring subsystem watches.
#
# Routes:
#   GET    /watchlist                what is watched, and when each monitor
#                                    last checked it
#   POST   /watchlist                add a ticker
#   DELETE /watchlist/{ticker}       stop watching one
#
# ══ REMOVAL IS A SOFT DELETE ══
#   Deactivating rather than deleting keeps the MonitorCheckpoint rows. Drop a
#   ticker for a month, add it back, and the filing monitor resumes from where
#   it left off — rather than reporting the company's entire filing history as
#   new in one burst.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import WatchItemOut, WatchItemRequest
from src.monitor.watchlist import add_ticker, remove_ticker
from src.persistence.repository import get_checkpoints, list_watchlist

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/watchlist", tags=["monitoring"])


def _watermarks(ticker: str, checkpoints: dict[str, str]) -> dict[str, str]:
    """Pull one ticker's per-monitor watermarks out of the flat key space."""
    prefix = f"{ticker}:"
    return {key[len(prefix) :]: value for key, value in checkpoints.items() if key.startswith(prefix)}


@router.get("", response_model=list[WatchItemOut], summary="What is being watched")
async def get_watchlist(include_inactive: bool = False) -> list[WatchItemOut]:
    """
    List watched tickers with each monitor's last-checked timestamp.

    An empty ``last_checked`` means that monitor has never run for that ticker,
    which is different from "checked and found nothing" — the monitors read it
    as "fall back to the default lookback".
    """
    rows = list_watchlist(active_only=not include_inactive)
    checkpoints = get_checkpoints([row["ticker"] for row in rows]) if rows else {}

    return [
        WatchItemOut(
            ticker=row["ticker"],
            company_name=row["company_name"] or row["ticker"],
            warmed_up=row["warmed_up"],
            added_at=row["added_at"],
            last_checked=_watermarks(row["ticker"], checkpoints),
        )
        for row in rows
    ]


@router.post("", response_model=WatchItemOut, status_code=201, summary="Watch a ticker")
async def post_watchlist(request: WatchItemRequest) -> WatchItemOut:
    """
    Add a ticker, resolving its registered name from EDGAR when not supplied.

    Idempotent: adding a ticker that is already watched reactivates it and
    returns it, rather than erroring. It deliberately does NOT reset
    ``warmed_up`` — a ticker whose events are already in the dedup index does
    not need another warmup cycle, and forcing one would delay real alerts.
    """
    try:
        item = add_ticker(request.ticker, company_name=request.company_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to add %s: %s", request.ticker, exc)
        raise HTTPException(status_code=400, detail=f"Could not add {request.ticker!r}: {exc}") from exc

    checkpoints = get_checkpoints([item["ticker"]])
    return WatchItemOut(
        ticker=item["ticker"],
        company_name=item["company_name"] or item["ticker"],
        warmed_up=item["warmed_up"],
        added_at="",
        last_checked=_watermarks(item["ticker"], checkpoints),
    )


@router.delete("/{ticker}", status_code=204, summary="Stop watching a ticker")
async def delete_watchlist(ticker: str) -> None:
    """Deactivate a ticker. 404 if it was not being watched."""
    if not remove_ticker(ticker):
        raise HTTPException(status_code=404, detail=f"{ticker.upper()} is not being watched")
