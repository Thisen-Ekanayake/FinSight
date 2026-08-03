# ═══════════════════════════════════════════════════════
# FinSight — Watchlist
# ═══════════════════════════════════════════════════════
#
# Purpose : Decide what this cycle watches, and how far back each monitor
#           should look.
#
# Public API:
#   ensure_seeded()                     first-run population from .env
#   add_ticker(ticker) / remove_ticker(ticker)
#   current_watchlist()                 -> list[WatchedTicker]
#   lookback_for(ticker, monitor, last_checked)  -> datetime
#   load_watchlist_node(state)          the graph's entry node
#
# ══ THE WATERMARK IS THE WHOLE DESIGN ══
#   A poller asks "what filings does AAPL have?" and gets four hundred. A
#   monitor asks "what has AAPL filed since 09:31 this morning?" and gets
#   none, cheaply, almost every time.
#
#   Two bounds keep that honest:
#     FLOOR  a missing watermark means DEFAULT_LOOKBACK_DAYS, not 1970. A
#            first-ever check must not report a decade of filings as new.
#     CEILING however stale the watermark, MAX_LOOKBACK_DAYS caps it. A process
#            down for a month must not wake up and ask EDGAR for a month of
#            history across the whole watchlist at once.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.monitor.config import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_WATCHLIST,
    MAX_LOOKBACK_DAYS,
    MONITOR_CHECKPOINT_KEYS,
)
from src.monitor.state import MonitorState, WatchedTicker
from src.persistence.repository import add_watch_item, get_checkpoints, list_watchlist, remove_watch_item

logger = logging.getLogger(__name__)


def _company_name(ticker: str) -> str:
    """Best-effort registrant name, falling back to the ticker itself."""
    from src.data.edgar import resolve_company_name

    return resolve_company_name(ticker) or ticker


def ensure_seeded() -> int:
    """
    Populate an empty watchlist from ``MONITOR_WATCHLIST``.

    Runs only when the table is empty, so a user who deliberately removed a
    default ticker does not get it back on the next start — the .env value is
    a starting point, not a floor.

    Returns
    -------
    int
        Tickers added. Zero when the watchlist already had entries.
    """
    if list_watchlist(active_only=False):
        return 0

    for ticker in DEFAULT_WATCHLIST:
        add_watch_item(ticker, company_name=_company_name(ticker))

    logger.info("Seeded watchlist from MONITOR_WATCHLIST: %s", ", ".join(DEFAULT_WATCHLIST))
    return len(DEFAULT_WATCHLIST)


def add_ticker(ticker: str, *, company_name: str = "") -> WatchedTicker:
    """
    Add a ticker, resolving its registered name if one was not supplied.

    Parameters
    ----------
    ticker : str
        US-listed symbol.
    company_name : str, optional
        Display name. Looked up from EDGAR when omitted.

    Returns
    -------
    WatchedTicker
    """
    symbol = ticker.strip().upper()
    row = add_watch_item(symbol, company_name=company_name or _company_name(symbol))
    return WatchedTicker(ticker=row["ticker"], company_name=row["company_name"], warmed_up=row["warmed_up"])


def remove_ticker(ticker: str) -> bool:
    """Deactivate a ticker. Returns True if it was being watched."""
    return remove_watch_item(ticker)


def current_watchlist() -> list[WatchedTicker]:
    """Return the active watchlist in graph-state shape."""
    return [
        WatchedTicker(
            ticker=row["ticker"],
            company_name=row["company_name"] or row["ticker"],
            warmed_up=row["warmed_up"],
        )
        for row in list_watchlist()
    ]


def lookback_for(ticker: str, monitor: str, last_checked: dict[str, str], *, now: datetime | None = None) -> datetime:
    """
    Resolve the ``since`` timestamp for one monitor on one ticker.

    Parameters
    ----------
    ticker : str
        Symbol.
    monitor : str
        Node name, e.g. ``"filing_monitor"``.
    last_checked : dict
        ``{"TICKER:key": iso}`` watermarks, as carried in graph state.
    now : datetime, optional
        Injected for tests.

    Returns
    -------
    datetime
        Timezone-aware UTC, clamped into
        ``[now - MAX_LOOKBACK_DAYS, now - 0]``.
    """
    moment = now or datetime.now(timezone.utc)
    key = f"{ticker.upper()}:{MONITOR_CHECKPOINT_KEYS.get(monitor, monitor)}"

    floor = moment - timedelta(days=MAX_LOOKBACK_DAYS)
    raw = last_checked.get(key)

    if not raw:
        return moment - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("Unparseable watermark %r for %s — falling back to default lookback", raw, key)
        return moment - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    # A watermark that arrived without an offset is UTC by construction; see
    # repository._iso_utc for why one can still slip through.
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)

    # A clock skew or a hand-edited row could put the watermark in the future,
    # which would make `since > now` and silently return nothing forever.
    return min(max(stamp, floor), moment)


def load_watchlist_node(state: MonitorState) -> dict:
    """
    Graph node: resolve what to watch and how far back to look.

    Reads the watchlist from SQLite unless one was injected into the state —
    which is how the CLI runs a single-ticker cycle and how the eval replays a
    frozen watchlist without touching the database.

    Returns a partial state with ``watchlist`` and ``last_checked``, both
    single-writer.
    """
    watchlist = state.get("watchlist") or current_watchlist()

    if not watchlist:
        logger.warning("Watchlist is empty — the cycle will produce nothing")
        return {"watchlist": [], "last_checked": {}}

    tickers = [item["ticker"] for item in watchlist]
    last_checked = get_checkpoints(tickers)

    cold = [t for t in tickers if not any(key.startswith(f"{t}:") for key in last_checked)]
    logger.info(
        "Cycle %s: %d tickers (%s)%s",
        state.get("cycle_id", "?"),
        len(tickers),
        ", ".join(tickers),
        f" — {len(cold)} never checked: {', '.join(cold)}" if cold else "",
    )

    return {"watchlist": watchlist, "last_checked": last_checked}
