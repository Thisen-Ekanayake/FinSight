# ═══════════════════════════════════════════════════════
# FinSight — Monitor Helpers
# ═══════════════════════════════════════════════════════
#
# Purpose : Shared plumbing so each monitor stays about its own data source.
#
# Public API:
#   candidate(...)        build a CandidateAlert
#   monitor(name)         decorator: timing, error capture, partial-state shape
#   bucket(value, size)   the magnitude rounding that makes a price key stable
#
# ══ A FAILING MONITOR MUST NOT KILL THE CYCLE ══
#   Same rule as the research specialists, and it matters more here. Branches
#   run concurrently in one superstep, so an exception in the news monitor for
#   one ticker would abort the superstep and discard the filing monitor's work
#   for every other ticker.
#
#   So every monitor is wrapped: failures become an entry in `monitor_errors`
#   and the cycle continues. A cycle that reports three of four sources beats
#   a cycle that reports nothing — and unlike the research path, there is no
#   human waiting to notice and retry.
#
# ══ AND IT MUST NOT ADVANCE ITS WATERMARK ══
#   The checkpoint is only written for monitors that returned cleanly. A
#   failed fetch that advanced its watermark anyway would skip whatever was
#   published during the outage, permanently and silently.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import functools
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from src.core.schemas import AlertType, Citation
from src.monitor.config import MONITOR_CHECKPOINT_KEYS
from src.monitor.state import CandidateAlert

logger = logging.getLogger(__name__)


def candidate(
    ticker: str,
    alert_type: AlertType,
    *,
    monitor_name: str,
    headline: str,
    detail: str,
    natural_key: str,
    metrics: dict[str, Any] | None = None,
    evidence: list[Citation] | None = None,
    company_name: str = "",
    observed_at: str = "",
) -> CandidateAlert:
    """
    Build one candidate alert.

    ``natural_key`` must identify the underlying EVENT, not the observation.
    Two cycles that see the same event have to produce the same key, or the
    free exact-match path never fires and every duplicate costs an embedding.
    """
    return CandidateAlert(
        ticker=ticker.upper(),
        company_name=company_name or ticker.upper(),
        alert_type=alert_type,
        monitor=monitor_name,
        headline=headline,
        detail=detail,
        natural_key=natural_key,
        metrics=metrics or {},
        evidence=evidence or [],
        observed_at=observed_at or datetime.now(timezone.utc).isoformat(),
    )


def sentence(parts: list[str]) -> str:
    """
    Join clauses into one sentence, capitalising ONLY the first character.

    ``str.capitalize()`` lower-cases everything after the first character,
    which turns "8-K" into "8-k", "RSI 28" into "rsi 28", and "Item 4.02" into
    "item 4.02". Those are identifiers, and mangling them makes an alert read
    as though the system does not know what it is looking at.
    """
    text = "; ".join(part for part in parts if part)
    return (text[:1].upper() + text[1:]) if text else ""


def bucket(value: float, size: float) -> int:
    """
    Round a magnitude down to a band, for use in a natural key.

    ══ WHY A PRICE KEY MUST ROUND BEFORE IT HASHES ══
    A price move re-measured an hour later is a different number and the same
    event. Hashing the raw percentage would give every intraday re-check a
    fresh identity, so the exact-key fast path would never hit and the same
    move would be re-embedded all day.

    Rounding to a band makes "-5.2%" and "-5.4%" collide exactly and for free.
    Moves that land either side of a band edge fall through to the semantic
    path, which is what it is for.

    Parameters
    ----------
    value : float
        Magnitude, signed or not.
    size : float
        Band width in the same units.

    Returns
    -------
    int
        The band index.
    """
    if size <= 0:
        raise ValueError("bucket size must be positive")
    return int(abs(value) // size)


def monitor(name: str) -> Callable:
    """
    Wrap a monitor function into a graph node.

    The wrapped function receives ``(payload)`` and returns
    ``(candidates, api_calls)``. The decorator handles timing, exception
    capture, and shaping the partial state — including the ``checked`` list
    that lets the watermark advance.

    Parameters
    ----------
    name : str
        Node name, used in logs, the audit trail, and the checkpoint key.

    Returns
    -------
    Callable
        A decorator producing a node with signature ``(payload) -> dict``.
    """
    checkpoint_key = MONITOR_CHECKPOINT_KEYS.get(name, name)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def node(payload: dict) -> dict:
            tickers = [t.upper() for t in payload.get("tickers") or []]
            label = ",".join(tickers[:3]) + ("..." if len(tickers) > 3 else "")
            started = time.monotonic()

            try:
                candidates, calls = func(payload)
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad: one branch failing must not abort the
                # superstep and discard every other monitor's work.
                message = f"{name}({label or 'n/a'}): {type(exc).__name__}: {exc}"
                logger.warning("Monitor failed — %s", message, exc_info=logger.isEnabledFor(logging.DEBUG))
                return {
                    "candidates": [],
                    "monitor_errors": [message],
                    "api_calls": [],
                    # Empty "checked": the watermark must NOT advance for a
                    # monitor that failed, or the gap is skipped forever.
                    "checked": [],
                }

            elapsed = int((time.monotonic() - started) * 1000)
            logger.info("%s(%s): %d candidates in %dms", name, label or "n/a", len(candidates), elapsed)

            return {
                "candidates": candidates,
                "monitor_errors": [],
                "api_calls": calls,
                "checked": [f"{ticker}:{checkpoint_key}" for ticker in tickers],
            }

        return node

    return decorator
