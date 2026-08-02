# ═══════════════════════════════════════════════════════
# FinSight — Rate Limiting & Daily Budgets
# ═══════════════════════════════════════════════════════
#
# Purpose : Two independent protections on the free-tier data APIs.
#             1. TokenBucket  — per-second pacing (SEC bans clients that burst)
#             2. DailyBudget  — per-day call caps (FMP gives you 250, total)
#
# Public API:
#   acquire(provider)           block until a call is permitted
#   check_budget(provider)      raise BudgetExhausted if the day is spent
#   record_call(provider)       increment the daily counter
#   guard(provider)             both of the above, in the right order
#   budget_status()             snapshot for GET /admin/budgets
#   reset_buckets()
#
# Persistence note:
#   Budgets live in memory for Phase 1 and move to SQLite in Phase 4, when the
#   persistence layer exists. The interface here does not change — only
#   _BUDGET_STATE's backing store does.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timezone
from typing import TypedDict

from src.core.errors import BudgetExhausted
from src.data.config import BUDGET_SOFT_LIMIT, DAILY_BUDGETS, RATE_LIMITS

logger = logging.getLogger(__name__)


class BudgetSnapshot(TypedDict):
    """Current daily-budget state for one provider."""

    provider: str
    used: int
    limit: int
    soft_limit: int
    remaining: int
    exhausted: bool
    resets_at: str


# ── Token bucket ────────────────────────────────────────
class TokenBucket:
    """
    A thread-safe token bucket for per-second request pacing.

    Tokens refill continuously at ``rate`` per second up to ``capacity``.
    ``acquire`` blocks until a token is available, so callers never need to
    think about sleeping. Thread-safe because parallel graph branches share
    one bucket per provider — that sharing is the point.

    Parameters
    ----------
    rate : float
        Sustained requests per second.
    capacity : float, optional
        Burst size. Defaults to one second's worth, minimum 1.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> float:
        """
        Block until ``tokens`` are available, then consume them.

        Returns
        -------
        float
            Seconds spent waiting — useful for logging and for spotting a
            provider chain that is silently throttling a hot path.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited

                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate

            time.sleep(sleep_for)
            waited += sleep_for


_BUCKETS: dict[str, TokenBucket] = {}
_BUCKETS_LOCK = threading.Lock()


def get_bucket(provider: str) -> TokenBucket:
    """Return the process-wide bucket for a provider, creating it on first use."""
    with _BUCKETS_LOCK:
        if provider not in _BUCKETS:
            rate = RATE_LIMITS.get(provider, 1.0)
            _BUCKETS[provider] = TokenBucket(rate)
            logger.debug("Token bucket for %s: %.2f req/s", provider, rate)
        return _BUCKETS[provider]


def acquire(provider: str, tokens: float = 1.0) -> float:
    """Block until the provider's bucket permits a call. Returns seconds waited."""
    waited = get_bucket(provider).acquire(tokens)
    if waited > 1.0:
        logger.info("Rate limit: waited %.2fs for %s", waited, provider)
    return waited


# ── Daily budgets ───────────────────────────────────────
_BUDGET_STATE: dict[str, tuple[date, int]] = {}
_BUDGET_LOCK = threading.Lock()


def _today() -> date:
    """Budgets reset at UTC midnight, matching how providers count."""
    return datetime.now(timezone.utc).date()


def _current_usage(provider: str) -> int:
    """Read today's call count, resetting the counter if the date rolled over."""
    day, count = _BUDGET_STATE.get(provider, (_today(), 0))
    if day != _today():
        _BUDGET_STATE[provider] = (_today(), 0)
        return 0
    return count


def check_budget(provider: str) -> None:
    """
    Verify the provider has budget left today.

    Enforces the SOFT limit (80% by default), not the hard cap — stopping at
    200/250 leaves headroom instead of failing mid-cycle at exactly the limit.

    Raises
    ------
    BudgetExhausted
        If the provider is disabled or its soft limit is reached.
    """
    limit = DAILY_BUDGETS.get(provider)
    if limit is None:
        return  # unmetered provider

    if limit == 0:
        raise BudgetExhausted(provider, "provider is disabled (daily budget is 0)")

    with _BUDGET_LOCK:
        used = _current_usage(provider)

    soft = int(limit * BUDGET_SOFT_LIMIT)
    if used >= soft:
        raise BudgetExhausted(
            provider,
            f"daily soft limit reached ({used}/{limit}, soft cap {soft}). Resets at UTC midnight.",
        )


def record_call(provider: str) -> int:
    """
    Increment the provider's daily counter and return the new total.

    Warns once when crossing 50% and again at the soft limit, so budget
    pressure is visible in logs before it becomes a failure.
    """
    limit = DAILY_BUDGETS.get(provider)
    if limit is None or limit == 0:
        return 0

    with _BUDGET_LOCK:
        used = _current_usage(provider) + 1
        _BUDGET_STATE[provider] = (_today(), used)

    soft = int(limit * BUDGET_SOFT_LIMIT)
    if used == soft:
        logger.warning("Budget: %s reached its soft limit (%d/%d) — disabled until UTC midnight", provider, used, limit)
    elif used == limit // 2:
        logger.info("Budget: %s at 50%% (%d/%d)", provider, used, limit)

    return used


def guard(provider: str) -> None:
    """
    Full pre-call check: budget first, then rate limit.

    Budget is checked BEFORE pacing so an exhausted provider fails instantly
    instead of sleeping in the token bucket only to be rejected anyway.

    Raises
    ------
    BudgetExhausted
        If the provider's daily budget is spent.
    """
    check_budget(provider)
    acquire(provider)


def budget_status() -> list[BudgetSnapshot]:
    """Snapshot every metered provider. Backs GET /admin/budgets in Phase 4."""
    with _BUDGET_LOCK:
        snapshots = []
        for provider, limit in DAILY_BUDGETS.items():
            used = _current_usage(provider)
            soft = int(limit * BUDGET_SOFT_LIMIT)
            snapshots.append(
                BudgetSnapshot(
                    provider=provider,
                    used=used,
                    limit=limit,
                    soft_limit=soft,
                    remaining=max(0, soft - used),
                    exhausted=limit == 0 or used >= soft,
                    resets_at="00:00 UTC",
                )
            )
    return snapshots


def reset_buckets() -> None:
    """Clear buckets and budget counters. For tests."""
    with _BUCKETS_LOCK:
        _BUCKETS.clear()
    with _BUDGET_LOCK:
        _BUDGET_STATE.clear()
