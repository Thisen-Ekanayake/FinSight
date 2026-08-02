# ═══════════════════════════════════════════════════════
# FinSight — Provider Fallback Chains
# ═══════════════════════════════════════════════════════
#
# Purpose : Run a chain of interchangeable data providers in trust order,
#           falling through on failure, and RECORD WHICH ONE ACTUALLY SERVED
#           the value.
#
# Public API:
#   run_chain(kind, providers, default=...) -> ChainResult
#   ChainResult
#
# Why the provenance matters:
#   Ingesting a fallback silently is how you end up with wrong numbers wearing
#   confident citations. The fundamentals chain tries EDGAR XBRL first, then
#   yfinance, then FMP — and those three do not always agree. So the provider
#   that served each value is stamped onto its Citation, the aggregator ranks
#   by SOURCE_TRUST, and a disagreement beyond tolerance is SURFACED in the
#   answer rather than silently resolved.
#
# Usage:
#   result = run_chain("prices", {
#       "yfinance": lambda: fetch_from_yfinance(ticker),
#       "finnhub":  lambda: fetch_from_finnhub(ticker),
#   })
#   result.value, result.provider, result.attempts
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from src.core.errors import DataSourceError, FinSightError
from src.data.config import PROVIDER_CHAINS

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ProviderAttempt:
    """One provider's outcome within a chain run."""

    provider: str
    ok: bool
    latency_ms: int
    error: str | None = None


@dataclass
class ChainResult(Generic[T]):
    """
    Outcome of a provider chain run.

    Attributes
    ----------
    value : T
        Whatever the successful provider returned, or ``default``.
    provider : str
        Which provider served the value. ``"none"`` if all failed.
    attempts : list of ProviderAttempt
        Every provider tried, in order — the audit trail.
    degraded : bool
        True when the value came from something other than the first-choice
        provider. Worth surfacing: it means the most-trusted source was
        unavailable.
    """

    value: T
    provider: str
    attempts: list[ProviderAttempt] = field(default_factory=list)
    degraded: bool = False

    @property
    def ok(self) -> bool:
        """True if any provider succeeded."""
        return self.provider != "none"


def run_chain(
    kind: str,
    providers: dict[str, Callable[[], T]],
    *,
    default: T | None = None,
    order: list[str] | None = None,
) -> ChainResult[T]:
    """
    Try each provider in trust order until one succeeds.

    Parameters
    ----------
    kind : str
        Chain name, keying into PROVIDER_CHAINS for the default ordering
        (e.g. ``"prices"``, ``"fundamentals"``, ``"news"``).
    providers : dict
        ``{provider_name: zero-arg callable}``. Only providers present in both
        this dict and the configured order are attempted.
    default : T, optional
        Value returned when every provider fails. If omitted, the last error
        is re-raised instead.
    order : list of str, optional
        Override the configured ordering.

    Returns
    -------
    ChainResult
        Carrying the value, the provider that served it, and every attempt.

    Raises
    ------
    DataSourceError
        If all providers fail and no ``default`` was supplied.
    """
    chain = order or PROVIDER_CHAINS.get(kind, list(providers))
    candidates = [name for name in chain if name in providers]

    if not candidates:
        raise DataSourceError(kind, f"no usable providers for chain {kind!r} (configured: {chain})")

    attempts: list[ProviderAttempt] = []
    last_error: Exception | None = None

    for position, name in enumerate(candidates):
        started = time.monotonic()
        try:
            value = providers[name]()
        except FinSightError as exc:
            # Expected failures — exhausted budget, rate limit, provider down.
            elapsed = int((time.monotonic() - started) * 1000)
            attempts.append(ProviderAttempt(name, ok=False, latency_ms=elapsed, error=str(exc)))
            last_error = exc
            logger.warning("Chain %s: %s failed (%s), trying next", kind, name, exc)
            continue
        except Exception as exc:  # noqa: BLE001 - a broken provider must not kill the chain
            elapsed = int((time.monotonic() - started) * 1000)
            attempts.append(ProviderAttempt(name, ok=False, latency_ms=elapsed, error=repr(exc)))
            last_error = exc
            logger.warning("Chain %s: %s raised %r, trying next", kind, name, exc)
            continue

        elapsed = int((time.monotonic() - started) * 1000)
        attempts.append(ProviderAttempt(name, ok=True, latency_ms=elapsed))

        degraded = position > 0
        if degraded:
            # Not an error, but the answer should be able to say the
            # authoritative source was unavailable.
            logger.warning(
                "Chain %s: DEGRADED — served by %s after %d failure(s); first choice was %s",
                kind,
                name,
                position,
                candidates[0],
            )
        else:
            logger.debug("Chain %s: served by %s in %dms", kind, name, elapsed)

        return ChainResult(value=value, provider=name, attempts=attempts, degraded=degraded)

    if default is not None:
        logger.error("Chain %s: ALL providers failed (%s); returning default", kind, candidates)
        return ChainResult(value=default, provider="none", attempts=attempts, degraded=True)

    raise DataSourceError(kind, f"all providers failed: {[a.provider for a in attempts]}") from last_error
