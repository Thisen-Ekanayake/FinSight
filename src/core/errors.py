# ═══════════════════════════════════════════════════════
# FinSight — Exception Hierarchy
# ═══════════════════════════════════════════════════════
#
# Purpose : One base class so callers can catch FinSightError broadly,
#           with specific subclasses where recovery differs.
#
# Public API:
#   FinSightError
#     ├─ ConfigurationError ─ MissingCredentialError
#     ├─ InfrastructureError ─ QdrantIsolationError
#     ├─ DataSourceError ─ RateLimitExceeded, BudgetExhausted
#     └─ VerificationFailed
# ═══════════════════════════════════════════════════════

from __future__ import annotations


class FinSightError(Exception):
    """Base class for every error FinSight raises deliberately."""


# ── Configuration ───────────────────────────────────────
class ConfigurationError(FinSightError):
    """Something in .env or the tool config is wrong or absent."""


class MissingCredentialError(ConfigurationError):
    """A required API key is not set."""


# ── Infrastructure ──────────────────────────────────────
class InfrastructureError(FinSightError):
    """A backing service is unreachable or misconfigured."""


class QdrantIsolationError(InfrastructureError):
    """
    The configured Qdrant is not FinSight's own instance.

    Raised when the client detects collections belonging to another project on
    this machine, which means QDRANT_URL points at the wrong port. Writing to
    that instance could damage unrelated data, so we refuse to proceed.
    """


# ── Data sources ────────────────────────────────────────
class DataSourceError(FinSightError):
    """
    An external data provider failed.

    Caught by the provider-chain executor in src/data/providers.py, which then
    falls through to the next provider rather than aborting the whole request.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class RateLimitExceeded(DataSourceError):
    """A provider returned 429, or our own token bucket refused the call."""


class BudgetExhausted(DataSourceError):
    """A provider's daily free-tier budget is spent; it is disabled until UTC midnight."""


# ── Agent output ────────────────────────────────────────
class VerificationFailed(FinSightError):
    """
    The citation verifier could not ground the answer.

    Only raised when a repair loop has already been exhausted; the normal path
    is to strip unsupported claims and attach a caveat instead.
    """
