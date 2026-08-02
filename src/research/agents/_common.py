# ═══════════════════════════════════════════════════════
# FinSight — Specialist Agent Helpers
# ═══════════════════════════════════════════════════════
#
# Purpose : Shared plumbing so each specialist stays about its own data
#           source rather than repeating error handling and bookkeeping.
#
# Public API:
#   finding(...)          build an AgentFinding
#   tool_record(...)      build a ToolCallRecord
#   specialist(name)      decorator: timing, error capture, partial-state shape
#
# ══ A FAILING SPECIALIST MUST NOT KILL THE FAN-OUT ══
#   Branches run concurrently in one superstep. An exception in one would
#   abort the whole superstep and lose the other three branches' work, so
#   every specialist is wrapped: failures become an entry in `errors` and the
#   graph continues with partial results. A partial answer that says what is
#   missing beats no answer.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from src.core.schemas import AgentFinding, Citation, ToolCallRecord

logger = logging.getLogger(__name__)


def finding(
    agent: str,
    claim: str,
    *,
    ticker: str | None = None,
    metric: str | None = None,
    value: float | str | None = None,
    unit: str | None = None,
    citations: list[Citation] | None = None,
    confidence: float = 1.0,
) -> AgentFinding:
    """
    Build one grounded claim.

    ``value`` is what the citation verifier matches numbers in the final
    answer against, so it must be the raw figure the tool returned — not a
    formatted or rounded string.
    """
    return AgentFinding(
        agent=agent,
        ticker=ticker,
        claim=claim,
        metric=metric,
        value=value,
        unit=unit,
        citations=citations or [],
        confidence=confidence,
        error=None,
    )


def tool_record(
    node: str,
    tool: str,
    *,
    args: dict[str, Any] | None = None,
    provider: str = "",
    cache_hit: bool = False,
    latency_ms: int = 0,
    ok: bool = True,
) -> ToolCallRecord:
    """Build one audit-trail entry for an external call."""
    return ToolCallRecord(
        node=node,
        tool=tool,
        args=args or {},
        provider_used=provider,
        cache_hit=cache_hit,
        latency_ms=latency_ms,
        ok=ok,
    )


def specialist(name: str) -> Callable:
    """
    Wrap a specialist function into a graph node.

    The wrapped function receives ``(payload, ticker)`` and returns
    ``(findings, citations, tool_calls)``. The decorator handles timing,
    exception capture, and shaping the partial state.

    Parameters
    ----------
    name : str
        Agent name, used in logs, findings, and the audit trail.

    Returns
    -------
    Callable
        A decorator producing a node with signature ``(payload) -> dict``.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def node(payload: dict) -> dict:
            ticker = (payload.get("ticker") or "").upper()
            started = time.monotonic()

            try:
                findings, citations, calls = func(payload, ticker)
            except Exception as exc:  # noqa: BLE001
                # Deliberately broad: one branch failing must not abort the
                # superstep and discard the other branches' results.
                elapsed = int((time.monotonic() - started) * 1000)
                message = f"{name}({ticker or 'n/a'}): {type(exc).__name__}: {exc}"
                logger.warning("Specialist failed — %s", message, exc_info=logger.isEnabledFor(logging.DEBUG))
                return {
                    "findings": [],
                    "citations": [],
                    "errors": [message],
                    "tool_calls": [tool_record(name, "unknown", latency_ms=elapsed, ok=False)],
                }

            elapsed = int((time.monotonic() - started) * 1000)
            logger.info("%s(%s): %d findings in %dms", name, ticker or "n/a", len(findings), elapsed)

            return {
                "findings": findings,
                "citations": citations,
                "errors": [],
                "tool_calls": calls,
            }

        return node

    return decorator
