# ═══════════════════════════════════════════════════════
# FinSight — UI HTTP Client
# ═══════════════════════════════════════════════════════
#
# Purpose : Every call the dashboard makes to the FastAPI backend, in one
#           place, returning plain dicts (the JSON body, parsed) rather than
#           a parallel set of Pydantic models — the API's schemas.py already
#           IS the contract; duplicating it here would be two places to keep
#           in sync for a UI that only ever reads what it just deserialised.
#
# Public API:
#   ApiError                              raised on any non-2xx response
#   ask(query, thread_id=None)            POST /research/query
#   get_thread(thread_id)                 GET  /research/threads/{id}
#   list_runs(limit)                      GET  /research/runs
#   run_cycle(warmup, tickers)            POST /monitor/cycles
#   get_pending_alerts(cycle_id)          GET  /monitor/cycles/{id}/pending
#   resume_cycle(cycle_id, decisions)     POST /monitor/cycles/{id}/resume
#   list_cycles(limit, status)            GET  /monitor/cycles
#   list_alerts(...)                      GET  /monitor/alerts
#   list_decisions(...)                   GET  /monitor/decisions
#   get_watchlist(include_inactive)       GET  /watchlist
#   add_ticker(ticker, company_name)      POST /watchlist
#   remove_ticker(ticker)                 DELETE /watchlist/{ticker} -> True
#   health()                              GET  /health
#   budgets()                             GET  /admin/budgets
#   config()                              GET  /admin/config
#
# ══ WHY requests, NOT THE API'S OWN PYTHON OBJECTS ══
#   The dashboard is deliberately a separate process from the API — it is the
#   thing a container restart, a different host, or a public deploy would
#   put behind a different origin entirely. Importing src.api.* directly
#   would work today and quietly stop making sense the moment the two are
#   deployed apart, which Phase 8's own docker-compose scaffolding already
#   anticipates (api and ui are SEPARATE services).
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import os
from typing import Any

import requests

API_URL: str = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# A research query fans out across specialists and an LLM synthesis step; a
# monitoring cycle hits up to four external data providers plus, sometimes,
# one summarisation call. Both can genuinely take the better part of a
# minute — the default below is for the many endpoints that are a single
# SQLite read.
DEFAULT_TIMEOUT: int = 30
LONG_TIMEOUT: int = 120


class ApiError(RuntimeError):
    """A non-2xx response, carrying the server's own `detail` message."""


def _request(method: str, path: str, *, timeout: int = DEFAULT_TIMEOUT, **kwargs: Any) -> Any:
    """
    Make one call and return its parsed JSON body.

    Raises
    ------
    ApiError
        On any non-2xx response, or if the backend cannot be reached at all —
        both are the caller's problem to display, never a raw traceback in
        a Streamlit page.
    """
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach the API at {API_URL} — is it running? ({exc})") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"{method} {path} -> {response.status_code}: {detail}")

    return response.json() if response.content else None


# ── Research ────────────────────────────────────────────
def ask(query: str, *, thread_id: str | None = None) -> dict[str, Any]:
    """Ask a research question. Returns the completed QueryResponse body."""
    body: dict[str, Any] = {"query": query}
    if thread_id:
        body["thread_id"] = thread_id
    result: dict[str, Any] = _request("POST", "/research/query", json=body, timeout=LONG_TIMEOUT)
    return result


def get_thread(thread_id: str) -> dict[str, Any]:
    """Replay one run's full audit trail from the checkpointer."""
    result: dict[str, Any] = _request("GET", f"/research/threads/{thread_id}")
    return result


def list_runs(*, limit: int = 25) -> list[dict[str, Any]]:
    """Recent research runs, newest first."""
    result: list[dict[str, Any]] = _request("GET", "/research/runs", params={"limit": limit})
    return result


# ── Monitor ─────────────────────────────────────────────
def run_cycle(*, warmup: bool = False, tickers: list[str] | None = None) -> dict[str, Any]:
    """
    Run one monitoring cycle now.

    May come back with ``status: "PENDING_APPROVAL"`` — a HIGH alert paused
    the graph before anything was dispatched. See resume_cycle.
    """
    result: dict[str, Any] = _request(
        "POST", "/monitor/cycles", json={"warmup": warmup, "tickers": tickers or []}, timeout=LONG_TIMEOUT
    )
    return result


def get_pending_alerts(cycle_id: str) -> list[dict[str, Any]]:
    """The HIGH alerts one paused cycle is actually waiting on, in full."""
    result: list[dict[str, Any]] = _request("GET", f"/monitor/cycles/{cycle_id}/pending")
    return result


def resume_cycle(cycle_id: str, decisions: dict[str, str]) -> dict[str, Any]:
    """Decide a paused cycle's pending alerts — ``{alert_id: "approve" | "reject"}``."""
    result: dict[str, Any] = _request(
        "POST", f"/monitor/cycles/{cycle_id}/resume", json={"decisions": decisions}, timeout=LONG_TIMEOUT
    )
    return result


def list_cycles(*, limit: int = 25, status: str | None = None) -> list[dict[str, Any]]:
    """Recent monitoring cycles, newest first, optionally filtered by status."""
    params: dict[str, Any] = {"limit": limit}
    if status:
        params["status"] = status
    result: list[dict[str, Any]] = _request("GET", "/monitor/cycles", params=params)
    return result


def list_alerts(
    *, limit: int = 50, ticker: str | None = None, severity: str | None = None, status: str | None = None
) -> list[dict[str, Any]]:
    """Fired alerts, newest first, optionally filtered."""
    params = {k: v for k, v in dict(limit=limit, ticker=ticker, severity=severity, status=status).items() if v}
    result: list[dict[str, Any]] = _request("GET", "/monitor/alerts", params=params)
    return result


def list_decisions(*, limit: int = 100, decision: str | None = None) -> list[dict[str, Any]]:
    """Every dedup decision, with the score behind it."""
    params: dict[str, Any] = {"limit": limit}
    if decision:
        params["decision"] = decision
    result: list[dict[str, Any]] = _request("GET", "/monitor/decisions", params=params)
    return result


# ── Watchlist ───────────────────────────────────────────
def get_watchlist(*, include_inactive: bool = False) -> list[dict[str, Any]]:
    """Watched tickers, with each monitor's last-checked watermark."""
    result: list[dict[str, Any]] = _request("GET", "/watchlist", params={"include_inactive": include_inactive})
    return result


def add_ticker(ticker: str, *, company_name: str = "") -> dict[str, Any]:
    """Add (or reactivate) a watched ticker."""
    result: dict[str, Any] = _request("POST", "/watchlist", json={"ticker": ticker, "company_name": company_name})
    return result


def remove_ticker(ticker: str) -> bool:
    """
    Stop watching a ticker (soft delete). Returns True on success.

    True rather than None: every other function in this module returns None
    from safe_call() ONLY on failure, since a successful call always has a
    body. DELETE has no body, and returning None here would make that
    ordinary success indistinguishable from safe_call's failure sentinel.
    """
    _request("DELETE", f"/watchlist/{ticker}")
    return True


# ── Admin ───────────────────────────────────────────────
def health() -> dict[str, Any]:
    """Liveness and the state of every dependency, including the scheduler."""
    result: dict[str, Any] = _request("GET", "/health")
    return result


def budgets() -> list[dict[str, Any]]:
    """Today's usage against each provider's daily allowance."""
    result: list[dict[str, Any]] = _request("GET", "/admin/budgets")
    return result


def config() -> dict[str, Any]:
    """Effective configuration, secrets excluded."""
    result: dict[str, Any] = _request("GET", "/admin/config")
    return result
