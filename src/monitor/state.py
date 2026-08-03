# ═══════════════════════════════════════════════════════
# FinSight — Monitoring Graph State
# ═══════════════════════════════════════════════════════
#
# Purpose : The typed state that travels through Subsystem 2's StateGraph.
#
# Public API:
#   MonitorState        the graph state
#   WatchedTicker       one row of the watchlist, as it travels in state
#   MonitorInput        what Send() delivers to one monitor branch
#   CandidateAlert      what a monitor emits — pre-severity, pre-dedup
#   Alert               what survives dedup and gets persisted
#   SuppressionRecord   a candidate that did not survive, and why
#   DecisionRecord      every dedup decision, for the Phase 7 sweep
#   new_cycle_state(), new_cycle_id()
#
# ══ THE REDUCER RULE, AGAIN — AND HARDER ══
#   Subsystem 1 fans out over (agent x ticker). This one fans out over
#   (monitor x ticker) with a deliberate asymmetry: price and macro are
#   dispatched ONCE for the whole watchlist, filings and news once PER TICKER.
#   A five-ticker cycle is therefore 1 + 1 + 5 + 5 = 12 concurrent branches
#   rather than 20, and they all write `candidates` in the same superstep.
#
#   Three keys carry `Annotated[list, operator.add]`:
#
#       candidates  monitor_errors  api_calls
#
#   Everything downstream of the fan-in — fired, suppressed, merged — is
#   written by alert_synthesizer alone and must NOT have a reducer. A reducer
#   there would be silently wrong rather than loud: the synthesizer runs once,
#   so operator.add would never actually merge anything, and the mistake would
#   only surface if a second writer were ever added.
#
# ══ WHY CandidateAlert AND Alert ARE DIFFERENT TYPES ══
#   A candidate has no alert_id, no severity, no canonical text, and no
#   dedup decision. Collapsing them into one optional-field type would make
#   every field optional and let a half-built candidate reach the dispatcher.
#   The type boundary IS the pipeline stage boundary.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from src.core.schemas import AlertType, Citation, Severity, ToolCallRecord


class WatchedTicker(TypedDict):
    """One watched ticker as it travels through graph state."""

    ticker: str
    company_name: str
    warmed_up: bool


class MonitorInput(TypedDict, total=False):
    """
    The payload Send() delivers to ONE monitor branch.

    ``tickers`` is a list rather than a single symbol because the batched
    monitors receive the whole watchlist in one branch. The per-ticker monitors
    receive a one-element list, so the monitors share a signature and the
    asymmetry lives entirely in graph.py where it is visible.
    """

    tickers: list[str]
    companies: dict[str, str]
    since: dict[str, str]
    cycle_id: str
    warmup: bool


class CandidateAlert(TypedDict):
    """
    Something a monitor noticed. Not yet an alert.

    ``natural_key`` is the identity of the underlying EVENT, chosen per type so
    that the same event observed twice produces the same key:

        NEW_FILING      accession number       — globally unique, immutable
        NEWS_SENTIMENT  article id / url hash  — one per article
        MACRO_EVENT     series id + release date
        PRICE_MOVE      trading date + direction + magnitude bucket

    The bucket in PRICE_MOVE is what makes that key stable at all: the same
    day's move re-measured an hour later is a different percentage and the same
    event, so the key rounds before it hashes.
    """

    ticker: str
    company_name: str
    alert_type: AlertType
    monitor: str
    headline: str
    detail: str
    natural_key: str
    metrics: dict[str, Any]
    evidence: list[Citation]
    observed_at: str


class Alert(TypedDict):
    """
    A candidate that survived dedup, with everything needed to dispatch it.

    ``occurrence_count`` counts how many times this event has been SEEN, not
    how many times it was reported — the whole purpose of the engine is that
    those two numbers differ.
    """

    alert_id: str
    ticker: str
    company_name: str
    alert_type: AlertType
    severity: Severity
    status: str
    headline: str
    detail: str
    canonical_text: str
    dedup_key: str
    metrics: dict[str, Any]
    evidence: list[Citation]
    occurrence_count: int
    first_seen_at: str
    last_seen_at: str
    fired_at: str
    parent_alert_id: str | None


class SuppressionRecord(TypedDict):
    """
    A candidate that was not reported, and the alert it collapsed into.

    Surfaced in the cycle report and the dashboard rather than hidden. A dedup
    engine whose decisions are invisible is indistinguishable from a system
    that is dropping alerts through a bug.
    """

    ticker: str
    alert_type: AlertType
    headline: str
    canonical_text: str
    parent_alert_id: str
    parent_headline: str
    score: float
    reason: str


class DecisionRecord(TypedDict):
    """One dedup decision, shaped for repository.record_dedup_decisions."""

    ticker: str
    alert_type: str
    severity: str
    decision: str
    reason: str
    dedup_key: str
    candidate_text: str
    parent_alert_id: str | None
    parent_text: str
    score: float


class MonitorState(TypedDict, total=False):
    """
    State for the autonomous monitoring graph.

    ``total=False`` so nodes return partial updates, matching ResearchState.
    """

    # ── Inputs / cycle identity (single writer: load_watchlist) ──
    cycle_id: str
    started_at: str
    warmup: bool
    watchlist: list[WatchedTicker]
    # {"AAPL:filing": iso_timestamp} — flat string keys because state is
    # serialised into the checkpointer and tuple keys do not survive JSON.
    last_checked: dict[str, str]

    # ── PARALLEL FAN-IN — reducers REQUIRED ──
    # Up to (2 + 2N) branches write these in one superstep.
    candidates: Annotated[list[CandidateAlert], operator.add]
    monitor_errors: Annotated[list[str], operator.add]
    api_calls: Annotated[list[ToolCallRecord], operator.add]
    # "AAPL:filing" for every (ticker, monitor) that fetched CLEANLY. Collected
    # here rather than written by the monitor itself, because a watermark must
    # advance when the cycle becomes DURABLE, not when the fetch returns: a
    # crash between the two would move the watermark past events that were
    # never persisted, and they would never be looked for again.
    checked: Annotated[list[str], operator.add]

    # ── Dedup outcome (single writer: alert_synthesizer) — NO reducers ──
    fired: list[Alert]
    suppressed: list[SuppressionRecord]
    merged: list[Alert]
    decisions: list[DecisionRecord]

    # ── Phase 7 lands here: human_approval, dispatcher ──
    pending_approval: list[Alert]
    approval_decisions: dict[str, str]
    dispatched: list[str]


def new_cycle_id(*, now: datetime | None = None) -> str:
    """
    Build a cycle identifier that sorts chronologically as a string.

    Format ``20260803T120000Z-a1b2c3d4``. The timestamp prefix makes a
    directory listing or a `LIKE '2026-08%'` query useful; the random suffix
    keeps two cycles started in the same second distinct.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def new_cycle_state(
    watchlist: list[WatchedTicker],
    *,
    cycle_id: str = "",
    warmup: bool = False,
    last_checked: dict[str, str] | None = None,
) -> MonitorState:
    """
    Build the initial state for one monitoring cycle.

    The three reducer-backed lists are seeded empty so monitors can append
    without a first-write special case.

    Parameters
    ----------
    watchlist : list of WatchedTicker
        Tickers to check this cycle.
    cycle_id : str, optional
        Generated when omitted.
    warmup : bool, default False
        Observe-only: candidates are indexed for future dedup but nothing is
        reported.
    last_checked : dict, optional
        ``{"TICKER:monitor": iso}`` watermarks.

    Returns
    -------
    MonitorState
    """
    return MonitorState(
        cycle_id=cycle_id or new_cycle_id(),
        started_at=datetime.now(timezone.utc).isoformat(),
        warmup=warmup,
        watchlist=watchlist,
        last_checked=last_checked or {},
        candidates=[],
        monitor_errors=[],
        api_calls=[],
        checked=[],
        fired=[],
        suppressed=[],
        merged=[],
        decisions=[],
        pending_approval=[],
        approval_decisions={},
        dispatched=[],
    )
