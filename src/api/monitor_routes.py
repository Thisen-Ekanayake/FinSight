# ═══════════════════════════════════════════════════════
# FinSight — Monitoring Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : HTTP surface for Subsystem 2.
#
# Routes:
#   POST /monitor/cycles            run one cycle now
#   GET  /monitor/cycles            recent cycles
#   GET  /monitor/cycles/{id}       one cycle
#   GET  /monitor/alerts            fired alerts
#   GET  /monitor/alerts/{id}       one alert
#   GET  /monitor/decisions         every dedup decision, with its score
#
# ══ /decisions EXISTS ON PURPOSE ══
#   Most projects hide their cleverest logic behind a number. A dedup engine
#   whose decisions are invisible is indistinguishable from one that is
#   dropping alerts through a bug — and it is Phase 7's threshold-sweep input,
#   which cannot be reconstructed after the fact because a parent's vector
#   drifts as its centroid updates.
#
# ══ WHY THE CYCLE RUNS IN A THREAD ══
#   The graph is synchronous: it embeds, calls Qdrant, and hits four data
#   providers, all blocking. Awaiting it directly on the event loop would
#   stall every other request for the length of a cycle.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import (
    AlertOut,
    CitationOut,
    CycleOut,
    CycleRunRequest,
    CycleRunResponse,
    DedupDecisionOut,
    SuppressionOut,
)
from src.persistence.repository import AlertRow, get_alert, get_cycle, list_alerts, list_cycles, list_dedup_decisions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitor", tags=["monitoring"])


def _alert_out(alert: dict[str, Any]) -> AlertOut:
    """Shape an Alert (from state) or an AlertRow (from SQLite) for the wire."""
    return AlertOut(
        alert_id=alert["alert_id"],
        cycle_id=alert.get("cycle_id", ""),
        ticker=alert.get("ticker", ""),
        alert_type=alert.get("alert_type", ""),
        severity=alert.get("severity", "LOW"),
        status=alert.get("status", "FIRED"),
        headline=alert.get("headline", ""),
        detail=alert.get("detail", ""),
        canonical_text=alert.get("canonical_text", ""),
        occurrence_count=int(alert.get("occurrence_count", 1)),
        first_seen_at=alert.get("first_seen_at", ""),
        last_seen_at=alert.get("last_seen_at", ""),
        fired_at=alert.get("fired_at", ""),
        parent_alert_id=alert.get("parent_alert_id"),
        evidence=[
            CitationOut(
                source_type=citation.get("source_type", ""),
                source_id=citation.get("source_id", ""),
                url=citation.get("url", ""),
                as_of=citation.get("as_of", ""),
                excerpt=citation.get("excerpt"),
            )
            for citation in alert.get("evidence", [])
        ],
    )


def _row_out(row: AlertRow) -> AlertOut:
    return _alert_out(dict(row))


# ── Cycles ──────────────────────────────────────────────
@router.post("/cycles", response_model=CycleRunResponse, summary="Run one monitoring cycle")
async def run_monitor_cycle(request: CycleRunRequest, http_request: Request) -> CycleRunResponse:
    """
    Run a cycle now and report what it found — and what it chose not to say.

    Run once with ``warmup: true`` before the first real cycle. A cold dedup
    index has nothing to match against, so cycle 1 would otherwise report every
    open filing, every recent article, and every price move in one burst.
    """
    from src.monitor.graph import run_cycle

    checkpointer = getattr(http_request.app.state, "checkpointer", None)

    started = time.monotonic()
    try:
        # to_thread because the graph blocks: embeddings, Qdrant, and four
        # data providers. Awaiting it inline would stall the whole event loop.
        state = await asyncio.to_thread(
            run_cycle,
            tickers=request.tickers or None,
            warmup=request.warmup,
            checkpointer=checkpointer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Monitoring cycle failed")
        raise HTTPException(status_code=500, detail=f"Cycle failed: {type(exc).__name__}: {exc}") from exc

    elapsed = int((time.monotonic() - started) * 1000)

    return CycleRunResponse(
        cycle_id=state.get("cycle_id", ""),
        warmup=bool(state.get("warmup")),
        candidate_count=len(state.get("candidates") or []),
        fired=[_alert_out(dict(a)) for a in state.get("fired") or []],
        merged=[_alert_out(dict(a)) for a in state.get("merged") or []],
        suppressed=[
            SuppressionOut(
                ticker=record["ticker"],
                alert_type=record["alert_type"],
                headline=record["headline"],
                parent_alert_id=record["parent_alert_id"],
                parent_headline=record["parent_headline"],
                score=record["score"],
                reason=record["reason"],
            )
            for record in state.get("suppressed") or []
        ],
        errors=list(state.get("monitor_errors") or []),
        duration_ms=elapsed,
    )


@router.get("/cycles", response_model=list[CycleOut], summary="Recent cycles")
async def get_cycles(limit: int = Query(default=25, ge=1, le=200)) -> list[CycleOut]:
    """List recent cycles, newest first."""
    return [CycleOut(**row) for row in list_cycles(limit=limit)]


@router.get("/cycles/{cycle_id}", response_model=CycleOut, summary="One cycle")
async def get_one_cycle(cycle_id: str) -> CycleOut:
    """
    Return one cycle by id.

    ``cycle_id`` is also the checkpointer thread id as ``monitor:{cycle_id}``,
    so a row here is the entry point into the cycle's full replayable state —
    the same relationship a research run has with its thread.
    """
    row = get_cycle(cycle_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No cycle {cycle_id}")
    return CycleOut(**row)


# ── Alerts ──────────────────────────────────────────────
@router.get("/alerts", response_model=list[AlertOut], summary="Fired alerts")
async def get_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    ticker: str | None = None,
    severity: str | None = Query(default=None, pattern="^(?i)(LOW|MED|HIGH)$"),
    status: str | None = None,
) -> list[AlertOut]:
    """
    List fired alerts, newest first.

    ``occurrence_count`` is how many times the underlying event has been SEEN,
    not how many times it was reported. The whole purpose of the engine is
    that those two numbers differ.
    """
    return [_row_out(row) for row in list_alerts(limit=limit, ticker=ticker, severity=severity, status=status)]


@router.get("/alerts/{alert_id}", response_model=AlertOut, summary="One alert")
async def get_one_alert(alert_id: str) -> AlertOut:
    """Return one alert by id."""
    row = get_alert(alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No alert {alert_id}")
    return _row_out(row)


# ── Dedup decisions ─────────────────────────────────────
@router.get("/decisions", response_model=list[DedupDecisionOut], summary="Dedup decisions and their scores")
async def get_decisions(
    limit: int = Query(default=100, ge=1, le=1000),
    decision: str | None = Query(default=None, description="FIRE, SUPPRESS_EXACT, SUPPRESS_SEMANTIC, MERGE, ..."),
) -> list[DedupDecisionOut]:
    """
    Every dedup decision, with the similarity score that produced it.

    FIREs are included, not only suppressions. A log of only the suppressions
    is a sample with no negatives — it can justify the threshold that produced
    it and nothing else, which is exactly the mistake this endpoint exists to
    let Phase 7 avoid.
    """
    return [DedupDecisionOut(**row) for row in list_dedup_decisions(limit=limit, decision=decision)]
