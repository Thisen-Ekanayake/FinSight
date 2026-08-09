# ═══════════════════════════════════════════════════════
# FinSight — Research Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : HTTP surface for Subsystem 1.
#
# Routes:
#   POST /research/query               ask a question, get a verified answer
#   POST /research/query/stream        the same, streamed as SSE
#   GET  /research/threads/{thread_id} replay a run's audit trail
#   GET  /research/runs                recent runs
#
# ══ THE AUDIT TRAIL IS REPLAYED, NOT LOGGED ══
#   /threads/{id} reads the checkpointer's state history — the actual
#   intermediate states the graph passed through, recovered from disk. It is
#   not a log written alongside the run, so it cannot drift from what happened
#   and it survives the process that produced it.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    CitationOut,
    ConflictOut,
    QueryRequest,
    QueryResponse,
    RunSummaryOut,
    SeriesPointOut,
    ThreadResponse,
    ThreadStep,
    ToolCallOut,
    UnsupportedClaimOut,
    VerificationOut,
)
from src.persistence.repository import (
    ResearchRunSummary,
    get_research_run,
    list_research_runs,
    record_research_run,
)
from src.research.config import RECURSION_LIMIT
from src.research.state import new_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

# An empty report, so a run that died before verification still serialises
# rather than 500-ing on a missing key.
_EMPTY_VERIFICATION = {
    "verified_claims": [],
    "unsupported_claims": [],
    "invalid_source_ids": [],
    "citation_coverage": 0.0,
    "passed": False,
}


def _graph(request: Request) -> Any:
    """Fetch the compiled graph built during lifespan."""
    graph = getattr(request.app.state, "research_graph", None)
    if graph is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(status_code=503, detail="Research graph is not ready")
    return graph


def _new_thread_id() -> str:
    """
    Mint a thread id.

    Prefixed by subsystem so one checkpoint database can hold research threads
    and Phase 7's ``monitor:{cycle_id}`` threads without either having to
    guess what the other's ids look like.
    """
    return f"research:{uuid.uuid4().hex[:16]}"


def _verification_out(report: dict[str, Any]) -> VerificationOut:
    """Translate a verification report into its API shape."""
    return VerificationOut(
        citation_coverage=report.get("citation_coverage", 0.0),
        passed=report.get("passed", False),
        verified_count=len(report.get("verified_claims", [])),
        unsupported_claims=[
            UnsupportedClaimOut(
                claim=claim.get("claim", ""),
                reason=claim.get("reason", ""),
                origin_agent=claim.get("origin_agent", ""),
                ticker=claim.get("ticker", ""),
            )
            for claim in report.get("unsupported_claims", [])
        ],
        invalid_source_ids=list(report.get("invalid_source_ids", [])),
    )


def _unique_citations(citations: list[dict[str, Any]]) -> list[CitationOut]:
    """Deduplicate citations by (type, id) — one filing cited eight times is one source."""
    seen: set[tuple[str, str]] = set()
    out: list[CitationOut] = []

    for citation in citations:
        key = (citation["source_type"], citation["source_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(CitationOut(**citation))

    return out


def _series(findings: list[dict[str, Any]]) -> list[SeriesPointOut]:
    """
    Pull the chartable points out of a run's findings.

    A finding qualifies only if it has a ticker, a genuinely numeric value, and
    a reporting period. Everything else is dropped rather than coerced —
    plotting a qualitative finding by forcing it onto an axis is precisely what
    src/viz/ambient.ts refuses to do, and the same rule applies here.

    ``period`` is a recent addition to AgentFinding, so findings replayed from
    a checkpoint written before it have no such key. The metric name has
    carried the period as a "name@period" suffix all along, so that is the
    fallback. It is read only — nothing here writes that format back, and the
    aggregator's key stays the aggregator's business.
    """
    points: list[SeriesPointOut] = []

    for item in findings:
        if item.get("error"):
            continue

        value = item.get("value")
        # bool is an int in Python. Left alone it would plot as a silent 0/1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue

        ticker = item.get("ticker")
        metric_key = item.get("metric")
        if not ticker or not metric_key:
            continue

        # "gross_margin@FY2025" -> ("gross_margin", "FY2025"). An explicit
        # period wins; the suffix is only consulted when there is none.
        name, _, suffix = metric_key.partition("@")
        period = item.get("period") or suffix
        if not period:
            continue

        citations = item.get("citations") or []
        points.append(
            SeriesPointOut(
                ticker=ticker,
                metric=name,
                period=period,
                value=float(value),
                unit=item.get("unit"),
                provider=citations[0].get("source_type") if citations else None,
                confidence=float(item.get("confidence", 1.0)),
            )
        )

    # Sorted so a line renders in period order without the client re-deriving
    # it, and so two identical runs produce byte-identical responses.
    points.sort(key=lambda p: (p.ticker, p.metric, p.period))
    return points


def _to_response(state: dict[str, Any], *, thread_id: str, latency_ms: int) -> QueryResponse:
    """Translate a final ResearchState into the API's response shape."""
    plan = state.get("plan") or {}
    report = state.get("verification") or _EMPTY_VERIFICATION

    return QueryResponse(
        thread_id=thread_id,
        query=state.get("query", ""),
        answer=state.get("final_answer") or state.get("draft_answer") or "",
        citations=_unique_citations(state.get("citations", [])),
        conflicts=[ConflictOut(**conflict) for conflict in state.get("conflicts", [])],
        verification=_verification_out(report),
        agents_used=list(plan.get("selected_agents", [])),
        tickers=list(plan.get("tickers", [])),
        branch_count=len(state.get("tool_calls", [])),
        repair_count=state.get("repair_count", 0),
        errors=list(state.get("errors", [])),
        latency_ms=latency_ms,
        series=_series(state.get("findings", [])),
    )


@router.post("/query", response_model=QueryResponse, summary="Ask a research question")
async def run_query(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Run one research query end to end and return the verified answer.

    Synchronous from the client's point of view: routing, fan-out, synthesis,
    verification, and up to one repair pass all complete before the response.
    Expect tens of seconds — the graph makes several real LLM and data-source
    calls. Use ``/query/stream`` to watch it progress.
    """
    graph = _graph(request)
    thread_id = body.thread_id or _new_thread_id()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    started = time.monotonic()
    try:
        state = await graph.ainvoke(new_state(body.query, thread_id=thread_id), config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research query failed: %s", body.query)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    latency_ms = int((time.monotonic() - started) * 1000)
    record_research_run(state, thread_id=thread_id, latency_ms=latency_ms)

    return _to_response(state, thread_id=thread_id, latency_ms=latency_ms)


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _stream_events(graph: Any, query: str, thread_id: str) -> AsyncIterator[str]:
    """
    Yield SSE frames as the graph runs.

    Streams both modes at once. ``updates`` carries each node's partial state
    as it completes, which is what makes the fan-out visible; ``values`` is the
    merged state after each superstep and is where the final answer comes
    from. The final state must NOT be assembled by merging the update frames —
    doing that overwrites the reducer-backed keys instead of concatenating
    them, and the audit trail ends up showing only the last branch.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    started = time.monotonic()
    final: dict[str, Any] = {}

    yield _sse("start", {"thread_id": thread_id, "query": query})

    try:
        async for mode, chunk in graph.astream(
            new_state(query, thread_id=thread_id),
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final = chunk
                continue

            for node, update in chunk.items():
                yield _sse(
                    "node",
                    {
                        "node": node,
                        "findings": len(update.get("findings", [])),
                        "errors": update.get("errors", []),
                        "plan": update.get("plan"),
                        "repairs": len(update.get("repair_targets", [])),
                    },
                )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streamed research query failed: %s", query)
        yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
        return

    latency_ms = int((time.monotonic() - started) * 1000)
    record_research_run(final, thread_id=thread_id, latency_ms=latency_ms)
    yield _sse("final", _to_response(final, thread_id=thread_id, latency_ms=latency_ms).model_dump())


@router.post("/query/stream", summary="Ask a research question, streamed")
async def run_query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """
    Run a query and stream node-by-node progress as Server-Sent Events.

    Frames: ``start``, one ``node`` per completed node, then ``final`` (or
    ``error``). The node frames are what make a four-branch fan-out legible
    while it is happening rather than after.
    """
    graph = _graph(request)
    thread_id = body.thread_id or _new_thread_id()

    return StreamingResponse(
        _stream_events(graph, body.query, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _pending_nodes(snapshot: Any) -> list[str]:
    """
    Names of the nodes a snapshot is about to execute.

    ══ WHERE NODE IDENTITY ACTUALLY LIVES ══
      Not in ``metadata["writes"]`` — LangGraph 1.2's CheckpointMetadata holds
      only ``source``, ``step``, and ``parents``. The node names are in
      ``snapshot.tasks``, and they describe the step AHEAD of the snapshot, not
      the one behind it. So the nodes that produced a given state are the
      PREVIOUS snapshot's pending set, which is how get_thread pairs them up.

      Duplicates are kept deliberately. Two tasks both named "fundamentals" is
      one Send per ticker in a single superstep — the repetition is the
      evidence that the fan-out was parallel, and deduplicating would erase
      exactly what the audit trail exists to show.
    """
    tasks = getattr(snapshot, "tasks", None) or ()
    names = [task.name for task in tasks if getattr(task, "name", None)]
    return names or list(snapshot.next or ())


def _summary_out(summary: ResearchRunSummary) -> RunSummaryOut:
    """Translate a repository summary into its API shape."""
    return RunSummaryOut(
        thread_id=summary["thread_id"],
        query=summary["query"],
        citation_coverage=summary["citation_coverage"],
        verification_passed=summary["verification_passed"],
        repair_count=summary["repair_count"],
        agents_used=summary["agents_used"],
        tickers=summary["tickers"],
        latency_ms=summary["latency_ms"],
        created_at=summary["created_at"],
    )


@router.get("/threads/{thread_id}", response_model=ThreadResponse, summary="Replay a run's audit trail")
async def get_thread(request: Request, thread_id: str) -> ThreadResponse:
    """
    Replay every superstep the graph passed through for one thread.

    Read back from the checkpointer, so this is what actually happened rather
    than what a logger recorded at the time. Each step names the nodes that
    wrote in it — several names in one step is a parallel fan-out, and that is
    the clearest evidence the branches really did run together.
    """
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}

    snapshots = [snapshot async for snapshot in graph.aget_state_history(config)]
    if not snapshots:
        raise HTTPException(status_code=404, detail=f"No checkpoint history for thread {thread_id!r}")

    # History arrives newest-first; an audit trail reads forwards.
    snapshots.reverse()

    steps: list[ThreadStep] = []
    previous_pending: list[str] = []

    for snapshot in snapshots:
        metadata = snapshot.metadata or {}
        values = snapshot.values or {}
        pending = _pending_nodes(snapshot)

        steps.append(
            ThreadStep(
                step=int(metadata.get("step", -1)),
                # A snapshot's tasks are the nodes ABOUT to run, so the nodes
                # that produced THIS state are the previous snapshot's pending
                # set. See _pending_nodes.
                nodes=sorted(previous_pending),
                next=pending,
                findings_total=len(values.get("findings", [])),
                citations_total=len(values.get("citations", [])),
                created_at=getattr(snapshot, "created_at", None),
            )
        )
        previous_pending = pending

    latest = snapshots[-1].values or {}
    report = latest.get("verification")
    summary = get_research_run(thread_id)

    return ThreadResponse(
        thread_id=thread_id,
        query=latest.get("query", ""),
        answer=latest.get("final_answer") or latest.get("draft_answer") or "",
        summary=_summary_out(summary) if summary else None,
        verification=_verification_out(report) if report else None,
        tool_calls=[ToolCallOut(**call) for call in latest.get("tool_calls", [])],
        steps=steps,
    )


@router.get("/runs", response_model=list[RunSummaryOut], summary="List recent runs")
async def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> list[RunSummaryOut]:
    """Return recent research runs, newest first."""
    return [_summary_out(summary) for summary in list_research_runs(limit=limit)]
