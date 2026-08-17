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
# ══ HISTORY IS PER ACCOUNT ══
#   A question, its answer and the thread id that replays it belong to whoever
#   asked. /runs lists only the caller's; /threads/{id} refuses anyone else's.
#   The refusal is a 404 with the same wording an unknown thread gets, so the
#   endpoint is not an oracle for which ids exist.
#
#   The checkpointer cannot help enforce this — it has no identity, only
#   thread ids — so research_runs.subject is the only record of ownership and
#   can_use_thread is the only gate. See _resolve_thread for the matching
#   rule on the write side.
#
# ══ THE AUDIT TRAIL IS REPLAYED, NOT LOGGED ══
#   /threads/{id} reads the checkpointer's state history — the actual
#   intermediate states the graph passed through, recovered from disk. It is
#   not a log written alongside the run, so it cannot drift from what happened
#   and it survives the process that produced it.
#
# ══ THE TWO QUERY ROUTES ARE THE ONLY METERED ONES ══
#   They are what costs real money — several LLM calls and a fan-out of data
#   -source requests each. Replaying a thread or listing runs is a SQLite read,
#   so a free account keeps full use of the dashboard and only the expensive
#   verb is counted. See src/api/quota.py.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.api.auth import CurrentIdentity, Identity
from src.api.quota import charge_query, refund_query
from src.api.schemas import (
    CitationOut,
    ConflictOut,
    QueryRequest,
    QueryResponse,
    QuotaExceededBody,
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
    can_use_thread,
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


def _resolve_thread(requested: str | None, identity: Identity) -> str:
    """
    Turn a client-supplied thread id into one this caller may write to.

    ══ THE HOLE THIS CLOSES ══
      thread_id arrives in the request BODY. Before this, supplying somebody
      else's appended to THEIR checkpoint history and overwrote THEIR
      research_runs row in place — a write primitive against another account,
      reachable by anyone who had read /research/runs once while it was still
      global. Scoping the reads without this would be theatre: hijack the row,
      become its owner, then read the transcript back through the front door.

      So a supplied id must already resolve to a thread this caller can see.
      Minting a new one is the SERVER's job — omit the field and get one.
      Nothing that ships sends this field, so the rule costs nothing today.

    404 rather than 403, with the same detail an unknown thread gets: a 403
    would confirm the id exists and belongs to someone else, and there is
    nothing a legitimate caller could do with that.
    """
    if not requested:
        return _new_thread_id()

    if not can_use_thread(requested, subject=identity.subject, unlimited=identity.unlimited):
        logger.info("Refused thread %s for %s — not theirs", requested, identity.email)
        raise HTTPException(status_code=404, detail=f"No checkpoint history for thread {requested!r}")

    return requested


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


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={402: {"model": QuotaExceededBody, "description": "Free queries exhausted"}},
    summary="Ask a research question",
)
async def run_query(request: Request, body: QueryRequest, identity: CurrentIdentity) -> QueryResponse:
    """
    Run one research query end to end and return the verified answer.

    Synchronous from the client's point of view: routing, fan-out, synthesis,
    verification, and up to one repair pass all complete before the response.
    Expect tens of seconds — the graph makes several real LLM and data-source
    calls. Use ``/query/stream`` to watch it progress.

    Costs one free query for an account outside the unlimited tier, refunded
    if the run fails.
    """
    # ══ WHY THE CHARGE IS HERE AND NOT IN A Depends ══
    #   FastAPI resolves dependencies BEFORE it validates the request body, so
    #   a Depends(charge_query) would bill a query that then 422s on
    #   QueryRequest's min_length. Charging in the handler puts the meter after
    #   validation. And it goes after _graph(), because a 503 from a lifespan
    #   that never ran is our failure, not a purchase.
    #
    #   _resolve_thread goes BEFORE the meter for the same reason: a refused
    #   thread id is the caller's mistake and the graph never runs, so a free
    #   account must not be billed for it.
    graph = _graph(request)
    thread_id = _resolve_thread(body.thread_id, identity)
    charge_query(identity)

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    started = time.monotonic()
    try:
        state = await graph.ainvoke(new_state(body.query, thread_id=thread_id), config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research query failed: %s", body.query)
        refund_query(identity)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    latency_ms = int((time.monotonic() - started) * 1000)
    record_research_run(
        state,
        thread_id=thread_id,
        subject=identity.subject,
        email=identity.email,
        latency_ms=latency_ms,
    )

    return _to_response(state, thread_id=thread_id, latency_ms=latency_ms)


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


async def _stream_events(graph: Any, query: str, thread_id: str, *, identity: Identity) -> AsyncIterator[str]:
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
        # Only reached for a real failure. A client that abandons the stream
        # raises CancelledError, which is not an Exception subclass here and
        # so is deliberately NOT refunded — Ask.tsx aborts on every resubmit,
        # and refunding that would make retyping free forever.
        refund_query(identity)
        yield _sse("error", {"detail": f"{type(exc).__name__}: {exc}"})
        return

    latency_ms = int((time.monotonic() - started) * 1000)
    record_research_run(
        final,
        thread_id=thread_id,
        subject=identity.subject,
        email=identity.email,
        latency_ms=latency_ms,
    )
    yield _sse("final", _to_response(final, thread_id=thread_id, latency_ms=latency_ms).model_dump())


@router.post(
    "/query/stream",
    responses={402: {"model": QuotaExceededBody, "description": "Free queries exhausted"}},
    summary="Ask a research question, streamed",
)
async def run_query_stream(request: Request, body: QueryRequest, identity: CurrentIdentity) -> StreamingResponse:
    """
    Run a query and stream node-by-node progress as Server-Sent Events.

    Frames: ``start``, one ``node`` per completed node, then ``final`` (or
    ``error``). The node frames are what make a four-branch fan-out legible
    while it is happening rather than after.

    Costs one free query for an account outside the unlimited tier, refunded
    if the run fails.
    """
    graph = _graph(request)

    # ══ BOTH OF THESE MUST HAPPEN HERE, NOT IN THE GENERATOR ══
    #   _stream_events does not run until the response body is being written,
    #   by which point the status line is already 200 and the only way to
    #   report a refusal is an `error` frame — indistinguishable to the client
    #   from a graph crash, and invisible to anything that checks status codes.
    #   Raising before StreamingResponse is constructed is what makes an
    #   exhausted account get a real 402 and a hijacked thread a real 404.
    #
    #   Order matters between them too: refusing a foreign thread must not
    #   cost the caller a query.
    thread_id = _resolve_thread(body.thread_id, identity)
    charge_query(identity)

    return StreamingResponse(
        _stream_events(graph, body.query, thread_id, identity=identity),
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
async def get_thread(request: Request, thread_id: str, identity: CurrentIdentity) -> ThreadResponse:
    """
    Replay every superstep the graph passed through for one of your threads.

    Read back from the checkpointer, so this is what actually happened rather
    than what a logger recorded at the time. Each step names the nodes that
    wrote in it — several names in one step is a parallel fan-out, and that is
    the clearest evidence the branches really did run together.
    """
    graph = _graph(request)

    # ══ THE CHECKPOINTER CANNOT DO THIS FOR US ══
    #   Its tables are LangGraph's, keyed by thread_id alone with no identity
    #   anywhere in them. An owner column on research_runs secures the
    #   LISTING; only this check secures the TRANSCRIPT — and it has to happen
    #   before aget_state_history rather than after, because by then the
    #   question text and the full answer are already in hand.
    if not can_use_thread(thread_id, subject=identity.subject, unlimited=identity.unlimited):
        logger.info("Refused thread %s for %s — not theirs", thread_id, identity.email)
        # Same status and byte-identical detail to the unknown-thread 404
        # below. A 403 here would confirm the thread exists, which is the one
        # fact worth withholding.
        raise HTTPException(status_code=404, detail=f"No checkpoint history for thread {thread_id!r}")

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
    summary = get_research_run(thread_id, subject=identity.subject, unlimited=identity.unlimited)

    return ThreadResponse(
        thread_id=thread_id,
        query=latest.get("query", ""),
        answer=latest.get("final_answer") or latest.get("draft_answer") or "",
        summary=_summary_out(summary) if summary else None,
        verification=_verification_out(report) if report else None,
        tool_calls=[ToolCallOut(**call) for call in latest.get("tool_calls", [])],
        steps=steps,
    )


@router.get("/runs", response_model=list[RunSummaryOut], summary="List your recent runs")
async def list_runs(identity: CurrentIdentity, limit: int = Query(default=50, ge=1, le=200)) -> list[RunSummaryOut]:
    """
    Return this account's recent research runs, newest first.

    Scoped to the caller: a question, its answer, and the thread id that
    replays it are all one account's business.
    """
    return [
        _summary_out(summary)
        for summary in list_research_runs(subject=identity.subject, unlimited=identity.unlimited, limit=limit)
    ]
