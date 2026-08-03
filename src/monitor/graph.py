# ═══════════════════════════════════════════════════════
# FinSight — Monitoring Graph
# ═══════════════════════════════════════════════════════
#
# Purpose : Wire the watchlist loader, four monitors, the alert synthesizer,
#           and the cycle persister into the second StateGraph.
#
# Public API:
#   monitor_fanout(state)       the conditional edge that spawns branches
#   alert_synthesizer_node(state)
#   human_approval_node(state)  interrupt() gate for HIGH alerts (Phase 7)
#   dispatcher_node(state)      resolve approvals, deliver everything else
#   persist_cycle_node(state)
#   build_monitor_graph(checkpointer=None)
#   run_cycle(...)              convenience wrapper — one cycle, start to finish or to a pause
#   resume_cycle(...)           continue a paused cycle with a human's decisions
#
# ══ THE ASYMMETRIC FAN-OUT ══
#   Subsystem 1 fans out over (agent x ticker), uniformly. This one does not,
#   and the difference is the rate-limit strategy expressed as topology:
#
#     price_monitor    ONE Send  for the whole watchlist   (yfinance batches)
#     macro_monitor    ONE Send  for the whole watchlist   (series are global)
#     filing_monitor   ONE Send  PER TICKER                (EDGAR is per-CIK)
#     news_monitor     ONE Send  PER TICKER                (Finnhub is per-symbol)
#
#   Five tickers is 1 + 1 + 5 + 5 = 12 branches, not 20. Ten tickers is 22, not
#   40. Writing this as four uniform per-ticker Sends would have cost ten
#   identical price requests and five identical CPI requests per cycle, and it
#   would have looked perfectly reasonable in the code.
#
# ══ PHASE 7: THE APPROVAL GATE IS DURABLE, NOT IN-PROCESS ══
#   human_approval_node calls interrupt() only when there is a HIGH alert to
#   decide — a pass-through otherwise, so a cycle with nothing HIGH never
#   pauses. When it DOES pause, graph.invoke() returns immediately with a
#   `__interrupt__` key in the result rather than blocking; the graph's state
#   is durably checkpointed at that point, so the process can restart, and a
#   LATER call to resume_cycle() with a human's decisions continues the SAME
#   run from exactly where it paused. This is why run_cycle now REQUIRES a
#   checkpointer rather than treating one as optional: without one, interrupt()
#   still "works" in the sense of not raising, but the pause it produces
#   cannot be resumed — a silent trap for a HIGH alert that quietly never
#   reaches a reader. See src/persistence/checkpointer.py.
#
# Usage:
#   ./run_monitor.sh --once --warmup
#   ./run_monitor.sh --once
#   ./run_monitor.sh --pending
#   ./run_monitor.sh --resume <cycle_id> --approve <alert_id> --reject <alert_id>
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time
from typing import Any

from src.monitor.config import DIAGNOSTIC_CYCLES, MAX_CANDIDATES_PER_CYCLE, MONITOR_NAMES
from src.monitor.monitors import MONITOR_NODES
from src.monitor.state import Alert, MonitorState, SuppressionRecord, new_cycle_state
from src.monitor.watchlist import load_watchlist_node, lookback_for

logger = logging.getLogger(__name__)

# Monitors that take the whole watchlist in one branch.
BATCHED_MONITORS: frozenset[str] = frozenset({"price_monitor", "macro_monitor"})


def monitor_fanout(state: MonitorState) -> list[Any]:
    """
    Conditional edge: expand the watchlist into concrete monitor branches.

    Parameters
    ----------
    state : MonitorState
        Must contain ``watchlist`` and ``last_checked``.

    Returns
    -------
    list of Send
        Empty only when the watchlist is empty, which is logged upstream.
    """
    from langgraph.types import Send

    watchlist = state.get("watchlist") or []
    if not watchlist:
        return []

    tickers = [item["ticker"] for item in watchlist]
    companies = {item["ticker"]: item["company_name"] for item in watchlist}
    last_checked = state.get("last_checked") or {}
    warmup = bool(state.get("warmup"))
    cycle_id = state.get("cycle_id", "")

    sends: list[Any] = []

    for name in MONITOR_NAMES:
        if name in BATCHED_MONITORS:
            # One request covers every symbol — or, for macro, no symbol at all.
            sends.append(
                Send(
                    name,
                    {
                        "tickers": tickers,
                        "companies": companies,
                        "since": {t: lookback_for(t, name, last_checked).isoformat() for t in tickers},
                        "cycle_id": cycle_id,
                        "warmup": warmup,
                    },
                )
            )
            continue

        for ticker in tickers:
            sends.append(
                Send(
                    name,
                    {
                        "tickers": [ticker],
                        "companies": {ticker: companies[ticker]},
                        "since": {ticker: lookback_for(ticker, name, last_checked).isoformat()},
                        "cycle_id": cycle_id,
                        "warmup": warmup,
                    },
                )
            )

    logger.info(
        "Fan-out: %d branches over %d tickers (%d batched, %d per-ticker)",
        len(sends),
        len(tickers),
        len(BATCHED_MONITORS),
        len(sends) - len(BATCHED_MONITORS),
    )
    return sends


def alert_synthesizer_node(state: MonitorState) -> dict:
    """
    Graph node: score, canonicalize, and deduplicate this cycle's candidates.

    Everything downstream of the fan-in is written here and only here, so none
    of these keys carries a reducer.

    Returns a partial state with ``fired``, ``suppressed``, ``merged``,
    ``decisions``, and ``pending_approval``.
    """
    from src.monitor.dedup import Decision, deduplicate

    candidates = state.get("candidates") or []
    if not candidates:
        logger.info("Cycle %s: no candidates", state.get("cycle_id", "?"))
        return {"fired": [], "suppressed": [], "merged": [], "decisions": [], "pending_approval": []}

    if len(candidates) > MAX_CANDIDATES_PER_CYCLE:
        # A data-source glitch that marks every bar as a 40% move must not turn
        # into hundreds of embeddings and hundreds of alerts. Truncating loudly
        # beats discovering it from the bill.
        logger.error(
            "Cycle %s produced %d candidates, capping at %d — check the monitors for a data fault",
            state.get("cycle_id", "?"),
            len(candidates),
            MAX_CANDIDATES_PER_CYCLE,
        )
        candidates = candidates[:MAX_CANDIDATES_PER_CYCLE]

    outcomes = deduplicate(
        candidates,
        warmup=bool(state.get("warmup")),
        log_distribution=_should_log_distribution(),
    )

    fired: list[Alert] = []
    merged: list[Alert] = []
    suppressed: list[SuppressionRecord] = []

    for outcome in outcomes:
        if outcome.decision == Decision.ESCALATE and outcome.alert:
            merged.append(outcome.alert)
        elif outcome.alert:
            fired.append(outcome.alert)
        elif outcome.suppression:
            suppressed.append(outcome.suppression)

    # An escalation IS a report — it reached a reader because its severity rose
    # above its parent's. It is tracked separately so the cycle summary can say
    # "2 new, 1 escalated" rather than blurring them into one number.
    reportable = fired + merged

    return {
        "fired": fired,
        "suppressed": suppressed,
        "merged": merged,
        "decisions": [outcome.record for outcome in outcomes],
        # Phase 7 gates on this. Populated now so the state shape is stable and
        # the approval node is a pure addition.
        "pending_approval": [alert for alert in reportable if alert["severity"] == "HIGH"],
    }


def human_approval_node(state: MonitorState) -> dict:
    """
    Graph node: pause for a human decision on every HIGH-severity finding.

    A pure pass-through when ``pending_approval`` is empty — interrupt() is
    never called, so a cycle with only LOW/MED findings never pauses at all.
    When it is not empty, the first execution raises GraphInterrupt and
    graph.invoke() returns immediately with a ``__interrupt__`` key; nothing
    past this point in the graph has run.

    ══ RE-EXECUTION ON RESUME ══
    LangGraph re-runs this node FROM THE TOP when Command(resume=...) is
    sent — see langgraph.types.interrupt. Everything above the interrupt()
    call is therefore recomputed, not reused. That is safe here because it is
    pure: ``pending`` is read straight out of state, which alert_synthesizer
    already wrote and this node never mutates.

    Returns
    -------
    dict
        ``approval_decisions``: ``{alert_id: "approve" | "reject"}``, empty
        when there was nothing to decide.
    """
    from langgraph.types import interrupt

    pending = state.get("pending_approval") or []
    if not pending:
        return {"approval_decisions": {}}

    decisions = interrupt(
        {
            "cycle_id": state.get("cycle_id", ""),
            "pending_approval": [dict(alert) for alert in pending],
        }
    )
    return {"approval_decisions": dict(decisions or {})}


def dispatcher_node(state: MonitorState) -> dict:
    """
    Graph node: resolve every approval decision, then deliver whatever fired.

    ══ WHERE PENDING_APPROVAL BECOMES FIRED OR REJECTED ══
    A LOW/MED alert already carries status FIRED — dedup.py set it at index
    time, and it is delivered unconditionally. A HIGH alert carries
    PENDING_APPROVAL, and approval_decisions says what was decided: this is
    where that decision is written back to the alert (for persist_cycle_node,
    downstream) and to its Qdrant point (via update_status, here — the SQLite
    row does not exist yet, so there is nothing there to update until persist
    runs). An alert with no entry in approval_decisions is treated as
    rejected rather than fired: the asymmetric-cost stance the rest of this
    engine takes is about missing a report, not about withholding one nobody
    explicitly approved.

    Returns
    -------
    dict
        ``fired``, ``merged`` with resolved statuses, and ``dispatched`` —
        the alert ids that at least one notification sink actually delivered.
    """
    from src.monitor.alert_store import update_status
    from src.monitor.config import REJECTED_STATUS
    from src.monitor.dispatch import dispatch_alert

    decisions = state.get("approval_decisions") or {}
    pending_ids = {alert["alert_id"] for alert in state.get("pending_approval") or []}

    def resolve(alert: Alert) -> Alert:
        if alert["alert_id"] not in pending_ids:
            return alert
        status = "FIRED" if decisions.get(alert["alert_id"]) == "approve" else REJECTED_STATUS
        update_status(alert["alert_id"], status)
        resolved: Alert = {**alert, "status": status}  # type: ignore[typeddict-item]
        return resolved

    fired = [resolve(alert) for alert in state.get("fired") or []]
    merged = [resolve(alert) for alert in state.get("merged") or []]

    dispatched = [
        alert["alert_id"] for alert in [*fired, *merged] if alert["status"] == "FIRED" and dispatch_alert(alert)
    ]

    if pending_ids:
        approved = sum(1 for i in pending_ids if decisions.get(i) == "approve")
        logger.info("Approvals resolved: %d/%d approved, %d dispatched", approved, len(pending_ids), len(dispatched))

    return {"fired": fired, "merged": merged, "dispatched": dispatched}


def _should_log_distribution() -> bool:
    """
    True for the first few cycles ever run.

    The score distribution is only diagnostic while the index is young — once
    it is full of real alerts, p90 is a property of the corpus rather than of
    the canonicalization, and logging it every cycle is noise.
    """
    from src.persistence.repository import list_cycles

    try:
        return len(list_cycles(limit=DIAGNOSTIC_CYCLES + 1)) <= DIAGNOSTIC_CYCLES
    except Exception:  # noqa: BLE001 - a diagnostic must never break a cycle
        return False


def persist_cycle_node(state: MonitorState) -> dict:
    """
    Graph node: write the cycle's durable record and advance the watermarks.

    ══ THE WATERMARK ADVANCES HERE, NOT IN THE MONITOR ══
    A monitor that fetched successfully has NOT yet made its findings durable.
    Advancing its watermark at fetch time would mean a crash between fetch and
    persist moves the watermark past events that were never recorded — and
    nothing would ever look for them again. So the alerts are written first,
    and only then does `checked` turn into checkpoints.
    """
    from src.persistence.repository import (
        mark_warmed_up,
        record_alert,
        record_dedup_decisions,
        set_checkpoint,
    )

    cycle_id = state.get("cycle_id", "")

    for alert in [*(state.get("fired") or []), *(state.get("merged") or [])]:
        record_alert(dict(alert), cycle_id=cycle_id)

    record_dedup_decisions([dict(record) for record in state.get("decisions") or []], cycle_id=cycle_id)

    # Only now, with the alerts on disk, is it safe to say these were checked.
    for entry in state.get("checked") or []:
        ticker, _, monitor_key = entry.partition(":")
        if ticker and monitor_key:
            set_checkpoint(ticker, monitor_key)

    if state.get("warmup"):
        # A warmup cycle's whole purpose is to populate the index. Marking the
        # tickers warm is what stops the next cycle repeating it.
        mark_warmed_up(item["ticker"] for item in state.get("watchlist") or [])

    logger.info(
        "Cycle %s persisted: %d fired, %d escalated, %d suppressed, %d watermarks",
        cycle_id,
        len(state.get("fired") or []),
        len(state.get("merged") or []),
        len(state.get("suppressed") or []),
        len(state.get("checked") or []),
    )
    return {}


def build_monitor_graph(checkpointer: Any = None) -> Any:
    """
    Build and compile the monitoring graph.

    Topology::

        START -> load_watchlist --(Send: batched + per-ticker)--> [4 monitors]
                                                                       |
                                                                       v
                                                            alert_synthesizer
                                                                       |
                                                                       v
                                                            human_approval  (interrupt() iff a HIGH is pending)
                                                                       |
                                                                       v
                                                                  dispatcher
                                                                       |
                                                                       v
                                                        persist_cycle -> END

    Unlike the research graph this one is acyclic: there is no repair loop,
    because there is no answer to repair. The cycle either observed something
    or it did not.

    ``human_approval`` and ``dispatcher`` sit unconditionally between the
    synthesizer and persist_cycle — no conditional edge is needed because
    human_approval_node is itself a pass-through when there is nothing HIGH.

    Parameters
    ----------
    checkpointer : optional
        With one, ``thread_id = f"monitor:{cycle_id}"`` makes each cycle an
        independently resumable thread. REQUIRED in practice: without one,
        ``interrupt()`` in human_approval_node still returns rather than
        raising, but the pause it produces cannot be resumed by a later
        process — see the module docstring. run_cycle() enforces this; a
        caller building the graph directly (as most tests do, with no HIGH
        candidates in play) is not stopped from omitting it.

    Returns
    -------
    CompiledStateGraph
    """
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(MonitorState)

    graph.add_node("load_watchlist", load_watchlist_node)
    for name in MONITOR_NAMES:
        graph.add_node(name, MONITOR_NODES[name])
    graph.add_node("alert_synthesizer", alert_synthesizer_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("persist_cycle", persist_cycle_node)

    graph.add_edge(START, "load_watchlist")
    graph.add_conditional_edges("load_watchlist", monitor_fanout, list(MONITOR_NAMES))

    for name in MONITOR_NAMES:
        graph.add_edge(name, "alert_synthesizer")
    graph.add_edge("alert_synthesizer", "human_approval")
    graph.add_edge("human_approval", "dispatcher")
    graph.add_edge("dispatcher", "persist_cycle")
    graph.add_edge("persist_cycle", END)

    return graph.compile(checkpointer=checkpointer)


def thread_id_for(cycle_id: str) -> str:
    """The checkpointer thread id a cycle's state lives under."""
    return f"monitor:{cycle_id}"


def run_cycle(
    *,
    tickers: list[str] | None = None,
    warmup: bool = False,
    checkpointer: Any = None,
) -> MonitorState:
    """
    Run one monitoring cycle — to completion, or to a pause for approval.

    Parameters
    ----------
    tickers : list of str, optional
        Override the stored watchlist — used by the CLI for a single-ticker
        run. The stored watchlist is loaded when omitted.
    warmup : bool, default False
        Observe-only: index candidates for future dedup, report nothing.
    checkpointer : optional
        LangGraph checkpointer. Without one, a cycle that raises a HIGH alert
        still returns a state that LOOKS paused (``__interrupt__`` is set),
        but nothing durable was written for it to resume FROM — a later
        resume_cycle() call would find no state and fail. A warning is logged
        rather than an exception raised, because a cycle with nothing HIGH in
        it runs correctly either way and most cycles are exactly that.

    Returns
    -------
    MonitorState
        The final state if the cycle completed, or the paused state (with a
        ``__interrupt__`` key and populated ``pending_approval``) if it did
        not. Check with ``"__interrupt__" in result``.
    """
    from src.monitor.state import WatchedTicker
    from src.monitor.watchlist import current_watchlist, ensure_seeded
    from src.persistence.repository import record_cycle

    if checkpointer is None:
        logger.warning(
            "run_cycle() called with no checkpointer — a HIGH alert this cycle would pause "
            "unrecoverably. See src/persistence/checkpointer.py."
        )

    ensure_seeded()

    if tickers:
        watchlist = [WatchedTicker(ticker=t.upper(), company_name=t.upper(), warmed_up=False) for t in tickers]
    else:
        watchlist = current_watchlist()

    state = new_cycle_state(watchlist, warmup=warmup)
    graph = build_monitor_graph(checkpointer=checkpointer)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id_for(state["cycle_id"])}}

    started = time.monotonic()
    result: MonitorState = graph.invoke(state, config=config)
    elapsed = int((time.monotonic() - started) * 1000)

    if "__interrupt__" in result:
        logger.info(
            "Cycle %s paused for approval — %d alert(s) pending",
            state["cycle_id"],
            len(result.get("pending_approval") or []),
        )
        record_cycle(dict(result), duration_ms=elapsed, status="PENDING_APPROVAL")
    else:
        record_cycle(dict(result), duration_ms=elapsed, status="COMPLETE")

    return result


def pending_alerts_for(cycle_id: str, *, checkpointer: Any) -> list[Alert]:
    """
    The HIGH alerts one paused cycle is actually waiting on.

    Reads the checkpointer directly rather than any SQLite row: the alerts
    are not written to AlertRecord until persist_cycle_node runs, which is
    exactly what has NOT happened yet for a paused cycle. This is the same
    extraction resume_cycle uses to validate a decisions dict, pulled out so
    the CLI and API can show a human what they are actually deciding on.

    Returns
    -------
    list of Alert
        Empty if the cycle does not exist or is not currently paused.
    """
    graph = build_monitor_graph(checkpointer=checkpointer)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id_for(cycle_id)}}

    snapshot = graph.get_state(config)
    if not snapshot.next:
        return []

    by_id: dict[str, Alert] = {}
    for task in snapshot.tasks:
        for interrupt in task.interrupts:
            for alert in (interrupt.value or {}).get("pending_approval", []):
                by_id[alert["alert_id"]] = alert
    return list(by_id.values())


def resume_cycle(
    cycle_id: str,
    decisions: dict[str, str],
    *,
    checkpointer: Any,
) -> MonitorState:
    """
    Continue a cycle that paused for approval, with a human's decisions.

    Parameters
    ----------
    cycle_id : str
        The paused cycle's id — from run_cycle's result, or
        ``GET /monitor/cycles?status=PENDING_APPROVAL``.
    decisions : dict
        ``{alert_id: "approve" | "reject"}``. Must cover EXACTLY the alert
        ids the cycle is actually waiting on — not a subset, not extras. A
        partial decisions dict would leave some alerts silently defaulted by
        dispatcher_node rather than actually decided, and an alert id that
        does not belong to this cycle is very likely a copy-paste mistake
        from a different one, which is worth a loud failure rather than a
        quiet no-op.
    checkpointer : LangGraph checkpointer
        MUST be connected to the same database run_cycle used — the paused
        state lives there, not in this process's memory. There is no
        default: silently opening a fresh one that happens to point at the
        same file is exactly the kind of implicit dependency that makes it
        surprising when it stops working after a config change.

    Returns
    -------
    MonitorState
        The completed state.

    Raises
    ------
    ValueError
        If the cycle is not currently paused, or ``decisions`` does not
        exactly match the pending alert ids.
    """
    from src.persistence.repository import record_cycle

    pending = pending_alerts_for(cycle_id, checkpointer=checkpointer)
    if not pending:
        raise ValueError(f"Cycle {cycle_id} is not paused — nothing to resume")

    pending_ids = {alert["alert_id"] for alert in pending}
    if set(decisions) != pending_ids:
        raise ValueError(
            f"decisions must cover exactly the pending alerts {sorted(pending_ids)}, got {sorted(decisions)}"
        )

    from langgraph.types import Command

    graph = build_monitor_graph(checkpointer=checkpointer)
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id_for(cycle_id)}}

    started = time.monotonic()
    result: MonitorState = graph.invoke(Command(resume=decisions), config=config)
    elapsed = int((time.monotonic() - started) * 1000)

    record_cycle(dict(result), duration_ms=elapsed, status="COMPLETE")
    logger.info("Cycle %s resumed and completed (%d fired)", cycle_id, len(result.get("fired") or []))
    return result


def cycle_report(state: MonitorState) -> str:
    """
    Render the human-readable cycle summary.

    Suppressions are printed WITH their score and matched parent. Most projects
    hide their cleverest logic; a dedup engine whose decisions are invisible is
    indistinguishable from one that is dropping alerts through a bug.

    A PAUSED cycle (``"__interrupt__" in state``) gets a distinct report: the
    graph stopped before dispatcher_node ran, so nothing — not even a LOW/MED
    finding — has actually been delivered yet, and rendering the ordinary
    "fired" section would claim otherwise.
    """
    cycle_id = state.get("cycle_id", "?")

    if "__interrupt__" in state:
        pending = state.get("pending_approval") or []
        lines = [
            "",
            f"  Cycle {cycle_id}  (PAUSED — awaiting approval, nothing dispatched yet)",
            f"  {len(pending)} HIGH alert(s) pending:",
            "",
        ]
        for alert in pending:
            lines.append(f"  ?? {alert['ticker'] or 'MACRO':<6} {alert['headline']}")
            lines.append(f"        {alert['detail'][:110]}")
            lines.append(f"        alert_id: {alert['alert_id']}")
        lines.append("")
        lines.append(f"  Resolve with: ./run_monitor.sh --resume {cycle_id} --approve <id>  (repeatable, or --reject)")
        lines.append("")
        return "\n".join(lines)

    fired = state.get("fired") or []
    merged = state.get("merged") or []
    suppressed = state.get("suppressed") or []
    candidates = state.get("candidates") or []
    errors = state.get("monitor_errors") or []

    exact = [s for s in suppressed if s["score"] >= 1.0]
    semantic = [s for s in suppressed if s["score"] < 1.0]
    rejected = [a for a in [*fired, *merged] if a.get("status") == "REJECTED"]
    delivered = [a for a in [*fired, *merged] if a.get("status", "FIRED") != "REJECTED"]

    lines = [
        "",
        f"  Cycle {cycle_id}{'  (WARMUP — nothing dispatched)' if state.get('warmup') else ''}",
        f"  {len(candidates)} candidates -> {len(delivered)} fired, {len(suppressed)} suppressed"
        f"{f' ({len(exact)} exact-key, {len(semantic)} semantic)' if suppressed else ''}"
        f"{f', {len(merged)} escalated' if merged else ''}"
        f"{f', {len(rejected)} rejected' if rejected else ''}",
        "",
    ]

    for alert in delivered:
        marker = "!!" if alert["severity"] == "HIGH" else "  "
        lines.append(f"  {marker} [{alert['severity']:<4}] {alert['ticker'] or 'MACRO':<6} {alert['headline']}")
        lines.append(f"        {alert['detail'][:110]}")

    if semantic:
        lines.append("")
        lines.append("  Suppressed (semantic):")
        for record in semantic:
            lines.append(f"     {record['score']:.3f}  {record['ticker'] or 'MACRO':<6} {record['headline'][:60]}")
            lines.append(f"            -> {record['parent_alert_id'][:8]}  {record['parent_headline'][:60]}")

    if errors:
        lines.append("")
        lines.append(f"  {len(errors)} monitor error(s):")
        lines.extend(f"     {message}" for message in errors[:5])

    lines.append("")
    return "\n".join(lines)
