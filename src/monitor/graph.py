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
#   persist_cycle_node(state)
#   build_monitor_graph(checkpointer=None)
#   run_cycle(...)              convenience wrapper
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
# ══ WHERE PHASE 7 LANDS ══
#   `human_approval` (an interrupt() gate for HIGH alerts) and `dispatcher`
#   insert between the synthesizer and persist_cycle. The state already carries
#   pending_approval / approval_decisions / dispatched so that change is an
#   edge rewiring rather than a state migration.
#
# Usage:
#   ./run_monitor.sh --once --warmup
#   ./run_monitor.sh --once
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
                                                        persist_cycle -> END

    Unlike the research graph this one is acyclic: there is no repair loop,
    because there is no answer to repair. The cycle either observed something
    or it did not.

    Parameters
    ----------
    checkpointer : optional
        With one, ``thread_id = f"monitor:{cycle_id}"`` makes each cycle an
        independently resumable thread — which is what Phase 7's interrupt gate
        needs to pause for hours and resume after a process restart.

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
    graph.add_node("persist_cycle", persist_cycle_node)

    graph.add_edge(START, "load_watchlist")
    graph.add_conditional_edges("load_watchlist", monitor_fanout, list(MONITOR_NAMES))

    for name in MONITOR_NAMES:
        graph.add_edge(name, "alert_synthesizer")
    graph.add_edge("alert_synthesizer", "persist_cycle")
    graph.add_edge("persist_cycle", END)

    return graph.compile(checkpointer=checkpointer)


def run_cycle(
    *,
    tickers: list[str] | None = None,
    warmup: bool = False,
    checkpointer: Any = None,
) -> MonitorState:
    """
    Run one monitoring cycle end to end.

    Parameters
    ----------
    tickers : list of str, optional
        Override the stored watchlist — used by the CLI for a single-ticker
        run. The stored watchlist is loaded when omitted.
    warmup : bool, default False
        Observe-only: index candidates for future dedup, report nothing.
    checkpointer : optional
        LangGraph checkpointer.

    Returns
    -------
    MonitorState
        Final state, including ``fired``, ``suppressed``, ``merged``, and the
        ``decisions`` audit trail.
    """
    from src.monitor.state import WatchedTicker
    from src.monitor.watchlist import current_watchlist, ensure_seeded
    from src.persistence.repository import record_cycle

    ensure_seeded()

    if tickers:
        watchlist = [WatchedTicker(ticker=t.upper(), company_name=t.upper(), warmed_up=False) for t in tickers]
    else:
        watchlist = current_watchlist()

    state = new_cycle_state(watchlist, warmup=warmup)
    graph = build_monitor_graph(checkpointer=checkpointer)

    config: dict[str, Any] = {"configurable": {"thread_id": f"monitor:{state['cycle_id']}"}}

    started = time.monotonic()
    result: MonitorState = graph.invoke(state, config=config)
    elapsed = int((time.monotonic() - started) * 1000)

    record_cycle(dict(result), duration_ms=elapsed)
    return result


def cycle_report(state: MonitorState) -> str:
    """
    Render the human-readable cycle summary.

    Suppressions are printed WITH their score and matched parent. Most projects
    hide their cleverest logic; a dedup engine whose decisions are invisible is
    indistinguishable from one that is dropping alerts through a bug.
    """
    fired = state.get("fired") or []
    merged = state.get("merged") or []
    suppressed = state.get("suppressed") or []
    candidates = state.get("candidates") or []
    errors = state.get("monitor_errors") or []

    exact = [s for s in suppressed if s["score"] >= 1.0]
    semantic = [s for s in suppressed if s["score"] < 1.0]

    lines = [
        "",
        f"  Cycle {state.get('cycle_id', '?')}{'  (WARMUP — nothing dispatched)' if state.get('warmup') else ''}",
        f"  {len(candidates)} candidates -> {len(fired)} fired, {len(suppressed)} suppressed"
        f"{f' ({len(exact)} exact-key, {len(semantic)} semantic)' if suppressed else ''}"
        f"{f', {len(merged)} escalated' if merged else ''}",
        "",
    ]

    for alert in [*fired, *merged]:
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
