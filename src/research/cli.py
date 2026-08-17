# ═══════════════════════════════════════════════════════
# FinSight — Research CLI
# ═══════════════════════════════════════════════════════
#
# Streams node-by-node progress, then prints the grounded answer with its
# citations and audit trail.
#
# Usage:
#   python -m src.research.cli "How did Apple's gross margin trend?"
#   python -m src.research.cli --plan-only "Compare AAPL and MSFT"
#   python -m src.research.cli --audit "What is the Fed funds rate?"
#
# Runs are checkpointed to the same database the API reads, so any query asked
# here is replayable at GET /research/threads/{thread_id}.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import sys
import time
import uuid

from src.core.logging_setup import configure_logging
from src.core.tracing import configure_tracing
from src.persistence.checkpointer import sync_checkpointer
from src.persistence.db import init_db
from src.persistence.repository import LEGACY_SUBJECT, record_research_run
from src.research.config import RECURSION_LIMIT
from src.research.graph import build_research_graph
from src.research.state import RoutePlan, new_state

RULE = "─" * 74


def _print_plan(plan: RoutePlan) -> None:
    branches = len(plan["selected_agents"]) * max(1, len(plan["tickers"]))
    print(f"\n{RULE}\nPLAN")
    print(f"  tickers    {', '.join(plan['tickers']) or '(none — macro only)'}")
    print(f"  agents     {', '.join(plan['selected_agents'])}")
    print(f"  timeframe  {plan['timeframe'] or '(unscoped)'}")
    print(f"  branches   {branches}")
    print(f"  reasoning  {plan['reasoning']}")
    for agent, question in sorted(plan["sub_questions"].items()):
        print(f"    {agent:14s} {question}")


def _print_verification(report: dict) -> None:
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"\n{RULE}\nVERIFICATION  {status}")
    print(f"  citation coverage   {report['citation_coverage']:.0%}")
    print(f"  grounded claims     {len(report['verified_claims'])}")

    for claim in report["unsupported_claims"]:
        print(f"  UNSUPPORTED  {claim['reason']}")
        print(f"               -> requery {claim['origin_agent']}({claim['ticker'] or 'n/a'})")
    for marker in report["invalid_source_ids"]:
        print(f"  BAD SOURCE   {marker}")


def _print_answer(state: dict, *, show_audit: bool) -> None:
    print(f"\n{RULE}\nANSWER\n{RULE}")
    print(state.get("final_answer") or state.get("draft_answer") or "(no answer produced)")

    report = state.get("verification")
    if report:
        _print_verification(report)

    citations = state.get("citations", [])
    if citations:
        seen = set()
        print(f"\n{RULE}\nSOURCES")
        for citation in citations:
            key = (citation["source_type"], citation["source_id"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{citation['source_type']}:{citation['source_id']}]  {citation['url']}")

    conflicts = state.get("conflicts", [])
    if conflicts:
        print(f"\n{RULE}\nCONFLICTS RESOLVED")
        for conflict in conflicts:
            reported = "; ".join(f"{s}={v:,.2f}" for s, v in conflict["values"])
            print(
                f"  {conflict['ticker'] or 'macro'} {conflict['metric']}: {reported} "
                f"-> used {conflict['chosen_source']} ({conflict['rel_difference'] * 100:.2f}% spread)"
            )

    errors = state.get("errors", [])
    if errors:
        print(f"\n{RULE}\nSPECIALIST FAILURES (answer is partial)")
        for error in errors:
            print(f"  {error}")

    if show_audit:
        print(f"\n{RULE}\nAUDIT TRAIL")
        for call in state.get("tool_calls", []):
            status = "ok " if call["ok"] else "ERR"
            cache = " (cached)" if call["cache_hit"] else ""
            print(
                f"  {status} {call['node']:14s} {call['tool']:20s} "
                f"{call['latency_ms']:>6}ms  via {call['provider_used'] or '-'}{cache}"
            )
        print(f"\n  findings collected: {len(state.get('findings', []))}")


def main(argv: list[str] | None = None) -> int:
    """Run a research query from the shell."""
    parser = argparse.ArgumentParser(description="Ask FinSight a financial research question")
    parser.add_argument("query", help="natural-language question")
    parser.add_argument("--plan-only", action="store_true", help="show the routing plan without executing")
    parser.add_argument("--audit", action="store_true", help="print the tool-call audit trail")
    parser.add_argument("--quiet", action="store_true", help="suppress per-node progress")
    parser.add_argument("--thread", default="", help="reuse a checkpoint thread id")
    args = parser.parse_args(argv)

    configure_logging(level="WARNING" if args.quiet else None)
    configure_tracing()

    if args.plan_only:
        from src.research.router import plan_query

        _print_plan(plan_query(args.query))
        return 0

    thread_id = args.thread or f"research:{uuid.uuid4().hex[:16]}"
    started = time.monotonic()
    final: dict = {}

    print(f"\n{RULE}\nQUERY  {args.query}")

    # Checkpointed even from the CLI, and on the same database the API uses,
    # so a run started here shows up at GET /research/threads/{thread_id}. One
    # audit trail, whichever door the query came through.
    with sync_checkpointer() as saver:
        graph = build_research_graph(checkpointer=saver)

        # Two stream modes at once:
        #   "updates" — each node's PARTIAL state as it completes, which is
        #               what makes the fan-out visible rather than a black box.
        #   "values"  — the full merged state after each superstep.
        #
        # The final state must come from "values". Merging the "updates"
        # chunks by hand would overwrite the reducer-backed keys instead of
        # concatenating them, so the audit trail would show only the last
        # branch's tool calls.
        for mode, chunk in graph.stream(
            new_state(args.query, thread_id=thread_id),
            config={"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": thread_id}},
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final = chunk
                continue

            for node, update in chunk.items():
                if node == "router":
                    _print_plan(update["plan"])
                    print(f"\n{RULE}\nBRANCHES")
                elif node == "citation_verifier":
                    targets = update.get("repair_targets", [])
                    if targets:
                        print(f"  {'repair':14s} re-querying {', '.join(t['origin_agent'] for t in targets)}")
                elif node not in ("aggregator", "finalize"):
                    count = len(update.get("findings", []))
                    failed = update.get("errors", [])
                    mark = "FAILED" if failed else f"{count} findings"
                    print(f"  {node:14s} {mark}")

    elapsed = time.monotonic() - started
    _print_answer(final, show_audit=args.audit)

    init_db()
    # ══ WHY THE CLI RECORDS NO OWNER ══
    #   There is no token here and no sign-in — nothing to attribute a run to.
    #   LEGACY_SUBJECT ('') puts CLI runs in the same bucket as everything
    #   written before ownership existed, and that is the only value correct in
    #   BOTH deployments: with auth on, the unlimited tier (the operator, who
    #   is the person at this shell) sees them; with auth off, every caller is
    #   an unlimited ANONYMOUS_USER and sees them too. Recording ANONYMOUS_USER
    #   instead would hide CLI runs from everyone the moment auth is switched
    #   on — including the operator who ran them.
    record_research_run(final, thread_id=thread_id, subject=LEGACY_SUBJECT, latency_ms=int(elapsed * 1000))

    print(f"\n{RULE}\ncompleted in {elapsed:.1f}s   thread {thread_id}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
