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
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import sys
import time

from src.core.logging_setup import configure_logging
from src.core.tracing import configure_tracing
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


def _print_answer(state: dict, *, show_audit: bool) -> None:
    print(f"\n{RULE}\nANSWER\n{RULE}")
    print(state.get("draft_answer") or "(no answer produced)")

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
    args = parser.parse_args(argv)

    configure_logging(level="WARNING" if args.quiet else None)
    configure_tracing()

    if args.plan_only:
        from src.research.router import plan_query

        _print_plan(plan_query(args.query))
        return 0

    graph = build_research_graph()
    started = time.monotonic()
    final: dict = {}

    print(f"\n{RULE}\nQUERY  {args.query}")

    # Two stream modes at once:
    #   "updates" — each node's PARTIAL state as it completes, which is what
    #               makes the fan-out visible rather than a black box.
    #   "values"  — the full merged state after each superstep.
    #
    # The final state must come from "values". Merging the "updates" chunks by
    # hand would overwrite the reducer-backed keys instead of concatenating
    # them, so the audit trail would show only the last branch's tool calls.
    for mode, chunk in graph.stream(
        new_state(args.query),
        config={"recursion_limit": RECURSION_LIMIT},
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final = chunk
            continue

        for node, update in chunk.items():
            if node == "router":
                _print_plan(update["plan"])
                print(f"\n{RULE}\nBRANCHES")
            elif node != "aggregator":
                count = len(update.get("findings", []))
                failed = update.get("errors", [])
                mark = "FAILED" if failed else f"{count} findings"
                print(f"  {node:14s} {mark}")

    _print_answer(final, show_audit=args.audit)
    print(f"\n{RULE}\ncompleted in {time.monotonic() - started:.1f}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
