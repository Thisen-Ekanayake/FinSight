# ═══════════════════════════════════════════════════════
# FinSight — Research Graph
# ═══════════════════════════════════════════════════════
#
# Purpose : Wire the router, specialists, and aggregator into a StateGraph.
#
# Public API:
#   route_fanout(state)      the conditional edge that spawns branches
#   build_research_graph()   compiled graph
#   run_research(query)      convenience wrapper
#
# ══ WHY Send() AND NOT FOUR STATIC PARALLEL EDGES ══
#   Send() gives a DYNAMIC fan-out: the branch set is computed at runtime from
#   the router's plan.
#
#     "Has NVDA been overbought?"          -> 1 branch
#     "Compare AAPL and MSFT margins"      -> 2 agents x 2 tickers = 4 branches
#     "What is the Fed funds rate?"        -> 1 branch, no ticker
#
#   Static edges would force every specialist to run on every query — four
#   times the API calls, four times the latency, and irrelevant findings
#   diluting the synthesis prompt. The alternative, gating each static edge
#   with a no-op check, is the same logic written worse.
#
#   The macro asymmetry is deliberate and encoded here rather than in the
#   agent: FRED series are economy-wide, so macro is dispatched ONCE per query
#   while company-scoped specialists are dispatched once PER TICKER. Fanning
#   macro out per ticker would issue N identical FRED calls for one answer.
#
# Usage:
#   python -m src.research.cli "How did Apple's gross margin trend?"
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any

from src.research.agents import AGENT_NODES
from src.research.aggregator import aggregator_node
from src.research.config import RECURSION_LIMIT
from src.research.router import router_node
from src.research.state import AGENT_NAMES, ResearchState, new_state

logger = logging.getLogger(__name__)

# Specialists that need a ticker. macro is excluded — see the module docstring.
COMPANY_SCOPED: frozenset[str] = frozenset({"fundamentals", "filings_rag", "technical"})


def route_fanout(state: ResearchState) -> list[Any]:
    """
    Conditional edge: expand the router's plan into concrete branches.

    This function IS the dynamic map-reduce. It returns a list of Send
    objects, each carrying the payload for one specialist invocation, and
    LangGraph executes them all in a single superstep whose partial states the
    reducers merge.

    Parameters
    ----------
    state : ResearchState
        Must contain ``plan``, written by the router.

    Returns
    -------
    list of Send
        One per branch. Empty only if the plan selected nothing, which the
        router's fallbacks are designed to prevent.
    """
    from langgraph.types import Send

    plan = state["plan"]
    tickers = plan["tickers"]
    sends: list[Any] = []

    for agent in plan["selected_agents"]:
        sub_question = plan["sub_questions"].get(agent, state["query"])

        if agent in COMPANY_SCOPED:
            # One branch per ticker — this is where a comparison multiplies.
            for ticker in tickers:
                sends.append(
                    Send(
                        agent,
                        {
                            "query": state["query"],
                            "ticker": ticker,
                            "sub_question": sub_question,
                            "timeframe": plan["timeframe"],
                        },
                    )
                )
        else:
            # Economy-wide: dispatched once regardless of ticker count.
            sends.append(
                Send(
                    agent,
                    {
                        "query": state["query"],
                        "ticker": "",
                        "sub_question": sub_question,
                        "timeframe": plan["timeframe"],
                    },
                )
            )

    logger.info("Fan-out: %d branches from %s x %s", len(sends), plan["selected_agents"], tickers or ["-"])
    return sends


def build_research_graph(checkpointer: Any = None) -> Any:
    """
    Build and compile the research graph.

    Topology::

        START -> router --(Send: dynamic fan-out)--> [specialists] -> aggregator -> END

    Parameters
    ----------
    checkpointer : optional
        LangGraph checkpointer. Wired in Phase 4 for durable execution and the
        thread-based audit trail.

    Returns
    -------
    CompiledStateGraph
    """
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ResearchState)

    graph.add_node("router", router_node)
    for name in AGENT_NAMES:
        graph.add_node(name, AGENT_NODES[name])
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "router")
    # path_map lists every node route_fanout may target; LangGraph needs it to
    # build the edge set, since the actual targets are only known at runtime.
    graph.add_conditional_edges("router", route_fanout, list(AGENT_NAMES))

    for name in AGENT_NAMES:
        graph.add_edge(name, "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile(checkpointer=checkpointer)


def run_research(query: str, *, thread_id: str = "") -> ResearchState:
    """
    Run one research query end to end.

    Parameters
    ----------
    query : str
        Natural-language question.
    thread_id : str, optional
        Checkpoint thread id, used from Phase 4.

    Returns
    -------
    ResearchState
        Final state, including ``draft_answer``, ``findings``, ``citations``,
        ``conflicts``, and the ``tool_calls`` audit trail.
    """
    graph = build_research_graph()
    config: dict[str, Any] = {"recursion_limit": RECURSION_LIMIT}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}

    result: ResearchState = graph.invoke(new_state(query, thread_id=thread_id), config=config)
    return result
