# ═══════════════════════════════════════════════════════
# FinSight — Router Node
# ═══════════════════════════════════════════════════════
#
# Purpose : Classify a query and plan which specialists to dispatch. This is
#           the node that makes the fan-out DYNAMIC — a single-ticker macro
#           question spawns one branch, a two-company comparison spawns six.
#
# Public API:
#   RouterOutput       structured-output schema
#   router_node(state) graph node
#   plan_query(query)  the plan step alone, for tests and reuse
#
# ══ SCHEMA SHAPE MATTERS ON GEMINI ══
#   Gemini's structured output goes through native JSON-schema/function
#   calling and is stricter than OpenAI's about nested optionals, unions, and
#   dict-valued fields. So RouterOutput is deliberately FLAT: only strings and
#   lists of strings, every field required, no dict fields.
#
#   The per-agent sub-questions therefore arrive as a parallel list of
#   "agent: question" strings rather than a dict, and are reassembled here.
#   Fighting the schema is not worth it — reshaping in Python is free.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field

from src.core.llm import get_llm
from src.core.tracing import trace_metadata
from src.research.config import (
    AGENT_CAPABILITIES,
    DEFAULT_TICKER_LIMIT,
    ROUTER_MODEL_TIER,
    ROUTER_PROMPT_SYSTEM,
    ROUTER_PROMPT_USER,
)
from src.research.state import AGENT_NAMES, ResearchState, RoutePlan

logger = logging.getLogger(__name__)

TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")


class RouterOutput(BaseModel):
    """
    Flat structured-output schema for the router.

    Every field is required and scalar-or-list-of-scalar. See the module
    docstring for why there are no dicts or optionals here.
    """

    tickers: list[str] = Field(
        description="US-listed ticker symbols, uppercase. Empty list for purely macro questions."
    )
    selected_agents: list[str] = Field(
        description="Specialist names to dispatch. Must be from: fundamentals, filings_rag, macro, technical."
    )
    sub_questions: list[str] = Field(
        description=(
            "One entry per selected agent, in the SAME ORDER, formatted exactly as " "'agent_name: focused question'."
        )
    )
    timeframe: str = Field(description="Short timeframe phrase, or empty string if not time-scoped.")
    reasoning: str = Field(description="One sentence explaining the routing choice.")


def _parse_sub_questions(raw: list[str], agents: list[str], query: str) -> dict[str, str]:
    """
    Reassemble "agent: question" strings into a dict.

    Falls back to the original query for any agent the model failed to pair,
    so a malformed line degrades to a slightly less focused search rather than
    an empty sub-question.
    """
    parsed: dict[str, str] = {}

    for line in raw:
        agent, sep, question = line.partition(":")
        agent = agent.strip().lower()
        if sep and agent in AGENT_NAMES and question.strip():
            parsed[agent] = question.strip()

    # Positional fallback: the model was asked to keep the same order.
    for i, agent in enumerate(agents):
        if agent not in parsed and i < len(raw):
            candidate = raw[i].partition(":")[2].strip() or raw[i].strip()
            if candidate:
                parsed[agent] = candidate

    for agent in agents:
        parsed.setdefault(agent, query)

    return parsed


def _sanitize(output: RouterOutput, query: str) -> RoutePlan:
    """
    Validate and normalise raw router output into a RoutePlan.

    An LLM can name a specialist that does not exist or hallucinate a ticker
    shape; both would break the fan-out, so they are filtered here rather than
    trusted downstream.
    """
    agents = [a.strip().lower() for a in output.selected_agents if a.strip().lower() in AGENT_NAMES]
    # Deduplicate while preserving the model's ordering.
    agents = list(dict.fromkeys(agents))

    tickers = [t.strip().upper() for t in output.tickers if TICKER_RE.match(t.strip().upper())]
    tickers = list(dict.fromkeys(tickers))[:DEFAULT_TICKER_LIMIT]

    if not agents:
        # Never dispatch nothing. Filings RAG is the safest default: it answers
        # the widest range of questions and degrades gracefully.
        logger.warning("Router selected no valid agents for %r; defaulting to filings_rag", query[:60])
        agents = ["filings_rag"]

    # Company-scoped specialists need a ticker. If the model found none, drop
    # them rather than fanning out over an empty ticker list — which would
    # spawn zero branches and produce an empty answer.
    if not tickers:
        company_agents = {"fundamentals", "filings_rag", "technical"}
        if set(agents) <= company_agents:
            logger.warning("Router found no tickers for %r; falling back to macro", query[:60])
            agents = ["macro"]
        else:
            agents = [a for a in agents if a not in company_agents]

    return RoutePlan(
        tickers=tickers,
        timeframe=output.timeframe.strip() or None,
        selected_agents=agents,
        sub_questions=_parse_sub_questions(output.sub_questions, agents, query),
        reasoning=output.reasoning.strip(),
    )


def plan_query(query: str, *, _mock_response: str | None = None) -> RoutePlan:
    """
    Produce a routing plan for a query.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    _mock_response : str, optional
        JSON matching RouterOutput, used to bypass the LLM in tests. Keeps
        unit tests free of API cost and network dependence.

    Returns
    -------
    RoutePlan
        Validated and normalised — agents are guaranteed to exist and tickers
        to be plausibly shaped.
    """
    if _mock_response is not None:
        return _sanitize(RouterOutput(**json.loads(_mock_response)), query)

    capabilities = "\n".join(f"  - {name}: {desc}" for name, desc in AGENT_CAPABILITIES.items())
    system = ROUTER_PROMPT_SYSTEM.format(capabilities=capabilities, ticker_limit=DEFAULT_TICKER_LIMIT)

    llm = get_llm(ROUTER_MODEL_TIER, temperature=0.0).with_structured_output(RouterOutput)  # type: ignore[arg-type]
    output = llm.invoke(
        [("system", system), ("human", ROUTER_PROMPT_USER.format(query=query))],
        config={"metadata": trace_metadata(phase="P3"), "tags": ["subsystem1", "router"]},
    )

    plan = _sanitize(output, query)  # type: ignore[arg-type]
    logger.info(
        "Router: %r -> agents=%s tickers=%s (%d branches)",
        query[:60],
        plan["selected_agents"],
        plan["tickers"] or "none",
        len(plan["selected_agents"]) * max(1, len(plan["tickers"])),
    )
    return plan


def router_node(state: ResearchState) -> dict:
    """
    Graph node: plan the query.

    Returns a partial state carrying only ``plan``. The router is the sole
    writer of that key, so it needs no reducer.
    """
    plan = plan_query(state["query"])
    return {"plan": plan}
