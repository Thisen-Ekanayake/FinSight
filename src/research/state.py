# ═══════════════════════════════════════════════════════
# FinSight — Research Graph State
# ═══════════════════════════════════════════════════════
#
# Purpose : The typed state that travels through Subsystem 1's StateGraph.
#           Every node reads and returns a partial of this shape.
#
# Public API:
#   ResearchState      the graph state
#   SpecialistInput    what Send() delivers to one specialist branch
#   RoutePlan          the router's decision
#   UnsupportedClaim, VerificationReport
#   AGENT_NAMES, new_state()
#
# ══ THE REDUCER RULE ══
#   Exactly four keys carry `Annotated[..., operator.add]`, because exactly
#   four nodes write them in the SAME superstep during the parallel fan-out:
#
#       findings  citations  errors  tool_calls
#
#   Every other key has a single writer. Writing a non-reducer key from two
#   concurrent branches raises:
#
#       InvalidUpdateError: At key 'x': Can receive only one value per step.
#
#   That error is worth triggering deliberately once — it is the fastest way
#   to understand that LangGraph executes fan-out branches as one superstep
#   and merges their partial states, rather than running them in sequence.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from src.core.schemas import AgentFinding, Citation, Conflict, ToolCallRecord

# The four specialist nodes the router may dispatch to. Kept here rather than
# in config.py because graph.py and router.py must agree on them exactly — a
# typo would produce a Send() to a node that does not exist.
AGENT_NAMES: tuple[str, ...] = ("fundamentals", "filings_rag", "macro", "technical")

AgentName = Literal["fundamentals", "filings_rag", "macro", "technical"]


class RoutePlan(TypedDict):
    """
    The router's decision about how to answer a query.

    ``sub_questions`` matters: each specialist receives a focused question
    rather than the raw user query, so the filings agent searches for what it
    can actually answer instead of the whole multi-part question.
    """

    tickers: list[str]
    timeframe: str | None
    selected_agents: list[str]
    sub_questions: dict[str, str]
    reasoning: str


class SpecialistInput(TypedDict):
    """
    The payload Send() delivers to ONE specialist branch.

    Note this is not the graph state — it is a per-branch argument. The
    specialist returns a partial ResearchState, which the reducers merge.
    """

    query: str
    ticker: str
    sub_question: str
    timeframe: str | None


class UnsupportedClaim(TypedDict):
    """
    A claim the citation verifier could not ground in tool output.

    ``origin_agent`` and ``suggested_requery`` are what make the repair loop
    targeted: only the agent that produced the bad claim is re-invoked, rather
    than re-running the whole fan-out.
    """

    claim: str
    reason: str
    origin_agent: str
    suggested_requery: str


class VerificationReport(TypedDict):
    """Output of the citation verifier. ``citation_coverage`` is the headline metric."""

    verified_claims: list[str]
    unsupported_claims: list[UnsupportedClaim]
    invalid_source_ids: list[str]
    citation_coverage: float
    passed: bool


class ResearchState(TypedDict, total=False):
    """
    State for the interactive research graph.

    ``total=False`` so nodes may return partial updates without declaring
    every key — which is how LangGraph state updates work in practice.
    """

    # ── Inputs (single writer: the caller) ──
    query: str
    thread_id: str

    # ── Router output (single writer: router) ──
    plan: RoutePlan

    # ── PARALLEL FAN-IN — reducers REQUIRED ──
    # Four specialist branches write these concurrently in one superstep.
    findings: Annotated[list[AgentFinding], operator.add]
    citations: Annotated[list[Citation], operator.add]
    errors: Annotated[list[str], operator.add]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]

    # ── Aggregation (single writer: aggregator) ──
    conflicts: list[Conflict]
    draft_answer: str

    # ── Verification (single writer: citation_verifier) — Phase 4 ──
    verification: VerificationReport
    repair_count: int

    # ── Output (single writer: finalize) ──
    final_answer: str


def new_state(query: str, *, thread_id: str = "") -> ResearchState:
    """
    Build the initial state for a research run.

    The four reducer-backed lists are seeded empty so specialists can append
    without a first-write special case.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    thread_id : str, optional
        Checkpoint thread identifier, used from Phase 4 onward.

    Returns
    -------
    ResearchState
    """
    return ResearchState(
        query=query,
        thread_id=thread_id,
        findings=[],
        citations=[],
        errors=[],
        tool_calls=[],
        conflicts=[],
        repair_count=0,
    )
