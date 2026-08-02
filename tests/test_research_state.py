# ═══════════════════════════════════════════════════════
# FinSight — Tests: Research Graph State
# ═══════════════════════════════════════════════════════
#
# The reducer configuration is the single most breakable thing in the graph:
# get it wrong and parallel fan-out either crashes or silently drops results.
# These tests pin it, including a live LangGraph run.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import operator
from typing import Annotated, TypedDict, get_args, get_origin, get_type_hints

import pytest

from src.research.state import AGENT_NAMES, ResearchState, new_state

# Keys written concurrently by the four specialist branches in one superstep.
FAN_IN_KEYS = {"findings", "citations", "errors", "tool_calls"}


def _reducer_keys() -> set[str]:
    """Keys whose annotation carries an operator.add reducer."""
    hints = get_type_hints(ResearchState, include_extras=True)
    found = set()
    for name, hint in hints.items():
        if get_origin(hint) is not Annotated:
            continue
        if operator.add in get_args(hint)[1:]:
            found.add(name)
    return found


class TestReducerConfiguration:
    """
    Exactly the fan-in keys get reducers — no more, no less.

    Too few: LangGraph raises InvalidUpdateError when two branches write.
    Too many: a single-writer key silently accumulates across repair-loop
    iterations instead of being replaced.
    """

    def test_fan_in_keys_have_reducers(self):
        assert FAN_IN_KEYS <= _reducer_keys()

    def test_no_other_key_has_a_reducer(self):
        assert _reducer_keys() == FAN_IN_KEYS

    def test_single_writer_keys_have_no_reducer(self):
        for key in ("plan", "draft_answer", "final_answer", "conflicts", "repair_count"):
            assert key not in _reducer_keys(), f"{key} has one writer and must not accumulate"

    def test_state_is_a_typeddict(self):
        assert issubclass(ResearchState, dict)

    def test_partial_updates_are_allowed(self):
        # total=False is what lets a node return only the keys it owns.
        assert ResearchState.__total__ is False


class TestNewState:
    def test_seeds_every_reducer_list_empty(self):
        state = new_state("test query")
        for key in FAN_IN_KEYS:
            assert state[key] == []  # type: ignore[literal-required]

    def test_carries_the_query(self):
        assert new_state("what is Apple's revenue?")["query"] == "what is Apple's revenue?"

    def test_repair_count_starts_at_zero(self):
        assert new_state("q")["repair_count"] == 0

    def test_thread_id_is_optional(self):
        assert new_state("q")["thread_id"] == ""
        assert new_state("q", thread_id="abc")["thread_id"] == "abc"


class TestAgentNames:
    def test_matches_the_planned_specialists(self):
        assert set(AGENT_NAMES) == {"fundamentals", "filings_rag", "macro", "technical"}

    def test_is_immutable(self):
        # graph.py and router.py must agree exactly; a mutable list invites drift.
        assert isinstance(AGENT_NAMES, tuple)


class TestReducerBehaviourInLangGraph:
    """
    Live LangGraph runs. These are the tests that actually prove the fan-in
    works, rather than just inspecting annotations.
    """

    def test_concurrent_writes_to_a_reducer_key_merge(self):
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send

        class S(TypedDict, total=False):
            items: Annotated[list[str], operator.add]

        def fan_out(_state):
            return [Send("worker", {"value": v}) for v in ("a", "b", "c")]

        def worker(payload):
            return {"items": [payload["value"]]}

        g = StateGraph(S)
        g.add_node("start", lambda s: {})
        g.add_node("worker", worker)
        g.add_edge(START, "start")
        g.add_conditional_edges("start", fan_out, ["worker"])
        g.add_edge("worker", END)

        result = g.compile().invoke({"items": []})
        assert sorted(result["items"]) == ["a", "b", "c"]

    def test_concurrent_writes_without_a_reducer_raise(self):
        """
        THE LESSON. Two branches writing one non-reducer key is an error, not
        a last-write-wins. Triggering it deliberately is the fastest way to
        understand that fan-out branches run as a single superstep.
        """
        from langgraph.errors import InvalidUpdateError
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send

        class S(TypedDict, total=False):
            value: str  # deliberately NO reducer

        def fan_out(_state):
            return [Send("worker", {"value": v}) for v in ("a", "b")]

        def worker(payload):
            return {"value": payload["value"]}

        g = StateGraph(S)
        g.add_node("start", lambda s: {})
        g.add_node("worker", worker)
        g.add_edge(START, "start")
        g.add_conditional_edges("start", fan_out, ["worker"])
        g.add_edge("worker", END)

        with pytest.raises(InvalidUpdateError, match="one value per step"):
            g.compile().invoke({})

    def test_a_single_branch_needs_no_reducer(self):
        # Confirms the error above is about CONCURRENCY, not the key itself.
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send

        class S(TypedDict, total=False):
            value: str

        g = StateGraph(S)
        g.add_node("start", lambda s: {})
        g.add_node("worker", lambda payload: {"value": payload["value"]})
        g.add_edge(START, "start")
        g.add_conditional_edges("start", lambda _s: [Send("worker", {"value": "only"})], ["worker"])
        g.add_edge("worker", END)

        assert g.compile().invoke({})["value"] == "only"

    def test_research_state_survives_a_real_fan_out(self):
        """The actual ResearchState shape under a four-branch fan-out."""
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Send

        def fan_out(_state):
            return [Send("specialist", {"agent": name}) for name in AGENT_NAMES]

        def specialist(payload):
            agent = payload["agent"]
            return {
                "findings": [{"agent": agent, "claim": f"{agent} finding"}],
                "citations": [{"source_type": "EDGAR", "source_id": f"acc-{agent}"}],
                "errors": [],
                "tool_calls": [{"node": agent, "tool": "t"}],
            }

        g = StateGraph(ResearchState)
        g.add_node("router", lambda s: {"plan": {"selected_agents": list(AGENT_NAMES)}})
        g.add_node("specialist", specialist)
        g.add_node("aggregator", lambda s: {"draft_answer": f"{len(s['findings'])} findings"})
        g.add_edge(START, "router")
        g.add_conditional_edges("router", fan_out, ["specialist"])
        g.add_edge("specialist", "aggregator")
        g.add_edge("aggregator", END)

        result = g.compile().invoke(new_state("test"))

        assert len(result["findings"]) == len(AGENT_NAMES)
        assert len(result["citations"]) == len(AGENT_NAMES)
        assert result["draft_answer"] == f"{len(AGENT_NAMES)} findings"
        # Single-writer key set once by the aggregator, not accumulated.
        assert isinstance(result["draft_answer"], str)
