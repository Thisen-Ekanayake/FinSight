# ═══════════════════════════════════════════════════════
# FinSight — Tests: Research Graph
# ═══════════════════════════════════════════════════════
#
# The fan-out topology is tested by inspecting the Send objects route_fanout
# emits — no LLM, no network, no spend. One end-to-end test is marked llm.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import pytest

from src.research.graph import COMPANY_SCOPED, build_research_graph, route_fanout
from src.research.state import AGENT_NAMES, RoutePlan, new_state


def _state(agents, tickers, *, query="test query", timeframe=None):
    state = new_state(query)
    state["plan"] = RoutePlan(
        tickers=list(tickers),
        timeframe=timeframe,
        selected_agents=list(agents),
        sub_questions={a: f"sub-question for {a}" for a in agents},
        reasoning="test",
    )
    return state


class TestFanOutWidth:
    """
    route_fanout IS the dynamic map-reduce. Branch count is computed at
    runtime from the plan, not fixed by the topology.
    """

    def test_single_agent_single_ticker_is_one_branch(self):
        assert len(route_fanout(_state(["technical"], ["NVDA"]))) == 1

    def test_comparison_multiplies_agents_by_tickers(self):
        sends = route_fanout(_state(["fundamentals", "technical"], ["AAPL", "MSFT"]))
        assert len(sends) == 4

    def test_three_tickers_one_agent_is_three_branches(self):
        assert len(route_fanout(_state(["fundamentals"], ["AAPL", "MSFT", "NVDA"]))) == 3

    def test_company_agent_with_no_tickers_spawns_nothing(self):
        # The router's fallbacks exist to prevent this reaching the graph.
        assert route_fanout(_state(["fundamentals"], [])) == []


class TestMacroAsymmetry:
    """
    FRED series are economy-wide. Fanning macro out per ticker would issue N
    identical calls for one answer, so it is dispatched once per query.
    """

    def test_macro_is_dispatched_once_regardless_of_ticker_count(self):
        sends = route_fanout(_state(["macro"], ["AAPL", "MSFT", "NVDA"]))
        assert len(sends) == 1

    def test_macro_receives_no_ticker(self):
        sends = route_fanout(_state(["macro"], ["AAPL"]))
        assert sends[0].arg["ticker"] == ""

    def test_macro_and_company_agents_mix_correctly(self):
        # macro once + fundamentals per ticker = 1 + 2 = 3
        sends = route_fanout(_state(["macro", "fundamentals"], ["AAPL", "MSFT"]))
        assert len(sends) == 3
        by_node: dict[str, int] = {}
        for send in sends:
            by_node[send.node] = by_node.get(send.node, 0) + 1
        assert by_node == {"macro": 1, "fundamentals": 2}

    def test_company_scoped_set_excludes_macro(self):
        assert "macro" not in COMPANY_SCOPED
        assert COMPANY_SCOPED == set(AGENT_NAMES) - {"macro"}


class TestSendPayloads:
    def test_each_send_targets_its_agent_node(self):
        sends = route_fanout(_state(["fundamentals", "technical"], ["AAPL"]))
        assert {s.node for s in sends} == {"fundamentals", "technical"}

    def test_payload_carries_the_focused_sub_question(self):
        # Not the raw user query — that is the point of sub_questions.
        sends = route_fanout(_state(["fundamentals"], ["AAPL"]))
        assert sends[0].arg["sub_question"] == "sub-question for fundamentals"

    def test_payload_carries_the_ticker(self):
        sends = route_fanout(_state(["fundamentals"], ["AAPL", "MSFT"]))
        assert {s.arg["ticker"] for s in sends} == {"AAPL", "MSFT"}

    def test_payload_carries_the_original_query(self):
        sends = route_fanout(_state(["fundamentals"], ["AAPL"], query="original question"))
        assert sends[0].arg["query"] == "original question"

    def test_payload_carries_the_timeframe(self):
        sends = route_fanout(_state(["fundamentals"], ["AAPL"], timeframe="fiscal 2025"))
        assert sends[0].arg["timeframe"] == "fiscal 2025"

    def test_missing_sub_question_falls_back_to_the_query(self):
        state = _state(["fundamentals"], ["AAPL"], query="fallback query")
        state["plan"]["sub_questions"] = {}
        assert route_fanout(state)[0].arg["sub_question"] == "fallback query"


class TestGraphConstruction:
    def test_compiles(self):
        assert build_research_graph() is not None

    def test_contains_every_node(self):
        nodes = set(build_research_graph().get_graph().nodes)
        for expected in ("router", "aggregator", *AGENT_NAMES):
            assert expected in nodes

    def test_specialists_all_converge_on_the_aggregator(self):
        edges = build_research_graph().get_graph().edges
        targets = {(e.source, e.target) for e in edges}
        for agent in AGENT_NAMES:
            assert (agent, "aggregator") in targets

    def test_accepts_a_checkpointer(self):
        # Phase 4 wires a real one; the parameter must already be threaded.
        from langgraph.checkpoint.memory import InMemorySaver

        assert build_research_graph(checkpointer=InMemorySaver()) is not None


@pytest.mark.llm
@pytest.mark.slow
class TestEndToEnd:
    """Real Gemini and real data sources."""

    def test_macro_query_produces_a_cited_answer(self):
        from src.research.graph import run_research

        state = run_research("What is the current Fed funds rate?")
        assert state["draft_answer"]
        assert state["citations"]
        assert any(c["source_type"] == "FRED" for c in state["citations"])

    def test_company_query_cites_an_accession_number(self):
        import re

        from src.research.graph import run_research

        state = run_research("What was Apple's revenue in fiscal 2025?")
        accessions = [c["source_id"] for c in state["citations"] if c["source_type"] == "EDGAR"]
        assert accessions
        assert any(re.fullmatch(r"\d{10}-\d{2}-\d{6}", a) for a in accessions)

    def test_audit_trail_records_every_branch(self):
        from src.research.graph import run_research

        state = run_research("Compare AAPL and MSFT revenue")
        # Two tickers x one agent = two branches, so two tool calls.
        assert len(state["tool_calls"]) >= 2
        assert all(call["node"] for call in state["tool_calls"])
