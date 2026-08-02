# ═══════════════════════════════════════════════════════
# FinSight — Tests: Router Node
# ═══════════════════════════════════════════════════════
#
# Uses _mock_response throughout, so these consume zero Vertex quota and zero
# spend. One `llm`-marked test hits the real model.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json

import pytest

from src.research.config import DEFAULT_TICKER_LIMIT
from src.research.router import RouterOutput, plan_query, router_node
from src.research.state import AGENT_NAMES


def _mock(
    *,
    tickers=("AAPL",),
    agents=("fundamentals",),
    sub_questions=None,
    timeframe="",
    reasoning="test",
) -> str:
    if sub_questions is None:
        sub_questions = [f"{a}: focused question for {a}" for a in agents]
    return json.dumps(
        {
            "tickers": list(tickers),
            "selected_agents": list(agents),
            "sub_questions": list(sub_questions),
            "timeframe": timeframe,
            "reasoning": reasoning,
        }
    )


class TestSchemaShape:
    """
    Gemini's structured output is stricter than OpenAI's about nested
    optionals, unions, and dicts. RouterOutput must stay flat.
    """

    def test_every_field_is_required(self):
        for field in RouterOutput.model_fields.values():
            assert field.is_required(), "an optional field invites Gemini schema rejection"

    def test_no_dict_valued_fields(self):
        for name, field in RouterOutput.model_fields.items():
            assert "dict" not in str(field.annotation).lower(), f"{name} must not be a dict"

    def test_annotations_are_flat(self):
        allowed = {"list[str]", "str"}
        for name, field in RouterOutput.model_fields.items():
            rendered = str(field.annotation).replace("<class 'str'>", "str")
            assert rendered in allowed, f"{name} has a non-flat annotation: {rendered}"


class TestPlanNormalisation:
    def test_uppercases_tickers(self):
        plan = plan_query("q", _mock_response=_mock(tickers=["aapl"]))
        assert plan["tickers"] == ["AAPL"]

    def test_deduplicates_tickers(self):
        plan = plan_query("q", _mock_response=_mock(tickers=["AAPL", "AAPL"]))
        assert plan["tickers"] == ["AAPL"]

    def test_caps_ticker_count(self):
        # Fan-out is agents x tickers; an unbounded list would burn quota.
        many = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "TSLA", "AMZN"]
        assert len(many) > DEFAULT_TICKER_LIMIT
        plan = plan_query("q", _mock_response=_mock(tickers=many))
        assert len(plan["tickers"]) == DEFAULT_TICKER_LIMIT

    def test_drops_implausible_tickers(self):
        plan = plan_query("q", _mock_response=_mock(tickers=["AAPL", "not a ticker", "TOOOOLONG"]))
        assert plan["tickers"] == ["AAPL"]

    def test_rejects_tickers_containing_digits(self):
        # US-listed symbols are alphabetic; a digit signals a hallucinated or
        # non-US symbol, and EDGAR would have nothing for it.
        plan = plan_query("q", _mock_response=_mock(tickers=["AAPL", "TCK0"]))
        assert plan["tickers"] == ["AAPL"]

    def test_drops_unknown_agents(self):
        # A hallucinated specialist name would Send() to a nonexistent node.
        plan = plan_query("q", _mock_response=_mock(agents=["fundamentals", "crystal_ball"]))
        assert plan["selected_agents"] == ["fundamentals"]

    def test_deduplicates_agents(self):
        plan = plan_query("q", _mock_response=_mock(agents=["macro", "macro"], tickers=[]))
        assert plan["selected_agents"] == ["macro"]

    def test_empty_timeframe_becomes_none(self):
        assert plan_query("q", _mock_response=_mock(timeframe=""))["timeframe"] is None

    def test_timeframe_is_preserved(self):
        plan = plan_query("q", _mock_response=_mock(timeframe="last two fiscal years"))
        assert plan["timeframe"] == "last two fiscal years"


class TestFallbacks:
    """The graph must always have something to dispatch."""

    def test_no_valid_agents_falls_back_to_filings_rag(self):
        plan = plan_query("q", _mock_response=_mock(agents=["nonsense"]))
        assert plan["selected_agents"] == ["filings_rag"]

    def test_company_agents_without_tickers_fall_back_to_macro(self):
        # Fanning out over an empty ticker list spawns ZERO branches and
        # yields an empty answer, so this case must be caught.
        plan = plan_query("q", _mock_response=_mock(agents=["fundamentals"], tickers=[]))
        assert plan["selected_agents"] == ["macro"]

    def test_macro_survives_without_tickers(self):
        plan = plan_query("q", _mock_response=_mock(agents=["macro"], tickers=[]))
        assert plan["selected_agents"] == ["macro"]

    def test_mixed_agents_without_tickers_keep_only_macro(self):
        plan = plan_query("q", _mock_response=_mock(agents=["macro", "technical"], tickers=[]))
        assert plan["selected_agents"] == ["macro"]


class TestSubQuestions:
    def test_parses_agent_prefixed_lines(self):
        plan = plan_query(
            "original",
            _mock_response=_mock(
                agents=["fundamentals", "macro"],
                sub_questions=["fundamentals: what was revenue?", "macro: what is CPI doing?"],
            ),
        )
        assert plan["sub_questions"]["fundamentals"] == "what was revenue?"
        assert plan["sub_questions"]["macro"] == "what is CPI doing?"

    def test_every_selected_agent_gets_a_sub_question(self):
        plan = plan_query(
            "original query",
            _mock_response=_mock(agents=["fundamentals", "macro"], sub_questions=[]),
        )
        for agent in plan["selected_agents"]:
            assert plan["sub_questions"][agent]

    def test_missing_pairs_fall_back_to_the_original_query(self):
        plan = plan_query(
            "original query",
            _mock_response=_mock(agents=["fundamentals", "macro"], sub_questions=["fundamentals: revenue?"]),
        )
        assert plan["sub_questions"]["macro"] == "original query"

    def test_unprefixed_lines_are_matched_positionally(self):
        plan = plan_query(
            "original",
            _mock_response=_mock(agents=["fundamentals"], sub_questions=["what was revenue?"]),
        )
        assert plan["sub_questions"]["fundamentals"] == "what was revenue?"

    def test_agent_names_are_case_insensitive(self):
        plan = plan_query(
            "original",
            _mock_response=_mock(agents=["fundamentals"], sub_questions=["Fundamentals: revenue?"]),
        )
        assert plan["sub_questions"]["fundamentals"] == "revenue?"


class TestFanOutWidth:
    """The router is what makes the fan-out dynamic rather than fixed."""

    def test_single_ticker_single_agent_is_one_branch(self):
        plan = plan_query("q", _mock_response=_mock(agents=["technical"], tickers=["NVDA"]))
        assert len(plan["selected_agents"]) * len(plan["tickers"]) == 1

    def test_comparison_multiplies_branches(self):
        plan = plan_query(
            "q",
            _mock_response=_mock(agents=["fundamentals", "technical"], tickers=["AAPL", "MSFT"]),
        )
        assert len(plan["selected_agents"]) * len(plan["tickers"]) == 4

    def test_macro_only_needs_no_tickers(self):
        plan = plan_query("q", _mock_response=_mock(agents=["macro"], tickers=[]))
        assert plan["tickers"] == []


class TestRouterNode:
    def test_returns_only_the_plan_key(self):
        from unittest.mock import patch

        with patch("src.research.router.plan_query", return_value={"selected_agents": ["macro"]}):
            update = router_node({"query": "q"})  # type: ignore[arg-type]
        # Single writer, so no reducer — it must not touch fan-in keys.
        assert set(update) == {"plan"}


@pytest.mark.llm
class TestLiveRouter:
    """Real Gemini. Costs a fraction of a cent per call on Vertex."""

    def test_routes_a_company_financials_question(self):
        plan = plan_query("What was Apple's revenue in fiscal 2025?")
        assert "AAPL" in plan["tickers"]
        assert "fundamentals" in plan["selected_agents"]

    def test_routes_a_pure_macro_question(self):
        plan = plan_query("What is the current Fed funds rate?")
        assert plan["selected_agents"] == ["macro"]
        assert plan["tickers"] == []

    def test_routes_a_comparison_to_multiple_tickers(self):
        plan = plan_query("Compare AAPL and MSFT revenue growth")
        assert set(plan["tickers"]) == {"AAPL", "MSFT"}

    def test_selected_agents_are_always_real(self):
        plan = plan_query("Has NVDA been overbought recently?")
        assert set(plan["selected_agents"]) <= set(AGENT_NAMES)
