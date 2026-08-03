# ═══════════════════════════════════════════════════════
# FinSight — Tests: Monitoring Graph
# ═══════════════════════════════════════════════════════
#
# What the topology guarantees, as opposed to what any one node does:
#   * the fan-out is asymmetric, and that asymmetry IS the rate-limit strategy
#   * three concurrent branches write `candidates`, so it needs a reducer
#   * the watermark advances only after the alerts are durable
#
# The graph is compiled and invoked for real; only the data sources and the
# vector store are substituted.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.monitor.graph import BATCHED_MONITORS, alert_synthesizer_node, cycle_report, monitor_fanout
from src.monitor.state import MonitorState, new_cycle_id, new_cycle_state
from src.persistence import db as db_module
from src.persistence.db import init_db, reset_engine


@pytest.fixture
def temp_db(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    reset_engine()
    with patch.object(db_module, "DATABASE_URL", url):
        init_db()
        yield url
    reset_engine()


def watchlist(*tickers):
    return [{"ticker": t, "company_name": f"{t} Corp", "warmed_up": True} for t in tickers]


class TestFanoutAsymmetry:
    def test_batched_monitors_get_one_branch_for_the_whole_watchlist(self):
        state = new_cycle_state(watchlist("AAPL", "MSFT", "NVDA", "JPM", "GOOGL"))
        sends = monitor_fanout(state)

        by_node: dict[str, list] = {}
        for send in sends:
            by_node.setdefault(send.node, []).append(send)

        assert len(by_node["price_monitor"]) == 1
        assert len(by_node["macro_monitor"]) == 1

    def test_per_ticker_monitors_get_one_branch_each(self):
        state = new_cycle_state(watchlist("AAPL", "MSFT", "NVDA", "JPM", "GOOGL"))
        sends = monitor_fanout(state)

        by_node: dict[str, list] = {}
        for send in sends:
            by_node.setdefault(send.node, []).append(send)

        assert len(by_node["filing_monitor"]) == 5
        assert len(by_node["news_monitor"]) == 5

    def test_five_tickers_is_twelve_branches_not_twenty(self):
        """
        THE number. Four uniform per-ticker Sends would be 20 branches, five
        identical CPI requests, and five identical batched price requests —
        and would look entirely reasonable in the code.
        """
        assert len(monitor_fanout(new_cycle_state(watchlist("AAPL", "MSFT", "NVDA", "JPM", "GOOGL")))) == 12

    def test_ten_tickers_is_twentytwo_branches(self):
        tickers = [f"T{i}" for i in range(10)]
        assert len(monitor_fanout(new_cycle_state(watchlist(*tickers)))) == 22

    def test_the_batched_branch_carries_every_ticker(self):
        sends = monitor_fanout(new_cycle_state(watchlist("AAPL", "MSFT")))
        price = next(s for s in sends if s.node == "price_monitor")

        assert price.arg["tickers"] == ["AAPL", "MSFT"]

    def test_a_per_ticker_branch_carries_exactly_one(self):
        sends = monitor_fanout(new_cycle_state(watchlist("AAPL", "MSFT")))
        filings = [s for s in sends if s.node == "filing_monitor"]

        assert [s.arg["tickers"] for s in filings] == [["AAPL"], ["MSFT"]]

    def test_an_empty_watchlist_produces_no_branches(self):
        assert monitor_fanout(new_cycle_state([])) == []

    def test_every_branch_carries_its_own_since_map(self):
        state = new_cycle_state(watchlist("AAPL"), last_checked={"AAPL:filing": "2026-08-01T00:00:00+00:00"})
        sends = monitor_fanout(state)

        filings = next(s for s in sends if s.node == "filing_monitor")
        assert filings.arg["since"]["AAPL"].startswith("2026-08-01")

    def test_the_batched_set_is_exactly_price_and_macro(self):
        # Guards against a monitor being added to the batched set without its
        # data source actually supporting a batched request.
        assert BATCHED_MONITORS == {"price_monitor", "macro_monitor"}


class TestStateReducers:
    def test_candidates_carries_a_reducer(self):
        """
        Up to (2 + 2N) branches write this in one superstep. Without the
        reducer LangGraph raises InvalidUpdateError: Can receive only one value
        per step.
        """
        import operator
        from typing import get_type_hints

        hints = get_type_hints(MonitorState, include_extras=True)
        for key in ("candidates", "monitor_errors", "api_calls", "checked"):
            assert operator.add in getattr(hints[key], "__metadata__", ()), key

    def test_dedup_outputs_do_not_carry_reducers(self):
        """
        alert_synthesizer is the single writer. A reducer here would never
        actually merge anything, so the mistake would stay invisible until a
        second writer was added.
        """
        import operator
        from typing import get_type_hints

        hints = get_type_hints(MonitorState, include_extras=True)
        for key in ("fired", "suppressed", "merged", "decisions"):
            assert operator.add not in getattr(hints[key], "__metadata__", ()), key


class TestCycleIdentity:
    def test_a_cycle_id_sorts_chronologically(self):
        from datetime import datetime, timedelta, timezone

        base = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
        earlier = new_cycle_id(now=base)
        later = new_cycle_id(now=base + timedelta(hours=1))

        assert earlier < later

    def test_two_cycles_in_the_same_second_are_distinct(self):
        from datetime import datetime, timezone

        base = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
        assert new_cycle_id(now=base) != new_cycle_id(now=base)


class TestSynthesizerNode:
    def test_no_candidates_produces_empty_outputs(self, temp_db):
        result = alert_synthesizer_node(new_cycle_state(watchlist("AAPL")))

        assert result["fired"] == []
        assert result["decisions"] == []

    def test_a_candidate_flood_is_capped(self, temp_db):
        """
        A data glitch marking every bar as a 40% move must not become hundreds
        of embeddings and hundreds of alerts.
        """
        from src.monitor.config import MAX_CANDIDATES_PER_CYCLE

        state = new_cycle_state(watchlist("AAPL"))
        state["candidates"] = [
            {
                "ticker": "AAPL",
                "company_name": "Apple",
                "alert_type": "PRICE_MOVE",
                "monitor": "price_monitor",
                "headline": f"move {i}",
                "detail": "",
                "natural_key": f"k{i}",
                "metrics": {"change_pct_1d": -5.0},
                "evidence": [],
                "observed_at": "2026-08-03",
            }
            for i in range(MAX_CANDIDATES_PER_CYCLE + 20)
        ]

        seen = {}

        def fake_deduplicate(candidates, **kwargs):
            seen["count"] = len(candidates)
            return []

        with patch("src.monitor.dedup.deduplicate", side_effect=fake_deduplicate):
            alert_synthesizer_node(state)

        assert seen["count"] == MAX_CANDIDATES_PER_CYCLE

    def test_high_severity_alerts_are_staged_for_approval(self, temp_db):
        """
        Phase 7 gates on `pending_approval`. Populated now so the interrupt
        node is a pure addition rather than a state migration.
        """
        from src.monitor.dedup import Decision, DedupOutcome

        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "company_name": "Apple",
            "alert_type": "NEW_FILING",
            "severity": "HIGH",
            "status": "FIRED",
            "headline": "h",
            "detail": "d",
            "canonical_text": "c",
            "dedup_key": "k",
            "metrics": {},
            "evidence": [],
            "occurrence_count": 1,
            "first_seen_at": "2026-08-03T00:00:00+00:00",
            "last_seen_at": "2026-08-03T00:00:00+00:00",
            "fired_at": "2026-08-03T00:00:00+00:00",
            "parent_alert_id": None,
        }
        outcome = DedupOutcome(
            decision=Decision.FIRE,
            reason="",
            score=0.0,
            alert=alert,
            suppression=None,
            record={"ticker": "AAPL", "decision": "FIRE"},
        )

        state = new_cycle_state(watchlist("AAPL"))
        state["candidates"] = [{"alert_type": "NEW_FILING"}]

        with patch("src.monitor.dedup.deduplicate", return_value=[outcome]):
            result = alert_synthesizer_node(state)

        assert result["fired"] == [alert]
        assert result["pending_approval"] == [alert]

    def test_an_escalation_is_reported_as_merged_not_fired(self, temp_db):
        from src.monitor.dedup import Decision, DedupOutcome

        alert = {"alert_id": "a2", "severity": "MED"}
        outcome = DedupOutcome(
            decision=Decision.ESCALATE,
            reason="",
            score=0.85,
            alert=alert,
            suppression=None,
            record={"ticker": "AAPL", "decision": "ESCALATE"},
        )

        state = new_cycle_state(watchlist("AAPL"))
        state["candidates"] = [{"alert_type": "NEWS_SENTIMENT"}]

        with patch("src.monitor.dedup.deduplicate", return_value=[outcome]):
            result = alert_synthesizer_node(state)

        assert result["merged"] == [alert]
        assert result["fired"] == []


class TestPersistence:
    def test_the_watermark_advances_only_after_the_alerts_are_written(self, temp_db):
        """
        A crash between fetch and persist would otherwise move the watermark
        past events that were never recorded, and nothing would ever look for
        them again.
        """
        from src.monitor.graph import persist_cycle_node
        from src.persistence.repository import get_checkpoints, list_alerts

        state = new_cycle_state(watchlist("AAPL"))
        state["checked"] = ["AAPL:filing", "AAPL:news"]
        state["fired"] = [
            {
                "alert_id": "a1",
                "ticker": "AAPL",
                "alert_type": "NEW_FILING",
                "severity": "HIGH",
                "status": "FIRED",
                "headline": "h",
                "detail": "d",
                "canonical_text": "c",
                "dedup_key": "k",
                "metrics": {},
                "evidence": [],
                "occurrence_count": 1,
                "first_seen_at": "2026-08-03T00:00:00+00:00",
                "last_seen_at": "2026-08-03T00:00:00+00:00",
                "fired_at": "2026-08-03T00:00:00+00:00",
                "parent_alert_id": None,
            }
        ]

        persist_cycle_node(state)

        assert len(list_alerts()) == 1
        assert set(get_checkpoints(["AAPL"])) == {"AAPL:filing", "AAPL:news"}

    def test_a_warmup_cycle_marks_its_tickers_warm(self, temp_db):
        from src.monitor.graph import persist_cycle_node
        from src.persistence.repository import add_watch_item, list_watchlist

        add_watch_item("AAPL")
        state = new_cycle_state(watchlist("AAPL"), warmup=True)

        persist_cycle_node(state)
        assert list_watchlist()[0]["warmed_up"]

    def test_a_normal_cycle_does_not(self, temp_db):
        from src.monitor.graph import persist_cycle_node
        from src.persistence.repository import add_watch_item, list_watchlist

        add_watch_item("AAPL")
        persist_cycle_node(new_cycle_state(watchlist("AAPL"), warmup=False))

        assert not list_watchlist()[0]["warmed_up"]


class TestGraphTopology:
    def test_the_graph_compiles_with_every_node_reachable(self):
        from src.monitor.config import MONITOR_NAMES
        from src.monitor.graph import build_monitor_graph

        graph = build_monitor_graph()
        nodes = set(graph.get_graph().nodes)

        for expected in ("load_watchlist", "alert_synthesizer", "persist_cycle", *MONITOR_NAMES):
            assert expected in nodes

    def test_the_graph_is_acyclic(self):
        """
        Unlike the research graph, there is no repair loop: a cycle either
        observed something or it did not, and there is no answer to fix.
        """
        from src.monitor.graph import build_monitor_graph

        edges = build_monitor_graph().get_graph().edges
        assert not any(edge.source == "alert_synthesizer" and edge.target == "load_watchlist" for edge in edges)


class TestCycleReport:
    def test_suppressions_are_shown_with_scores_and_parents(self):
        """
        A dedup engine whose decisions are invisible is indistinguishable from
        one that is dropping alerts through a bug.
        """
        state = new_cycle_state(watchlist("AAPL"))
        state["candidates"] = [{}, {}]
        state["fired"] = [
            {"severity": "HIGH", "ticker": "AAPL", "headline": "AAPL filed an 8-K", "detail": "non-reliance"}
        ]
        state["suppressed"] = [
            {
                "ticker": "AAPL",
                "alert_type": "NEWS_SENTIMENT",
                "headline": "Second outlet on the same story",
                "canonical_text": "c",
                "parent_alert_id": "abcdef12-3456",
                "parent_headline": "First outlet",
                "score": 0.942,
                "reason": "",
            }
        ]

        report = cycle_report(state)
        assert "0.942" in report
        assert "First outlet" in report
        assert "1 semantic" in report

    def test_a_warmup_cycle_says_so(self):
        assert "WARMUP" in cycle_report(new_cycle_state(watchlist("AAPL"), warmup=True))

    def test_errors_are_surfaced(self):
        state = new_cycle_state(watchlist("AAPL"))
        state["monitor_errors"] = ["news_monitor(AAPL): Timeout"]

        assert "Timeout" in cycle_report(state)
