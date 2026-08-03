# ═══════════════════════════════════════════════════════
# FinSight — Tests: HTTP API
# ═══════════════════════════════════════════════════════
#
# The graph is replaced by a stub, so these tests exercise the API — routing,
# translation, error handling, persistence — without an LLM, Qdrant, or a
# single unit of quota. What the graph itself does is tested elsewhere.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.persistence import db as db_module


def _final_state(**overrides: Any) -> dict[str, Any]:
    state = {
        "query": "What was Apple's revenue in fiscal 2025?",
        "final_answer": "Apple reported revenue of $416.2B [SRC:EDGAR:0000320193-25-000079].",
        "draft_answer": "Apple reported revenue of $416.2B [SRC:EDGAR:0000320193-25-000079].",
        "plan": {
            "tickers": ["AAPL"],
            "timeframe": "fiscal 2025",
            "selected_agents": ["fundamentals"],
            "sub_questions": {"fundamentals": "revenue?"},
            "reasoning": "numeric lookup",
        },
        "findings": [{"agent": "fundamentals"}],
        "citations": [
            {
                "source_type": "EDGAR",
                "source_id": "0000320193-25-000079",
                "url": "https://sec.gov/x",
                "as_of": "2025-09-27",
                "excerpt": None,
            },
            # A duplicate: one filing cited twice is still one source.
            {
                "source_type": "EDGAR",
                "source_id": "0000320193-25-000079",
                "url": "https://sec.gov/x",
                "as_of": "2025-09-27",
                "excerpt": None,
            },
        ],
        "conflicts": [],
        "errors": [],
        "tool_calls": [
            {
                "node": "fundamentals",
                "tool": "get_fundamentals_history",
                "args": {},
                "provider_used": "edgar_xbrl",
                "cache_hit": False,
                "latency_ms": 812,
                "ok": True,
            }
        ],
        "repair_count": 0,
        "verification": {
            "verified_claims": ["$416.2B -> revenue@2025 FY"],
            "unsupported_claims": [],
            "invalid_source_ids": [],
            "citation_coverage": 1.0,
            "passed": True,
        },
    }
    state.update(overrides)
    return state


@dataclass
class _Task:
    """Stand-in for a LangGraph PregelTask."""

    name: str


@dataclass
class _Snapshot:
    """
    Stand-in for a LangGraph StateSnapshot.

    Shaped from a REAL checkpoint database, not from memory. An earlier
    version of this stub carried node names in ``metadata["writes"]``, the
    tests passed, and the live endpoint returned an audit trail with every
    node column blank — LangGraph 1.2's metadata holds only source/step/
    parents, and node identity lives in ``tasks``.
    """

    values: dict[str, Any]
    metadata: dict[str, Any]
    next: tuple[str, ...] = ()
    tasks: tuple[_Task, ...] = ()
    created_at: str = "2026-08-02T12:00:00+00:00"


@dataclass
class _StubGraph:
    """A graph that returns a canned final state without calling anything."""

    state: dict[str, Any] = field(default_factory=_final_state)
    history: list[_Snapshot] = field(default_factory=list)
    raises: Exception | None = None

    async def ainvoke(self, _input: Any, config: Any = None) -> dict[str, Any]:
        if self.raises:
            raise self.raises
        return self.state

    async def astream(self, _input: Any, config: Any = None, stream_mode: Any = None):
        if self.raises:
            raise self.raises
        yield "updates", {"router": {"plan": self.state["plan"]}}
        yield "updates", {"fundamentals": {"findings": [{"a": 1}], "errors": []}}
        yield "values", self.state

    async def aget_state_history(self, config: Any):
        for snapshot in self.history:
            yield snapshot


def _default_history() -> list[_Snapshot]:
    """
    Newest-first, as LangGraph returns it.

    A snapshot's tasks are the nodes about to run, so this reads backwards as:
    nothing pending after finalize; finalize pending after the aggregator;
    the aggregator pending after two parallel specialists; those two pending
    after the router.
    """
    return [
        _Snapshot(values=_final_state(), metadata={"step": 3}),
        _Snapshot(
            values=_final_state(),
            metadata={"step": 2},
            next=("finalize",),
            tasks=(_Task("finalize"),),
        ),
        _Snapshot(
            values=_final_state(),
            metadata={"step": 1},
            next=("aggregator",),
            tasks=(_Task("aggregator"),),
        ),
        # One superstep, two branches — the shape that proves a parallel fan-out.
        _Snapshot(
            values={},
            metadata={"step": -1},
            next=("fundamentals", "technical"),
            tasks=(_Task("fundamentals"), _Task("technical")),
        ),
    ]


@pytest.fixture
def client(tmp_path):
    """
    A TestClient over an app whose graph is a stub and whose database is
    temporary. The real lifespan still runs, so the wiring under test is the
    wiring that ships.
    """
    import src.api.main as main_module

    graph = _StubGraph(history=_default_history())

    @asynccontextmanager
    async def fake_checkpointer():
        yield object()

    db_module.reset_engine()
    with (
        patch.object(db_module, "DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}"),
        patch.object(main_module, "async_checkpointer", fake_checkpointer),
        patch.object(main_module, "build_research_graph", lambda checkpointer=None: graph),
        patch.object(main_module, "_connect_qdrant", lambda: (object(), "connected")),
    ):
        app = main_module.create_app()
        with TestClient(app) as test_client:
            test_client.graph = graph  # type: ignore[attr-defined]
            yield test_client

    db_module.reset_engine()


class TestHealth:
    def test_reports_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_reports_every_dependency(self, client):
        body = client.get("/health").json()
        assert body["database"] and body["checkpointer"] and body["qdrant"]

    def test_names_the_llm_backend(self, client):
        assert client.get("/health").json()["llm_backend"] in {"vertex", "aistudio"}


class TestQuery:
    def test_returns_the_verified_answer(self, client):
        body = client.post("/research/query", json={"query": "What was Apple's revenue?"}).json()
        assert "416.2B" in body["answer"]
        assert body["verification"]["citation_coverage"] == 1.0
        assert body["verification"]["passed"]

    def test_mints_a_thread_id(self, client):
        body = client.post("/research/query", json={"query": "What was Apple's revenue?"}).json()
        assert body["thread_id"].startswith("research:")

    def test_an_explicit_thread_id_is_honoured(self, client):
        body = client.post("/research/query", json={"query": "Apple revenue?", "thread_id": "mine"}).json()
        assert body["thread_id"] == "mine"

    def test_duplicate_citations_collapse_to_one_source(self, client):
        body = client.post("/research/query", json={"query": "Apple revenue?"}).json()
        assert len(body["citations"]) == 1

    def test_the_plan_is_reported(self, client):
        body = client.post("/research/query", json={"query": "Apple revenue?"}).json()
        assert body["agents_used"] == ["fundamentals"]
        assert body["tickers"] == ["AAPL"]

    def test_a_too_short_query_is_rejected_before_any_spend(self, client):
        assert client.post("/research/query", json={"query": "hi"}).status_code == 422

    def test_a_graph_failure_becomes_a_500_not_a_traceback(self, client):
        client.graph.raises = RuntimeError("vertex exploded")
        response = client.post("/research/query", json={"query": "what happened?"})
        assert response.status_code == 500
        assert "vertex exploded" in response.json()["detail"]

    def test_an_unverified_answer_still_returns(self, client):
        # An honest failed verification is a result, not an error.
        client.graph.state = _final_state(
            verification={
                "verified_claims": [],
                "unsupported_claims": [
                    {
                        "claim": "Revenue was $500B.",
                        "reason": "matches no value any tool returned",
                        "origin_agent": "fundamentals",
                        "ticker": "AAPL",
                    }
                ],
                "invalid_source_ids": [],
                "citation_coverage": 0.0,
                "passed": False,
            }
        )
        body = client.post("/research/query", json={"query": "Apple revenue?"}).json()
        assert not body["verification"]["passed"]
        assert body["verification"]["unsupported_claims"][0]["origin_agent"] == "fundamentals"

    def test_a_run_that_never_reached_verification_still_serialises(self, client):
        state = _final_state()
        del state["verification"]
        client.graph.state = state
        assert client.post("/research/query", json={"query": "Apple revenue?"}).status_code == 200


class TestRunHistory:
    def test_a_query_is_persisted(self, client):
        client.post("/research/query", json={"query": "Apple revenue?", "thread_id": "t1"})
        runs = client.get("/research/runs").json()
        assert [run["thread_id"] for run in runs] == ["t1"]

    def test_the_listing_records_coverage(self, client):
        client.post("/research/query", json={"query": "Apple revenue?", "thread_id": "t1"})
        assert client.get("/research/runs").json()[0]["citation_coverage"] == 1.0

    def test_the_limit_is_bounded(self, client):
        assert client.get("/research/runs?limit=500").status_code == 422


class TestAuditTrail:
    def test_replays_every_superstep(self, client):
        body = client.get("/research/threads/t1").json()
        assert len(body["steps"]) == 4

    def test_steps_read_forwards(self, client):
        # The checkpointer returns newest-first; an audit trail reads forwards.
        steps = client.get("/research/threads/t1").json()["steps"]
        assert [step["step"] for step in steps] == [-1, 1, 2, 3]

    def test_a_parallel_superstep_names_every_node_that_ran(self, client):
        """
        Two node names in ONE step is the clearest evidence the branches
        really ran together rather than in sequence.
        """
        steps = client.get("/research/threads/t1").json()["steps"]
        parallel = next(step for step in steps if step["step"] == 1)
        assert parallel["nodes"] == ["fundamentals", "technical"]

    def test_the_first_step_has_no_predecessor(self, client):
        # Nothing ran to produce the input state.
        steps = client.get("/research/threads/t1").json()["steps"]
        assert steps[0]["nodes"] == []

    def test_duplicate_task_names_are_preserved(self, client):
        """
        Two tasks both named "fundamentals" is one Send per ticker in a single
        superstep. Deduplicating would erase exactly what the audit trail
        exists to show.
        """
        client.graph.history = [
            _Snapshot(values=_final_state(), metadata={"step": 1}),
            _Snapshot(
                values={},
                metadata={"step": 0},
                next=("fundamentals", "fundamentals"),
                tasks=(_Task("fundamentals"), _Task("fundamentals")),
            ),
        ]
        steps = client.get("/research/threads/t1").json()["steps"]
        assert steps[1]["nodes"] == ["fundamentals", "fundamentals"]

    def test_the_tool_call_audit_is_included(self, client):
        body = client.get("/research/threads/t1").json()
        assert body["tool_calls"][0]["provider_used"] == "edgar_xbrl"

    def test_the_summary_is_attached_once_the_run_is_recorded(self, client):
        client.post("/research/query", json={"query": "Apple revenue?", "thread_id": "t1"})
        assert client.get("/research/threads/t1").json()["summary"]["thread_id"] == "t1"

    def test_an_unknown_thread_is_a_404(self, client):
        client.graph.history = []
        assert client.get("/research/threads/nope").status_code == 404


class TestStreaming:
    def test_frames_arrive_in_order(self, client):
        with client.stream("POST", "/research/query/stream", json={"query": "Apple revenue?"}) as response:
            events = [line[7:] for line in response.iter_lines() if line.startswith("event: ")]
        assert events[0] == "start"
        assert events[-1] == "final"
        assert "node" in events

    def test_the_final_frame_carries_the_answer(self, client):
        with client.stream("POST", "/research/query/stream", json={"query": "Apple revenue?"}) as response:
            payloads = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
        assert "416.2B" in json.loads(payloads[-1])["answer"]

    def test_a_failure_is_streamed_as_an_error_frame(self, client):
        client.graph.raises = RuntimeError("boom")
        with client.stream("POST", "/research/query/stream", json={"query": "Apple revenue?"}) as response:
            events = [line[7:] for line in response.iter_lines() if line.startswith("event: ")]
        assert events[-1] == "error"


class TestAdmin:
    def test_budgets_list_every_provider(self, client):
        from src.data.config import DAILY_BUDGETS

        body = client.get("/admin/budgets").json()
        assert {row["provider"] for row in body} == set(DAILY_BUDGETS)

    def test_config_reports_the_qdrant_port(self, client):
        # 6335, not 6333 — Athena owns 6333.
        assert "6335" in client.get("/admin/config").json()["qdrant_url"]

    def test_config_leaks_no_secret(self, client):
        """
        An endpoint whose job is to report configuration is the easiest place
        to leak a key. Named fields only, never an iteration over the config
        module.
        """
        body = json.dumps(client.get("/admin/config").json()).lower()
        for forbidden in ("api_key", "apikey", "secret", "token", "credential", "password"):
            assert forbidden not in body


# ═══════════════════════════════════════════════════════
# Monitoring (Phase 6)
# ═══════════════════════════════════════════════════════


def _alert(**overrides: Any) -> dict[str, Any]:
    alert = {
        "alert_id": "11111111-2222-3333-4444-555555555555",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "alert_type": "NEW_FILING",
        "severity": "HIGH",
        "status": "FIRED",
        "headline": "AAPL filed an 8-K: non-reliance on previously issued financial statements",
        "detail": "8-K filed 2026-08-03; carrying Item 4.02.",
        "canonical_text": "AAPL Apple Inc. | NEW_FILING | auditor flagged non-reliance",
        "dedup_key": "abc123",
        "metrics": {"form_type": "8-K", "items": ["4.02"]},
        "evidence": [
            {
                "source_type": "EDGAR",
                "source_id": "0000320193-26-000010",
                "url": "https://sec.gov/x",
                "as_of": "2026-08-03",
                "excerpt": None,
            }
        ],
        "occurrence_count": 1,
        "first_seen_at": "2026-08-03T12:00:00+00:00",
        "last_seen_at": "2026-08-03T12:00:00+00:00",
        "fired_at": "2026-08-03T12:00:00+00:00",
        "parent_alert_id": None,
    }
    alert.update(overrides)
    return alert


class TestWatchlistRoutes:
    def test_add_then_list(self, client):
        with patch("src.data.edgar.resolve_company_name", return_value="Tesla, Inc."):
            created = client.post("/watchlist", json={"ticker": "tsla"})

        assert created.status_code == 201
        assert created.json()["ticker"] == "TSLA"

        listed = client.get("/watchlist").json()
        assert [row["ticker"] for row in listed] == ["TSLA"]

    def test_adding_twice_is_idempotent(self, client):
        with patch("src.data.edgar.resolve_company_name", return_value="Apple Inc."):
            client.post("/watchlist", json={"ticker": "AAPL"})
            client.post("/watchlist", json={"ticker": "AAPL"})

        assert len(client.get("/watchlist").json()) == 1

    def test_delete_is_a_soft_delete(self, client):
        """
        Hard-deleting would drop the MonitorCheckpoint rows, and re-adding the
        ticker would then report its entire filing history as new.
        """
        with patch("src.data.edgar.resolve_company_name", return_value="Apple Inc."):
            client.post("/watchlist", json={"ticker": "AAPL"})

        assert client.delete("/watchlist/AAPL").status_code == 204
        assert client.get("/watchlist").json() == []
        assert len(client.get("/watchlist", params={"include_inactive": True}).json()) == 1

    def test_deleting_an_unwatched_ticker_is_404(self, client):
        assert client.delete("/watchlist/NOPE").status_code == 404

    def test_watermarks_are_reported_per_monitor(self, client):
        from src.persistence.repository import set_checkpoint

        with patch("src.data.edgar.resolve_company_name", return_value="Apple Inc."):
            client.post("/watchlist", json={"ticker": "AAPL"})
        set_checkpoint("AAPL", "filing")

        row = client.get("/watchlist").json()[0]
        assert "filing" in row["last_checked"]
        assert "news" not in row["last_checked"]  # never checked is ABSENT, not zero


class TestMonitorRoutes:
    def test_running_a_cycle_reports_what_it_suppressed(self, client):
        """
        The suppressions are part of the response, not hidden. A dedup engine
        whose decisions are invisible is indistinguishable from one dropping
        alerts through a bug.
        """
        state = {
            "cycle_id": "20260803T120000Z-abcd1234",
            "warmup": False,
            "candidates": [{}, {}, {}],
            "fired": [_alert()],
            "merged": [],
            "suppressed": [
                {
                    "ticker": "AAPL",
                    "alert_type": "NEWS_SENTIMENT",
                    "headline": "Second outlet, same story",
                    "canonical_text": "c",
                    "parent_alert_id": "p1",
                    "parent_headline": "First outlet",
                    "score": 0.942,
                    "reason": "semantic duplicate",
                }
            ],
            "monitor_errors": [],
            "watchlist": [],
        }

        with patch("src.monitor.graph.run_cycle", return_value=state):
            body = client.post("/monitor/cycles", json={"warmup": False}).json()

        assert body["candidate_count"] == 3
        assert len(body["fired"]) == 1
        assert body["suppressed"][0]["score"] == 0.942
        assert body["suppressed"][0]["parent_headline"] == "First outlet"

    def test_a_warmup_cycle_is_flagged(self, client):
        state = {"cycle_id": "c1", "warmup": True, "candidates": [], "fired": [], "merged": [], "suppressed": []}

        with patch("src.monitor.graph.run_cycle", return_value=state):
            body = client.post("/monitor/cycles", json={"warmup": True}).json()

        assert body["warmup"] is True
        assert body["fired"] == []

    def test_a_failing_cycle_is_a_500_not_a_hang(self, client):
        with patch("src.monitor.graph.run_cycle", side_effect=RuntimeError("qdrant down")):
            response = client.post("/monitor/cycles", json={})

        assert response.status_code == 500
        assert "qdrant down" in response.json()["detail"]

    def test_alerts_are_listed_and_filterable(self, client):
        from src.persistence.repository import record_alert

        record_alert(_alert(alert_id="a1", ticker="AAPL", severity="HIGH"), cycle_id="c1")
        record_alert(_alert(alert_id="a2", ticker="MSFT", severity="LOW"), cycle_id="c1")

        assert len(client.get("/monitor/alerts").json()) == 2
        assert len(client.get("/monitor/alerts", params={"ticker": "AAPL"}).json()) == 1
        assert len(client.get("/monitor/alerts", params={"severity": "high"}).json()) == 1

    def test_an_alert_carries_its_evidence(self, client):
        from src.persistence.repository import record_alert

        record_alert(_alert(alert_id="a1"), cycle_id="c1")
        body = client.get("/monitor/alerts/a1").json()

        assert body["evidence"][0]["source_id"] == "0000320193-26-000010"
        assert body["severity"] == "HIGH"

    def test_an_unknown_alert_is_404(self, client):
        assert client.get("/monitor/alerts/ghost").status_code == 404

    def test_decisions_include_the_fires_not_only_the_suppressions(self, client):
        """
        Phase 7's threshold sweep needs negatives. A log of only suppressions
        can justify the threshold that produced it and nothing else.
        """
        from src.persistence.repository import record_dedup_decisions

        record_dedup_decisions(
            [
                {"ticker": "AAPL", "alert_type": "NEWS_SENTIMENT", "decision": "FIRE", "score": 0.0},
                {"ticker": "AAPL", "alert_type": "NEWS_SENTIMENT", "decision": "SUPPRESS_SEMANTIC", "score": 0.94},
            ],
            cycle_id="c1",
        )

        body = client.get("/monitor/decisions").json()
        assert {row["decision"] for row in body} == {"FIRE", "SUPPRESS_SEMANTIC"}

    def test_cycles_are_listed_newest_first(self, client):
        from src.persistence.repository import record_cycle

        record_cycle({"cycle_id": "c1", "started_at": "2026-08-01T00:00:00+00:00", "watchlist": []}, duration_ms=1)
        record_cycle({"cycle_id": "c2", "started_at": "2026-08-03T00:00:00+00:00", "watchlist": []}, duration_ms=2)

        assert [row["cycle_id"] for row in client.get("/monitor/cycles").json()] == ["c2", "c1"]

    def test_an_unknown_cycle_is_404(self, client):
        assert client.get("/monitor/cycles/nope").status_code == 404
