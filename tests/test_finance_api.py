# ═══════════════════════════════════════════════════════
# FinSight — Tests: finance, RL and broker routes
# ═══════════════════════════════════════════════════════
#
# Training runs inline (no thread) on synthetic prices, and the broker is a
# fake that records orders — so these cover the whole propose → approve →
# submit path without a network.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from src.api.auth import Identity, require_identity
from src.broker import config as broker_config
from src.broker import service as broker_service
from src.broker.alpaca import BrokerError
from src.persistence import db as db_module
from src.rl import service as rl_service


class FakeBroker:
    """Stands in for AlpacaPaperClient. Class-level so tests can inspect it."""

    orders: list[dict[str, Any]] = []
    positions_held: list[dict[str, Any]] = []
    fail_with: str | None = None

    def __init__(self, key_id: str, secret: str) -> None:
        if secret == "bad-secret":
            raise BrokerError("unauthorized", 401)
        self.key_id = key_id

    def __enter__(self) -> "FakeBroker":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def account(self) -> dict[str, Any]:
        if self.key_id == "PKREJECTED":
            raise BrokerError("Alpaca GET /v2/account -> 401: unauthorized", 401)
        return {"equity": 10_000.0, "cash": 10_000.0, "status": "ACTIVE", "trading_blocked": False}

    def positions(self) -> list[dict[str, Any]]:
        return list(self.positions_held)

    def submit_market_order(self, **order: Any) -> dict[str, Any]:
        if FakeBroker.fail_with:
            raise BrokerError(FakeBroker.fail_with, 403)
        FakeBroker.orders.append(order)
        return {"id": f"alp-{len(FakeBroker.orders)}"}


def _synthetic_loader(tickers: list[str]) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(1)
    T = 500
    cols = [rng.normal(0.001 if i == 0 else -0.0005, 0.01, T) for i in range(len(tickers))]
    return np.stack(cols, axis=1), [f"2024-{i:04d}" for i in range(T)]


def _make_client(tmp_path, subject: str):
    import src.api.main as main_module

    @asynccontextmanager
    async def fake_checkpointer():
        yield object()

    class _Q:
        def collection_exists(self, name: str) -> bool:
            return True

    db_module.reset_engine()
    patches = [
        patch.object(db_module, "DATABASE_URL", f"sqlite:///{tmp_path / 'fin.db'}"),
        patch.object(main_module, "async_checkpointer", fake_checkpointer),
        patch.object(main_module, "build_research_graph", lambda checkpointer=None: object()),
        patch.object(main_module, "_connect_qdrant", lambda: (_Q(), "connected")),
        patch("src.vectorstore.collections.ensure_collections", lambda: {}),
        patch.object(rl_service, "_spawn", lambda target: target()),
        patch.object(rl_service, "load_returns", _synthetic_loader),
        patch.object(broker_service, "client_factory", FakeBroker),
        patch.object(broker_config, "BROKER_ENCRYPTION_KEY", Fernet.generate_key().decode()),
    ]
    return main_module, patches


@pytest.fixture
def api(tmp_path):
    main_module, patches = _make_client(tmp_path, "alice")
    for p in patches:
        p.start()
    FakeBroker.orders = []
    FakeBroker.positions_held = []
    FakeBroker.fail_with = None
    try:
        app = main_module.create_app()
        who = {"subject": "alice"}
        app.dependency_overrides[require_identity] = lambda: Identity(
            subject=who["subject"], email=f"{who['subject']}@example.com", unlimited=False
        )
        with TestClient(app) as client:
            client.who = who  # type: ignore[attr-defined]
            yield client
    finally:
        for p in reversed(patches):
            p.stop()
        db_module.reset_engine()


class TestBudget:
    def test_summary_tracks_spend_against_limits(self, api):
        api.put("/finance/categories", json={"name": "Groceries", "monthly_limit": 400, "kind": "need"})
        api.post("/finance/entries", json={"occurred_on": "2026-11-02", "amount": 3000, "direction": "income"})
        api.post(
            "/finance/entries",
            json={"occurred_on": "2026-11-03", "amount": 120.5, "direction": "expense", "category": "Groceries"},
        )
        body = api.get("/finance/summary", params={"month": "2026-11"}).json()
        assert body["income"] == 3000 and body["expenses"] == 120.5
        assert body["savings_rate"] == pytest.approx((3000 - 120.5) / 3000, abs=1e-4)
        groceries = next(c for c in body["categories"] if c["name"] == "Groceries")
        assert groceries["spent"] == 120.5 and groceries["monthly_limit"] == 400
        assert len(body["daily"]["spent"]) == 30
        assert body["daily"]["spent"][2] == 120.5
        assert "not financial advice" in body["disclaimer"]

    def test_bad_input_is_422(self, api):
        assert api.post("/finance/entries", json={"occurred_on": "nov 3", "amount": 1}).status_code == 422
        assert api.post("/finance/entries", json={"occurred_on": "2026-11-03", "amount": -5}).status_code == 422
        assert api.get("/finance/summary", params={"month": "2026-13"}).status_code == 422

    def test_goals_progress(self, api):
        goal = api.post("/finance/goals", json={"name": "Emergency", "target": 6000, "saved": 1000}).json()
        updated = api.post(f"/finance/goals/{goal['id']}/contribute", json={"amount": 500}).json()
        assert updated["saved"] == 1500
        # Withdrawals cannot take a goal below zero.
        assert api.post(f"/finance/goals/{goal['id']}/contribute", json={"amount": -9999}).json()["saved"] == 0

    def test_records_are_private_to_their_owner(self, api):
        api.post("/finance/goals", json={"name": "Alice's", "target": 100})
        entry = api.post("/finance/entries", json={"occurred_on": "2026-11-01", "amount": 5}).json()
        api.who["subject"] = "bob"
        assert api.get("/finance/goals").json() == []
        assert api.get("/finance/entries").json() == []
        assert api.delete(f"/finance/entries/{entry['id']}").status_code == 404


class TestTraining:
    def test_a_run_trains_to_done_with_a_backtest(self, api):
        run = api.post("/rl/runs", json={"tickers": ["aaa", "bbb"], "episodes": 80}).json()
        body = api.get(f"/rl/runs/{run['run_id']}").json()
        assert body["status"] == "DONE", body["error"]
        assert body["progress"] == 1.0
        assert body["tickers"] == ["AAA", "BBB"]
        assert set(body["result"]["target_weights"]) == {"AAA", "BBB", "CASH"}
        assert body["result"]["backtest"]["policy"]

    def test_ticker_count_is_validated(self, api):
        assert api.post("/rl/runs", json={"tickers": ["AAA"]}).status_code == 422

    def test_a_failed_download_marks_the_run_failed(self, api):
        def broken(_tickers):
            raise ValueError("No price history for ZZZZ")

        with patch.object(rl_service, "load_returns", broken):
            run = api.post("/rl/runs", json={"tickers": ["AAA", "ZZZZ"], "episodes": 50}).json()
        body = api.get(f"/rl/runs/{run['run_id']}").json()
        assert body["status"] == "FAILED" and "ZZZZ" in body["error"]

    def test_runs_are_private(self, api):
        run = api.post("/rl/runs", json={"tickers": ["AAA", "BBB"], "episodes": 50}).json()
        api.who["subject"] = "bob"
        assert api.get(f"/rl/runs/{run['run_id']}").status_code == 404

    def test_agents_list_is_honest_about_what_trains(self, api):
        agents = {a["id"]: a["status"] for a in api.get("/rl/agents").json()}
        assert agents["portfolio"] == "available"
        assert set(agents) == {"portfolio", "budget", "timing", "alerts"}


class TestBroker:
    def _link(self, api) -> None:
        r = api.post("/broker/link", json={"key_id": "PKTEST12345", "secret": "good-secret"})
        assert r.status_code == 200, r.text

    def _proposals(self, api) -> list[dict]:
        run = api.post("/rl/runs", json={"tickers": ["AAA", "BBB"], "episodes": 80}).json()
        r = api.post("/broker/proposals", json={"run_id": run["run_id"]})
        assert r.status_code == 200, r.text
        return r.json()

    def test_live_keys_are_refused(self, api):
        r = api.post("/broker/link", json={"key_id": "AKLIVE12345", "secret": "good-secret"})
        assert r.status_code == 422 and "LIVE" in r.json()["detail"]
        assert api.get("/broker/status").json()["linked"] is False

    def test_bad_keys_are_not_stored(self, api):
        r = api.post("/broker/link", json={"key_id": "PKREJECTED", "secret": "good-secret"})
        assert r.status_code == 422
        assert api.get("/broker/status").json()["linked"] is False

    def test_keys_are_stored_encrypted(self, api):
        from src.finance.repository import get_broker_link

        self._link(api)
        stored = get_broker_link(subject="alice")
        assert stored and "PKTEST12345" not in stored["key_id_enc"] and "good-secret" not in stored["secret_enc"]
        status = api.get("/broker/status").json()
        assert status["linked"] and status["key_hint"] == "2345" and status["account"]["equity"] == 10_000

    def test_linking_is_disabled_without_an_encryption_key(self, api):
        with patch.object(broker_config, "BROKER_ENCRYPTION_KEY", ""):
            r = api.post("/broker/link", json={"key_id": "PKTEST12345", "secret": "good-secret"})
        assert r.status_code == 503

    def test_proposals_wait_for_approval(self, api):
        self._link(api)
        proposals = self._proposals(api)
        assert proposals and all(p["status"] == "PENDING" for p in proposals)
        assert FakeBroker.orders == []  # generation submits nothing

    def test_approve_submits_once_and_reject_submits_nothing(self, api):
        self._link(api)
        proposals = self._proposals(api)
        first, rest = proposals[0], [p["proposal_id"] for p in proposals[1:]]

        decided = api.post("/broker/proposals/decide", json={"approve": [first["proposal_id"]], "reject": rest}).json()
        states = {p["proposal_id"]: p["status"] for p in decided}
        assert states[first["proposal_id"]] == "SUBMITTED"
        assert all(states[pid] == "REJECTED" for pid in rest)
        assert len(FakeBroker.orders) == 1
        assert FakeBroker.orders[0]["client_order_id"] == first["proposal_id"]

        # A second approval of the same proposal is a no-op, not a second order.
        api.post("/broker/proposals/decide", json={"approve": [first["proposal_id"]]})
        assert len(FakeBroker.orders) == 1

    def test_kill_switch_blocks_submission(self, api):
        self._link(api)
        proposals = self._proposals(api)
        with patch.object(broker_config, "TRADING_KILL_SWITCH", True):
            r = api.post("/broker/proposals/decide", json={"approve": [proposals[0]["proposal_id"]]})
        assert r.status_code == 423 and FakeBroker.orders == []

    def test_pause_blocks_submission(self, api):
        self._link(api)
        proposals = self._proposals(api)
        api.post("/broker/pause", json={"paused": True})
        r = api.post("/broker/proposals/decide", json={"approve": [proposals[0]["proposal_id"]]})
        assert r.status_code == 423 and FakeBroker.orders == []

    def test_a_broker_failure_is_recorded_on_the_proposal(self, api):
        self._link(api)
        proposals = self._proposals(api)
        FakeBroker.fail_with = "market closed"
        decided = api.post("/broker/proposals/decide", json={"approve": [proposals[0]["proposal_id"]]}).json()
        assert decided[0]["status"] == "FAILED" and "market closed" in decided[0]["error"]

    def test_new_proposals_supersede_old_ones(self, api):
        self._link(api)
        old = self._proposals(api)
        self._proposals(api)
        statuses = {p["proposal_id"]: p["status"] for p in api.get("/broker/proposals").json()}
        assert all(statuses[p["proposal_id"]] == "SUPERSEDED" for p in old)

    def test_another_account_cannot_approve_my_proposals(self, api):
        self._link(api)
        proposals = self._proposals(api)
        api.who["subject"] = "mallory"
        api.post("/broker/link", json={"key_id": "PKMALLORY99", "secret": "good-secret"})
        api.post("/broker/proposals/decide", json={"approve": [proposals[0]["proposal_id"]]})
        assert FakeBroker.orders == []
