# ═══════════════════════════════════════════════════════
# FinSight — Tests: Alpaca paper client and rebalance planner
# ═══════════════════════════════════════════════════════
# httpx.MockTransport stands in for Alpaca: nothing here opens a socket.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json

import httpx
import pytest

from src.broker.alpaca import PAPER_BASE_URL, AlpacaPaperClient, BrokerError
from src.broker.rebalance import plan_rebalance


class TestPaperOnly:
    def test_host_is_the_paper_host(self):
        assert PAPER_BASE_URL == "https://paper-api.alpaca.markets"

    def test_requests_go_to_paper_with_both_headers(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={"equity": "100000", "cash": "100000", "status": "ACTIVE"})

        with AlpacaPaperClient("PKTEST1234", "secret-xyz", transport=httpx.MockTransport(handler)) as client:
            account = client.account()

        assert account["equity"] == 100_000.0
        assert seen[0].url.host == "paper-api.alpaca.markets"
        assert seen[0].headers["APCA-API-KEY-ID"] == "PKTEST1234"

    def test_missing_keys_are_refused(self):
        with pytest.raises(BrokerError):
            AlpacaPaperClient("", "x")


class TestOrders:
    def test_submits_a_notional_market_order_with_the_idempotency_key(self):
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "ord-1", "status": "accepted"})

        with AlpacaPaperClient("PKTEST1234", "s3cretkey", transport=httpx.MockTransport(handler)) as client:
            order = client.submit_market_order(symbol="aapl", side="buy", notional=123.456, client_order_id="p-1")

        assert order["id"] == "ord-1"
        assert bodies[0] == {
            "symbol": "AAPL",
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "notional": "123.46",
            "client_order_id": "p-1",
        }

    def test_a_duplicate_submit_returns_the_original_order(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(422, json={"message": "client_order_id must be unique"})
            assert request.url.params["client_order_id"] == "p-1"
            return httpx.Response(200, json={"id": "ord-original"})

        with AlpacaPaperClient("PKTEST1234", "s3cretkey", transport=httpx.MockTransport(handler)) as client:
            order = client.submit_market_order(symbol="AAPL", side="buy", notional=10, client_order_id="p-1")
        assert order["id"] == "ord-original"

    def test_other_errors_raise_with_status(self):
        handler = lambda request: httpx.Response(403, json={"message": "forbidden"})  # noqa: E731
        with AlpacaPaperClient("PKTEST1234", "s3cretkey", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(BrokerError) as info:
                client.submit_market_order(symbol="AAPL", side="buy", notional=10, client_order_id="p-1")
        assert info.value.status == 403

    @pytest.mark.parametrize("side,notional", [("hold", 10), ("buy", 0)])
    def test_bad_orders_never_leave_the_process(self, side, notional):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
            raise AssertionError("request sent")

        with AlpacaPaperClient("PKTEST1234", "s3cretkey", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(BrokerError):
                client.submit_market_order(symbol="AAPL", side=side, notional=notional, client_order_id="x")


class TestRebalance:
    LIMITS = dict(max_position_pct=0.4, max_order_notional=5_000, min_order_notional=5, threshold_pct=0.01)

    def test_buys_from_all_cash(self):
        orders = plan_rebalance({"AAA": 0.3, "BBB": 0.3, "CASH": 0.4}, 10_000, [], **self.LIMITS)
        assert {o["symbol"]: (o["side"], o["notional"]) for o in orders} == {
            "AAA": ("buy", 3000.0),
            "BBB": ("buy", 3000.0),
        }

    def test_position_cap(self):
        orders = plan_rebalance({"AAA": 0.9, "CASH": 0.1}, 10_000, [], **self.LIMITS)
        assert orders[0]["notional"] == 4000.0
        assert "capped" in orders[0]["reason"]

    def test_order_cap(self):
        orders = plan_rebalance({"AAA": 0.4, "CASH": 0.6}, 100_000, [], **self.LIMITS)
        assert orders[0]["notional"] == 5000.0
        assert "clipped" in orders[0]["reason"]

    def test_sells_come_first_and_never_exceed_the_holding(self):
        positions = [{"symbol": "AAA", "market_value": 4000}]
        orders = plan_rebalance({"AAA": 0.0, "BBB": 0.3, "CASH": 0.7}, 10_000, positions, **self.LIMITS)
        assert orders[0]["side"] == "sell" and orders[0]["symbol"] == "AAA"
        assert orders[0]["notional"] == 4000.0

    def test_small_drift_is_ignored(self):
        positions = [{"symbol": "AAA", "market_value": 2990}]
        assert plan_rebalance({"AAA": 0.3, "CASH": 0.7}, 10_000, positions, **self.LIMITS) == []

    def test_holdings_outside_the_universe_are_untouched(self):
        positions = [{"symbol": "MANUAL", "market_value": 5000}]
        orders = plan_rebalance({"AAA": 0.4, "CASH": 0.6}, 10_000, positions, **self.LIMITS)
        assert [o["symbol"] for o in orders] == ["AAA"]
        # 40% of the 5,000 NOT already held by hand.
        assert orders[0]["notional"] == 2000.0

    def test_zero_equity_plans_nothing(self):
        assert plan_rebalance({"AAA": 1.0}, 0, []) == []
