# ═══════════════════════════════════════════════════════
# FinSight — Alpaca PAPER trading client
# ═══════════════════════════════════════════════════════
#
# Purpose : The four calls FinSight makes to a broker — account, positions,
#           submit an order, look an order up — against Alpaca's paper API.
#
# Public API:
#   PAPER_BASE_URL, AlpacaPaperClient, BrokerError
#
# ══ THE PAPER HOST IS A CONSTANT, AND IT IS CHECKED ══
#   Alpaca's live and paper APIs are the same API on different hosts, and a
#   paper key used against the live host simply fails — but a LIVE key used
#   against a configurable host would trade real money. So the host is not
#   configurable: it is this constant, and the constructor asserts it even
#   when a transport is injected for tests.
#
# ══ IDEMPOTENCY ══
#   Every order carries client_order_id = the FinSight proposal id. If a
#   submit times out after Alpaca accepted it, the retry is rejected as a
#   duplicate and we look the original up instead of placing a second order.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_TIMEOUT = httpx.Timeout(10.0)


class BrokerError(RuntimeError):
    """A broker call failed. ``status`` is the HTTP status, or 0 for transport errors."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class AlpacaPaperClient:
    """A minimal synchronous client for Alpaca's paper-trading REST API."""

    def __init__(self, key_id: str, secret: str, *, transport: httpx.BaseTransport | None = None) -> None:
        if not key_id or not secret:
            raise BrokerError("Alpaca key id and secret are both required")
        self._http = httpx.Client(
            base_url=PAPER_BASE_URL,
            headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret},
            timeout=_TIMEOUT,
            transport=transport,
        )
        # Belt and braces: nothing above should be able to change this.
        assert str(self._http.base_url).rstrip("/") == PAPER_BASE_URL, "Alpaca client must target the paper host"

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "AlpacaPaperClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── transport ─────────────────────────────────────

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BrokerError(f"Could not reach Alpaca paper API: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("message", response.text)
            except ValueError:
                detail = response.text
            raise BrokerError(f"Alpaca {method} {path} -> {response.status_code}: {detail}", response.status_code)
        return response.json() if response.content else None

    # ── reads ─────────────────────────────────────────

    def account(self) -> dict[str, Any]:
        data = self._call("GET", "/v2/account")
        return {
            "account_number": data.get("account_number", ""),
            "status": data.get("status", ""),
            "currency": data.get("currency", "USD"),
            "equity": float(data.get("equity") or 0),
            "cash": float(data.get("cash") or 0),
            "buying_power": float(data.get("buying_power") or 0),
            "last_equity": float(data.get("last_equity") or 0),
            "trading_blocked": bool(data.get("trading_blocked", False)),
        }

    def positions(self) -> list[dict[str, Any]]:
        rows = self._call("GET", "/v2/positions") or []
        return [
            {
                "symbol": row.get("symbol", ""),
                "qty": float(row.get("qty") or 0),
                "market_value": float(row.get("market_value") or 0),
                "avg_entry_price": float(row.get("avg_entry_price") or 0),
                "current_price": float(row.get("current_price") or 0),
                "unrealized_pl": float(row.get("unrealized_pl") or 0),
                "unrealized_plpc": float(row.get("unrealized_plpc") or 0),
            }
            for row in rows
        ]

    def order_by_client_id(self, client_order_id: str) -> dict[str, Any]:
        return dict(self._call("GET", "/v2/orders:by_client_order_id", params={"client_order_id": client_order_id}))

    # ── writes ────────────────────────────────────────

    def submit_market_order(self, *, symbol: str, side: str, notional: float, client_order_id: str) -> dict[str, Any]:
        """
        Submit a fractional market DAY order sized in dollars.

        A duplicate ``client_order_id`` means an earlier attempt already landed;
        that original order is returned rather than raising.
        """
        if side not in {"buy", "sell"}:
            raise BrokerError(f"side must be buy or sell, not {side!r}")
        if notional <= 0:
            raise BrokerError("notional must be positive")
        body = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "notional": f"{notional:.2f}",
            "client_order_id": client_order_id,
        }
        try:
            return dict(self._call("POST", "/v2/orders", json=body))
        except BrokerError as exc:
            if exc.status == 422 and "client_order_id" in str(exc):
                logger.info("Order %s already submitted; returning the original", client_order_id)
                return self.order_by_client_id(client_order_id)
            raise
