# ═══════════════════════════════════════════════════════
# FinSight — Rebalance planner
# ═══════════════════════════════════════════════════════
#
# Purpose : Turn an agent's target weights plus a paper account's current
#           holdings into a short list of dollar-sized orders, clipped to the
#           guardrails in src/broker/config.py.
#
# Public API:
#   plan_rebalance(target_weights, equity, positions) -> list[PlannedOrder]
#
# ══ SCOPE: THE AGENT'S UNIVERSE ONLY ══
#   A position the agent was not trained on is left exactly as it is. The
#   target weights apply to the slice of equity NOT already held outside the
#   universe — so someone holding a stock by hand never sees the agent propose
#   to sell it.
#
# Pure: no I/O, no clock. Everything it needs is an argument.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, TypedDict

from src.broker import config


class PlannedOrder(TypedDict):
    symbol: str
    side: str
    notional: float
    current_weight: float
    target_weight: float
    reason: str


def plan_rebalance(
    target_weights: dict[str, float],
    equity: float,
    positions: list[dict[str, Any]],
    *,
    max_position_pct: float | None = None,
    max_order_notional: float | None = None,
    min_order_notional: float | None = None,
    threshold_pct: float | None = None,
) -> list[PlannedOrder]:
    """
    Plan the orders that move the agent's universe toward ``target_weights``.

    ``target_weights`` may include ``CASH``; it is never traded, only implied.
    Sells come first so that their proceeds fund the buys.
    """
    max_pos = config.MAX_POSITION_PCT if max_position_pct is None else max_position_pct
    max_order = config.MAX_ORDER_NOTIONAL if max_order_notional is None else max_order_notional
    min_order = config.MIN_ORDER_NOTIONAL if min_order_notional is None else min_order_notional
    threshold = config.REBALANCE_THRESHOLD_PCT if threshold_pct is None else threshold_pct

    if equity <= 0:
        return []

    universe = [s.upper() for s in target_weights if s.upper() != "CASH"]
    held = {str(p["symbol"]).upper(): float(p.get("market_value") or 0.0) for p in positions}
    outside = sum(value for symbol, value in held.items() if symbol not in universe)
    budget = max(0.0, equity - outside)

    orders: list[PlannedOrder] = []
    for symbol in universe:
        raw = float(target_weights.get(symbol, target_weights.get(symbol.lower(), 0.0)))
        capped = min(max(raw, 0.0), max_pos)
        target_value = capped * budget
        current_value = held.get(symbol, 0.0)
        delta = target_value - current_value

        if abs(delta) < max(min_order, threshold * equity):
            continue

        side = "buy" if delta > 0 else "sell"
        notional = min(abs(delta), max_order)
        if side == "sell":
            notional = min(notional, current_value)

        reason = f"{symbol}: {current_value / equity:.1%} → {capped * budget / equity:.1%} of equity"
        if capped < raw:
            reason += f" (capped from {raw * budget / equity:.1%} by the {max_pos:.0%} position limit)"
        if notional < abs(delta):
            reason += f"; clipped to the ${max_order:,.0f} order limit"

        orders.append(
            PlannedOrder(
                symbol=symbol,
                side=side,
                notional=round(notional, 2),
                current_weight=round(current_value / equity, 4),
                target_weight=round(capped * budget / equity, 4),
                reason=reason,
            )
        )

    orders.sort(key=lambda o: (o["side"] != "sell", -o["notional"]))
    return orders
