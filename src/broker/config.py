# ═══════════════════════════════════════════════════════
# FinSight — Broker guardrails
# ═══════════════════════════════════════════════════════
#
# Purpose : The hard limits every agent-proposed order is clipped to, and the
#           global kill switch. Read from the environment once at import.
#
# ══ PAPER ONLY, BY CONSTRUCTION ══
#   There is no setting here that selects a live endpoint, and that is the
#   point: src/broker/alpaca.py hard-codes the paper host and refuses any
#   other. Turning real-money trading on is a code change and a review, never
#   an environment variable someone flips on a server.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import os


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# Fernet key used to encrypt each account's broker credentials at rest.
# Generate one with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Unset means broker linking is disabled — keys are never stored in plaintext.
BROKER_ENCRYPTION_KEY: str = os.getenv("BROKER_ENCRYPTION_KEY", "")

# Global kill switch. When true, no order is submitted for ANY account; proposals
# can still be generated and reviewed.
TRADING_KILL_SWITCH: bool = os.getenv("TRADING_KILL_SWITCH", "false").lower() in {"1", "true", "yes"}

# Per-order and per-position caps, in dollars of the paper account.
MAX_ORDER_NOTIONAL: float = _float("MAX_ORDER_NOTIONAL", 5_000.0)
MAX_POSITION_PCT: float = _float("MAX_POSITION_PCT", 0.40)
MAX_ORDERS_PER_DAY: int = int(_float("MAX_ORDERS_PER_DAY", 20))

# Alpaca rejects fractional orders under $1; smaller drifts are not worth a trade.
MIN_ORDER_NOTIONAL: float = _float("MIN_ORDER_NOTIONAL", 5.0)
REBALANCE_THRESHOLD_PCT: float = _float("REBALANCE_THRESHOLD_PCT", 0.01)
