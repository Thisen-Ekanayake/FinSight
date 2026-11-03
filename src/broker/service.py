# ═══════════════════════════════════════════════════════
# FinSight — Broker service
# ═══════════════════════════════════════════════════════
#
# Purpose : Link an account to Alpaca paper, turn a finished RL run into
#           order PROPOSALS, and submit only the ones a person approved.
#
# Public API:
#   link, unlink, status, set_paused, propose, decide, ServiceError
#
# ══ THE ORDER PATH ══
#   agent run → plan_rebalance → PENDING proposals → a person approves →
#   guardrail checks (kill switch, pause, daily cap, size) → APPROVED →
#   SUBMITTING (conditional UPDATE: at most one winner) → Alpaca paper with
#   client_order_id = proposal id → SUBMITTED | FAILED.
#
#   No step is skippable. In particular nothing here submits on generation:
#   an agent can only ever ASK.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Any, Callable

from src.broker import config
from src.broker.alpaca import AlpacaPaperClient, BrokerError
from src.broker.rebalance import plan_rebalance
from src.broker.secrets import EncryptionUnavailable, decrypt, encrypt, encryption_available
from src.finance import repository as repo

logger = logging.getLogger(__name__)

# Swapped in tests for a fake that never opens a socket.
ClientFactory = Callable[[str, str], AlpacaPaperClient]
client_factory: ClientFactory = AlpacaPaperClient


class ServiceError(RuntimeError):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _client(subject: str) -> AlpacaPaperClient:
    link = repo.get_broker_link(subject=subject)
    if link is None:
        raise ServiceError("No Alpaca paper account is linked.", 404)
    try:
        return client_factory(decrypt(link["key_id_enc"]), decrypt(link["secret_enc"]))
    except EncryptionUnavailable as exc:
        raise ServiceError(str(exc), 503) from exc


def limits() -> dict[str, Any]:
    return {
        "max_order_notional": config.MAX_ORDER_NOTIONAL,
        "max_position_pct": config.MAX_POSITION_PCT,
        "max_orders_per_day": config.MAX_ORDERS_PER_DAY,
        "min_order_notional": config.MIN_ORDER_NOTIONAL,
        "kill_switch": config.TRADING_KILL_SWITCH,
    }


def link(*, subject: str, key_id: str, secret: str) -> dict[str, Any]:
    """Verify the keys against the paper API, then store them encrypted."""
    if not encryption_available():
        raise ServiceError("Broker linking is disabled on this deployment (no BROKER_ENCRYPTION_KEY).", 503)
    key_id, secret = key_id.strip(), secret.strip()
    if key_id.startswith("AK"):
        # Alpaca LIVE key ids start AK; paper ids start PK. Refuse live keys
        # outright so they are never stored, even though the paper host would
        # reject them anyway.
        raise ServiceError("That looks like a LIVE Alpaca key (AK…). FinSight only accepts PAPER keys (PK…).", 422)
    try:
        with client_factory(key_id, secret) as client:
            account = client.account()
    except BrokerError as exc:
        raise ServiceError(f"Alpaca paper rejected those keys: {exc}", 422) from exc

    repo.save_broker_link(subject=subject, key_id_enc=encrypt(key_id), secret_enc=encrypt(secret), key_hint=key_id[-4:])
    return account


def unlink(*, subject: str) -> bool:
    return repo.delete_broker_link(subject=subject)


def set_paused(*, subject: str, paused: bool) -> None:
    if not repo.set_trading_paused(subject=subject, paused=paused):
        raise ServiceError("No Alpaca paper account is linked.", 404)


def status(*, subject: str) -> dict[str, Any]:
    link_row = repo.get_broker_link(subject=subject)
    out: dict[str, Any] = {
        "linked": link_row is not None,
        "encryption_available": encryption_available(),
        "paused": bool(link_row and link_row["trading_paused"]),
        "key_hint": link_row["key_hint"] if link_row else "",
        "limits": limits(),
        "account": None,
        "positions": [],
        "error": None,
    }
    if link_row is None:
        return out
    try:
        with _client(subject) as client:
            out["account"] = client.account()
            out["positions"] = client.positions()
    except (BrokerError, ServiceError) as exc:
        out["error"] = str(exc)
    return out


def propose(*, subject: str, run_id: str) -> list[dict[str, Any]]:
    run = repo.get_run(subject=subject, run_id=run_id)
    if run is None:
        raise ServiceError("Training run not found.", 404)
    if run["status"] != "DONE" or not run["result"]:
        raise ServiceError("That run has not finished training.", 409)

    try:
        with _client(subject) as client:
            account = client.account()
            positions = client.positions()
    except BrokerError as exc:
        raise ServiceError(str(exc), 502) from exc

    orders = plan_rebalance(run["result"]["target_weights"], account["equity"], positions)
    return repo.replace_pending_proposals(subject=subject, run_id=run_id, orders=[dict(o) for o in orders])


def _submit_one(client: AlpacaPaperClient, *, subject: str, proposal: dict[str, Any]) -> None:
    """Move one PENDING proposal through SUBMITTING to SUBMITTED or FAILED."""
    proposal_id = proposal["proposal_id"]
    if proposal["notional"] > config.MAX_ORDER_NOTIONAL:
        repo.transition_proposal(
            subject=subject,
            proposal_id=proposal_id,
            from_status="PENDING",
            to_status="FAILED",
            error="Exceeds the per-order limit.",
        )
        return
    if not repo.transition_proposal(
        subject=subject, proposal_id=proposal_id, from_status="PENDING", to_status="SUBMITTING"
    ):
        return  # someone else won the race
    try:
        order = client.submit_market_order(
            symbol=proposal["symbol"],
            side=proposal["side"],
            notional=proposal["notional"],
            client_order_id=proposal_id,
        )
    except BrokerError as exc:
        logger.warning("Order %s failed: %s", proposal_id, exc)
        repo.transition_proposal(
            subject=subject, proposal_id=proposal_id, from_status="SUBMITTING", to_status="FAILED", error=str(exc)[:500]
        )
        return
    repo.transition_proposal(
        subject=subject,
        proposal_id=proposal_id,
        from_status="SUBMITTING",
        to_status="SUBMITTED",
        broker_order_id=str(order.get("id", "")),
    )


def _check_can_trade(subject: str) -> None:
    if config.TRADING_KILL_SWITCH:
        raise ServiceError("Trading is halted on this deployment (kill switch). Nothing was submitted.", 423)
    link_row = repo.get_broker_link(subject=subject)
    if link_row is None:
        raise ServiceError("No Alpaca paper account is linked.", 404)
    if link_row["trading_paused"]:
        raise ServiceError("Trading is paused for your account. Resume it to submit orders.", 423)


def decide(*, subject: str, approve: list[str], reject: list[str]) -> list[dict[str, Any]]:
    """Reject, then approve-and-submit. Returns the proposals touched, in their final state."""
    touched = set(approve) | set(reject)
    for proposal_id in reject:
        repo.transition_proposal(subject=subject, proposal_id=proposal_id, from_status="PENDING", to_status="REJECTED")

    if approve:
        _check_can_trade(subject)
        pending = {p["proposal_id"]: p for p in repo.list_proposals(subject=subject, status="PENDING")}
        chosen = [pending[a] for a in dict.fromkeys(approve) if a in pending]  # unknown / decided ids are skipped
        budget = config.MAX_ORDERS_PER_DAY - repo.count_submitted_today(subject=subject)
        if len(chosen) > budget:
            raise ServiceError(f"That would exceed today's limit of {config.MAX_ORDERS_PER_DAY} orders.", 429)
        if chosen:
            with _client(subject) as client:
                for proposal in chosen:
                    _submit_one(client, subject=subject, proposal=proposal)

    return [p for p in repo.list_proposals(subject=subject, limit=200) if p["proposal_id"] in touched]
