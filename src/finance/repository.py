# ═══════════════════════════════════════════════════════
# FinSight — Finance repository
# ═══════════════════════════════════════════════════════
#
# Purpose : Every read and write of the finance tables. Each function takes
#           ``subject`` as a required keyword and filters on it, so no caller
#           can forget to scope a query to its owner.
#
# Returns plain dicts, never ORM objects — sessions are short and a detached
# instance read after its session closes is a bug waiting to happen.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, update

from src.persistence.db import session_scope
from src.persistence.models import (
    BrokerLink,
    BudgetCategory,
    LedgerEntry,
    OrderProposal,
    PolicyRun,
    SavingsGoal,
    utcnow,
)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


# ── Budget categories ───────────────────────────────────


def _category(row: BudgetCategory) -> dict[str, Any]:
    return {"id": row.id, "name": row.name, "monthly_limit": row.monthly_limit, "kind": row.kind}


def list_categories(*, subject: str) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.scalars(select(BudgetCategory).where(BudgetCategory.subject == subject).order_by(BudgetCategory.name))
        return [_category(r) for r in rows]


def upsert_category(*, subject: str, name: str, monthly_limit: float, kind: str) -> dict[str, Any]:
    with session_scope() as s:
        row = s.scalar(select(BudgetCategory).where(BudgetCategory.subject == subject, BudgetCategory.name == name))
        if row is None:
            row = BudgetCategory(subject=subject, name=name)
            s.add(row)
        row.monthly_limit = monthly_limit
        row.kind = kind
        s.flush()
        return _category(row)


def delete_category(*, subject: str, category_id: int) -> bool:
    with session_scope() as s:
        row = s.scalar(
            select(BudgetCategory).where(BudgetCategory.subject == subject, BudgetCategory.id == category_id)
        )
        if row is None:
            return False
        s.delete(row)
        return True


# ── Ledger ──────────────────────────────────────────────


def _entry(row: LedgerEntry) -> dict[str, Any]:
    return {
        "id": row.id,
        "occurred_on": row.occurred_on,
        "amount": row.amount,
        "direction": row.direction,
        "category": row.category,
        "note": row.note,
    }


def list_entries(*, subject: str, month: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    with session_scope() as s:
        query = select(LedgerEntry).where(LedgerEntry.subject == subject)
        if month:
            query = query.where(LedgerEntry.occurred_on.like(f"{month}-%"))
        query = query.order_by(LedgerEntry.occurred_on.desc(), LedgerEntry.id.desc()).limit(limit)
        return [_entry(r) for r in s.scalars(query)]


def add_entry(
    *, subject: str, occurred_on: str, amount: float, direction: str, category: str, note: str
) -> dict[str, Any]:
    with session_scope() as s:
        row = LedgerEntry(
            subject=subject,
            occurred_on=occurred_on,
            amount=amount,
            direction=direction,
            category=category,
            note=note,
        )
        s.add(row)
        s.flush()
        return _entry(row)


def delete_entry(*, subject: str, entry_id: int) -> bool:
    with session_scope() as s:
        row = s.scalar(select(LedgerEntry).where(LedgerEntry.subject == subject, LedgerEntry.id == entry_id))
        if row is None:
            return False
        s.delete(row)
        return True


# ── Goals ───────────────────────────────────────────────


def _goal(row: SavingsGoal) -> dict[str, Any]:
    return {"id": row.id, "name": row.name, "target": row.target, "saved": row.saved, "deadline": row.deadline}


def list_goals(*, subject: str) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.scalars(select(SavingsGoal).where(SavingsGoal.subject == subject).order_by(SavingsGoal.id))
        return [_goal(r) for r in rows]


def add_goal(*, subject: str, name: str, target: float, saved: float, deadline: str | None) -> dict[str, Any]:
    with session_scope() as s:
        row = SavingsGoal(subject=subject, name=name, target=target, saved=saved, deadline=deadline)
        s.add(row)
        s.flush()
        return _goal(row)


def contribute_to_goal(*, subject: str, goal_id: int, amount: float) -> dict[str, Any] | None:
    with session_scope() as s:
        row = s.scalar(select(SavingsGoal).where(SavingsGoal.subject == subject, SavingsGoal.id == goal_id))
        if row is None:
            return None
        row.saved = max(0.0, row.saved + amount)
        s.flush()
        return _goal(row)


def delete_goal(*, subject: str, goal_id: int) -> bool:
    with session_scope() as s:
        row = s.scalar(select(SavingsGoal).where(SavingsGoal.subject == subject, SavingsGoal.id == goal_id))
        if row is None:
            return False
        s.delete(row)
        return True


# ── Monthly summary ─────────────────────────────────────


def month_summary(*, subject: str, month: str) -> dict[str, Any]:
    """
    Income, spend and per-category progress for one ``YYYY-MM``.

    ``daily`` is cumulative spend by day-of-month next to an even-pace line,
    which is what the budget burn-down chart draws.
    """
    entries = list_entries(subject=subject, month=month, limit=10_000)
    categories = list_categories(subject=subject)

    income = sum(e["amount"] for e in entries if e["direction"] == "income")
    expenses = sum(e["amount"] for e in entries if e["direction"] == "expense")

    spent_by: dict[str, float] = {}
    for e in entries:
        if e["direction"] == "expense":
            spent_by[e["category"] or "Uncategorised"] = (
                spent_by.get(e["category"] or "Uncategorised", 0.0) + e["amount"]
            )

    rows = []
    for cat in categories:
        spent = spent_by.pop(cat["name"], 0.0)
        rows.append({**cat, "spent": round(spent, 2)})
    for name, spent in sorted(spent_by.items()):
        rows.append({"id": None, "name": name, "monthly_limit": 0.0, "kind": "need", "spent": round(spent, 2)})

    year, mon = (int(x) for x in month.split("-"))
    first = date(year, mon, 1)
    days = ((first.replace(day=28) + timedelta(days=4)).replace(day=1) - first).days
    per_day = [0.0] * days
    for e in entries:
        if e["direction"] == "expense":
            day = int(e["occurred_on"][8:10])
            if 1 <= day <= days:
                per_day[day - 1] += e["amount"]
    running, cumulative = 0.0, []
    for value in per_day:
        running += value
        cumulative.append(round(running, 2))

    total_limit = sum(c["monthly_limit"] for c in categories)
    return {
        "month": month,
        "income": round(income, 2),
        "expenses": round(expenses, 2),
        "net": round(income - expenses, 2),
        "savings_rate": round((income - expenses) / income, 4) if income > 0 else 0.0,
        "budget_total": round(total_limit, 2),
        "categories": rows,
        "daily": {
            "spent": cumulative,
            "pace": [round(total_limit * (i + 1) / days, 2) for i in range(days)],
        },
    }


# ── RL policy runs ──────────────────────────────────────


def _run(row: PolicyRun) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "agent": row.agent,
        "tickers": [t for t in row.tickers.split(",") if t],
        "params": json.loads(row.params or "{}"),
        "status": row.status,
        "stage": row.stage,
        "episodes_done": row.episodes_done,
        "episodes_total": row.episodes_total,
        "progress": round(row.episodes_done / row.episodes_total, 4) if row.episodes_total else 0.0,
        "curve": json.loads(row.curve or "[]"),
        "result": json.loads(row.result) if row.result else None,
        "error": row.error,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def create_run(*, subject: str, agent: str, tickers: list[str], params: dict[str, Any], episodes: int) -> str:
    run_id = uuid.uuid4().hex
    with session_scope() as s:
        s.add(
            PolicyRun(
                run_id=run_id,
                subject=subject,
                agent=agent,
                tickers=",".join(tickers),
                params=json.dumps(params),
                episodes_total=episodes,
                stage="Queued",
            )
        )
    return run_id


def update_run(run_id: str, **fields: Any) -> None:
    """Patch a run from the training thread. JSON fields are serialised here."""
    for key in ("curve", "result"):
        if key in fields and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    fields["updated_at"] = utcnow()
    with session_scope() as s:
        s.execute(update(PolicyRun).where(PolicyRun.run_id == run_id).values(**fields))


def get_run(*, subject: str, run_id: str) -> dict[str, Any] | None:
    with session_scope() as s:
        row = s.scalar(select(PolicyRun).where(PolicyRun.subject == subject, PolicyRun.run_id == run_id))
        return _run(row) if row else None


def list_runs(*, subject: str, limit: int = 20) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.scalars(
            select(PolicyRun).where(PolicyRun.subject == subject).order_by(PolicyRun.created_at.desc()).limit(limit)
        )
        out = []
        for row in rows:
            item = _run(row)
            # The list view needs headline numbers, not the full series.
            if item["result"]:
                item["result"] = {
                    "target_weights": item["result"].get("target_weights"),
                    "metrics": item["result"].get("metrics"),
                }
            item["curve"] = item["curve"][-40:]
            out.append(item)
        return out


def count_active_runs(*, subject: str | None = None) -> int:
    with session_scope() as s:
        query = select(func.count()).select_from(PolicyRun).where(PolicyRun.status.in_(("QUEUED", "RUNNING")))
        if subject is not None:
            query = query.where(PolicyRun.subject == subject)
        return int(s.scalar(query) or 0)


def fail_stale_runs() -> int:
    """Mark runs a previous process left RUNNING as failed — their thread died with it."""
    with session_scope() as s:
        result = s.execute(
            update(PolicyRun)
            .where(PolicyRun.status.in_(("QUEUED", "RUNNING")))
            .values(status="FAILED", error="Interrupted by a server restart.", updated_at=utcnow())
        )
        return int(getattr(result, "rowcount", 0) or 0)


# ── Broker link ─────────────────────────────────────────


def get_broker_link(*, subject: str) -> dict[str, Any] | None:
    with session_scope() as s:
        row = s.scalar(select(BrokerLink).where(BrokerLink.subject == subject))
        if row is None:
            return None
        return {
            "key_id_enc": row.key_id_enc,
            "secret_enc": row.secret_enc,
            "key_hint": row.key_hint,
            "trading_paused": row.trading_paused,
            "created_at": _iso(row.created_at),
        }


def save_broker_link(*, subject: str, key_id_enc: str, secret_enc: str, key_hint: str) -> None:
    with session_scope() as s:
        row = s.scalar(select(BrokerLink).where(BrokerLink.subject == subject))
        if row is None:
            row = BrokerLink(subject=subject, key_id_enc=key_id_enc, secret_enc=secret_enc)
            s.add(row)
        row.key_id_enc = key_id_enc
        row.secret_enc = secret_enc
        row.key_hint = key_hint


def delete_broker_link(*, subject: str) -> bool:
    with session_scope() as s:
        row = s.scalar(select(BrokerLink).where(BrokerLink.subject == subject))
        if row is None:
            return False
        s.delete(row)
        return True


def set_trading_paused(*, subject: str, paused: bool) -> bool:
    with session_scope() as s:
        result = s.execute(update(BrokerLink).where(BrokerLink.subject == subject).values(trading_paused=paused))
        return bool(getattr(result, "rowcount", 0))


# ── Order proposals ─────────────────────────────────────


def _proposal(row: OrderProposal) -> dict[str, Any]:
    return {
        "proposal_id": row.proposal_id,
        "run_id": row.run_id,
        "symbol": row.symbol,
        "side": row.side,
        "notional": row.notional,
        "current_weight": row.current_weight,
        "target_weight": row.target_weight,
        "reason": row.reason,
        "status": row.status,
        "broker_order_id": row.broker_order_id,
        "error": row.error,
        "created_at": _iso(row.created_at),
        "decided_at": _iso(row.decided_at),
    }


def replace_pending_proposals(*, subject: str, run_id: str, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Supersede every PENDING proposal for ``subject`` and record ``orders`` as the new set."""
    with session_scope() as s:
        s.execute(
            update(OrderProposal)
            .where(OrderProposal.subject == subject, OrderProposal.status == "PENDING")
            .values(status="SUPERSEDED", decided_at=utcnow())
        )
        rows = []
        for order in orders:
            row = OrderProposal(
                proposal_id=uuid.uuid4().hex,
                subject=subject,
                run_id=run_id,
                symbol=order["symbol"],
                side=order["side"],
                notional=order["notional"],
                current_weight=order["current_weight"],
                target_weight=order["target_weight"],
                reason=order["reason"],
            )
            s.add(row)
            rows.append(row)
        s.flush()
        return [_proposal(r) for r in rows]


def list_proposals(*, subject: str, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with session_scope() as s:
        query = select(OrderProposal).where(OrderProposal.subject == subject)
        if status:
            query = query.where(OrderProposal.status == status)
        query = query.order_by(OrderProposal.created_at.desc(), OrderProposal.id.desc()).limit(limit)
        return [_proposal(r) for r in s.scalars(query)]


def transition_proposal(*, subject: str, proposal_id: str, from_status: str, to_status: str, **fields: Any) -> bool:
    """
    Move a proposal between states IF it is still in ``from_status``.

    A conditional UPDATE, so two concurrent requests cannot both win — this is
    what makes "approve" safe to double-click.
    """
    with session_scope() as s:
        result = s.execute(
            update(OrderProposal)
            .where(
                OrderProposal.subject == subject,
                OrderProposal.proposal_id == proposal_id,
                OrderProposal.status == from_status,
            )
            .values(status=to_status, decided_at=utcnow(), **fields)
        )
        return bool(getattr(result, "rowcount", 0))


def count_submitted_today(*, subject: str) -> int:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    with session_scope() as s:
        return int(
            s.scalar(
                select(func.count())
                .select_from(OrderProposal)
                .where(
                    OrderProposal.subject == subject,
                    OrderProposal.status.in_(("SUBMITTING", "SUBMITTED")),
                    OrderProposal.decided_at >= start,
                )
            )
            or 0
        )
