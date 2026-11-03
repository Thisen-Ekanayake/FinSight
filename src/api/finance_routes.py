# ═══════════════════════════════════════════════════════
# FinSight — Finance management routes
# ═══════════════════════════════════════════════════════
#
# Purpose : Budgets, ledger, goals (/finance), RL agents (/rl), and the Alpaca
#           paper broker (/broker). Every route resolves the caller and scopes
#           every read and write to their Google ``sub``.
#
# ══ NOT FINANCIAL ADVICE ══
#   Agents here learn from past prices and PROPOSE. A person approves every
#   order, and orders only ever go to a paper (simulated-money) account.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.auth import CurrentIdentity
from src.broker import service as broker
from src.finance import repository as repo
from src.rl import alert_service
from src.rl import service as rl

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TICKER = re.compile(r"^[A-Za-z.\-]{1,10}$")

DISCLAIMER = (
    "Educational tool, not financial advice. Agents learn from past prices only; "
    "orders go to an Alpaca PAPER account with simulated money and require your approval."
)

finance_router = APIRouter(prefix="/finance", tags=["finance"])
rl_router = APIRouter(prefix="/rl", tags=["agents"])
broker_router = APIRouter(prefix="/broker", tags=["broker"])
# Alerts are the monitor's, and the monitor is the unlimited tier's — this
# router is mounted with the same `reserved` guard in src/api/main.py.
alerts_router = APIRouter(prefix="/rl/alerts", tags=["agents"])


def _this_month() -> str:
    return date.today().strftime("%Y-%m")


# ── Schemas ─────────────────────────────────────────────


class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    monthly_limit: float = Field(ge=0, le=10_000_000)
    kind: Literal["need", "want", "savings"] = "need"


class EntryIn(BaseModel):
    occurred_on: str
    amount: float = Field(gt=0, le=100_000_000)
    direction: Literal["income", "expense"] = "expense"
    category: str = Field(default="", max_length=64)
    note: str = Field(default="", max_length=256)

    @field_validator("occurred_on")
    @classmethod
    def _day(cls, value: str) -> str:
        if not _DAY.match(value):
            raise ValueError("occurred_on must be YYYY-MM-DD")
        date.fromisoformat(value)
        return value


class GoalIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    target: float = Field(gt=0, le=100_000_000)
    saved: float = Field(default=0, ge=0)
    deadline: str | None = None

    @field_validator("deadline")
    @classmethod
    def _deadline(cls, value: str | None) -> str | None:
        if value:
            date.fromisoformat(value)
        return value or None


class ContributionIn(BaseModel):
    amount: float = Field(ge=-100_000_000, le=100_000_000)


class TrainIn(BaseModel):
    tickers: list[str] = Field(min_length=rl.MIN_ASSETS, max_length=rl.MAX_ASSETS)
    risk_aversion: float = Field(default=1.0, ge=0, le=10)
    episodes: int = Field(default=600, ge=50, le=rl.MAX_EPISODES)
    cost_bps: float = Field(default=10, ge=0, le=100)
    # "ppo": neural actor-critic (default). "reinforce": the linear baseline.
    algorithm: Literal["ppo", "reinforce"] = "ppo"

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, value: list[str]) -> list[str]:
        for ticker in value:
            if not _TICKER.match(ticker.strip()):
                raise ValueError(f"{ticker!r} is not a ticker symbol")
        return value


class BudgetTrainIn(BaseModel):
    rounds: int = Field(default=3000, ge=500, le=10_000)


class ApplyIn(BaseModel):
    run_id: str


class FeedbackIn(BaseModel):
    # null clears the feedback.
    useful: bool | None


class LinkIn(BaseModel):
    key_id: str = Field(min_length=8, max_length=64)
    secret: str = Field(min_length=8, max_length=128)


class PauseIn(BaseModel):
    paused: bool


class ProposeIn(BaseModel):
    run_id: str


class DecideIn(BaseModel):
    approve: list[str] = Field(default_factory=list, max_length=100)
    reject: list[str] = Field(default_factory=list, max_length=100)


# ── /finance ────────────────────────────────────────────


@finance_router.get("/summary", summary="Month at a glance")
async def get_summary(identity: CurrentIdentity, month: str | None = None) -> dict[str, Any]:
    month = month or _this_month()
    if not _MONTH.match(month):
        raise HTTPException(422, "month must be YYYY-MM")
    summary = repo.month_summary(subject=identity.subject, month=month)
    goals = repo.list_goals(subject=identity.subject)
    summary["goals"] = goals
    summary["goals_saved"] = round(sum(g["saved"] for g in goals), 2)
    summary["goals_target"] = round(sum(g["target"] for g in goals), 2)
    summary["disclaimer"] = DISCLAIMER
    return summary


@finance_router.get("/categories")
async def get_categories(identity: CurrentIdentity) -> list[dict[str, Any]]:
    return repo.list_categories(subject=identity.subject)


@finance_router.put("/categories", summary="Create or update a budget category")
async def put_category(body: CategoryIn, identity: CurrentIdentity) -> dict[str, Any]:
    return repo.upsert_category(
        subject=identity.subject, name=body.name.strip(), monthly_limit=body.monthly_limit, kind=body.kind
    )


@finance_router.delete("/categories/{category_id}", status_code=204)
async def delete_category(category_id: int, identity: CurrentIdentity) -> None:
    if not repo.delete_category(subject=identity.subject, category_id=category_id):
        raise HTTPException(404, "Category not found")


@finance_router.get("/entries")
async def get_entries(identity: CurrentIdentity, month: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if month and not _MONTH.match(month):
        raise HTTPException(422, "month must be YYYY-MM")
    return repo.list_entries(subject=identity.subject, month=month, limit=max(1, min(limit, 1000)))


@finance_router.post("/entries", status_code=201)
async def post_entry(body: EntryIn, identity: CurrentIdentity) -> dict[str, Any]:
    return repo.add_entry(subject=identity.subject, **body.model_dump())


@finance_router.delete("/entries/{entry_id}", status_code=204)
async def delete_entry(entry_id: int, identity: CurrentIdentity) -> None:
    if not repo.delete_entry(subject=identity.subject, entry_id=entry_id):
        raise HTTPException(404, "Entry not found")


@finance_router.get("/goals")
async def get_goals(identity: CurrentIdentity) -> list[dict[str, Any]]:
    return repo.list_goals(subject=identity.subject)


@finance_router.post("/goals", status_code=201)
async def post_goal(body: GoalIn, identity: CurrentIdentity) -> dict[str, Any]:
    return repo.add_goal(subject=identity.subject, **body.model_dump())


@finance_router.post("/goals/{goal_id}/contribute")
async def post_contribution(goal_id: int, body: ContributionIn, identity: CurrentIdentity) -> dict[str, Any]:
    goal = repo.contribute_to_goal(subject=identity.subject, goal_id=goal_id, amount=body.amount)
    if goal is None:
        raise HTTPException(404, "Goal not found")
    return goal


@finance_router.delete("/goals/{goal_id}", status_code=204)
async def delete_goal(goal_id: int, identity: CurrentIdentity) -> None:
    if not repo.delete_goal(subject=identity.subject, goal_id=goal_id):
        raise HTTPException(404, "Goal not found")


# ── /rl ─────────────────────────────────────────────────


@rl_router.get("/agents", summary="The RL agents and which can train today")
async def get_agents() -> list[dict[str, Any]]:
    return rl.AGENTS


@rl_router.post("/runs", status_code=202, summary="Start training the portfolio agent")
async def post_run(body: TrainIn, identity: CurrentIdentity) -> dict[str, Any]:
    try:
        run_id = rl.start_portfolio_run(
            subject=identity.subject,
            tickers=body.tickers,
            risk_aversion=body.risk_aversion,
            episodes=body.episodes,
            cost_bps=body.cost_bps,
            algorithm=body.algorithm,
        )
    except rl.TrainingRefused as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    run = repo.get_run(subject=identity.subject, run_id=run_id)
    assert run is not None
    return run


@rl_router.post("/timing/runs", status_code=202, summary="Start training the trade-timing agent (signals only)")
async def post_timing_run(body: TrainIn, identity: CurrentIdentity) -> dict[str, Any]:
    try:
        run_id = rl.start_timing_run(
            subject=identity.subject,
            tickers=body.tickers,
            risk_aversion=body.risk_aversion,
            episodes=body.episodes,
            cost_bps=body.cost_bps,
            algorithm=body.algorithm,
        )
    except rl.TrainingRefused as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    run = repo.get_run(subject=identity.subject, run_id=run_id)
    assert run is not None
    return run


@rl_router.post("/budget/runs", status_code=202, summary="Train the budget planner on your own ledger")
async def post_budget_run(body: BudgetTrainIn, identity: CurrentIdentity) -> dict[str, Any]:
    try:
        run_id = rl.start_budget_run(subject=identity.subject, rounds=body.rounds)
    except rl.TrainingRefused as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    run = repo.get_run(subject=identity.subject, run_id=run_id)
    assert run is not None
    return run


@rl_router.post("/budget/apply", summary="Apply a budget plan's category limits")
async def post_budget_apply(body: ApplyIn, identity: CurrentIdentity) -> list[dict[str, Any]]:
    """
    Write the plan's recommended monthly limits onto your categories.

    Only limits change — no money moves, and categories the plan does not
    mention are left alone. Re-applying an older plan simply rewrites them.
    """
    run = repo.get_run(subject=identity.subject, run_id=body.run_id)
    if run is None or run["agent"] != "budget":
        raise HTTPException(404, "Budget plan not found")
    if run["status"] != "DONE" or not run["result"]:
        raise HTTPException(409, "That plan has not finished training")
    kinds = {c["name"]: c["kind"] for c in repo.list_categories(subject=identity.subject)}
    for name, limit in run["result"]["plan"]["category_limits"].items():
        if name in kinds:
            repo.upsert_category(subject=identity.subject, name=name, monthly_limit=float(limit), kind=kinds[name])
    return repo.list_categories(subject=identity.subject)


@rl_router.get("/runs")
async def get_runs(identity: CurrentIdentity, limit: int = 20, agent: str | None = None) -> list[dict[str, Any]]:
    runs = repo.list_runs(subject=identity.subject, limit=max(1, min(limit, 100)), agent=agent)
    return runs


@rl_router.get("/runs/{run_id}", summary="One run — poll this for live progress")
async def get_run(run_id: str, identity: CurrentIdentity) -> dict[str, Any]:
    run = repo.get_run(subject=identity.subject, run_id=run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run


# ── /broker ─────────────────────────────────────────────


def _translate(exc: broker.ServiceError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@broker_router.get("/status", summary="Paper account, positions, guardrails")
async def get_status(identity: CurrentIdentity) -> dict[str, Any]:
    return {**broker.status(subject=identity.subject), "disclaimer": DISCLAIMER}


@broker_router.post("/link", summary="Link Alpaca PAPER keys (verified, then stored encrypted)")
async def post_link(body: LinkIn, identity: CurrentIdentity) -> dict[str, Any]:
    try:
        account = broker.link(subject=identity.subject, key_id=body.key_id, secret=body.secret)
    except broker.ServiceError as exc:
        raise _translate(exc) from exc
    return {"linked": True, "account": account}


@broker_router.delete("/link", status_code=204)
async def delete_link(identity: CurrentIdentity) -> None:
    if not broker.unlink(subject=identity.subject):
        raise HTTPException(404, "No account is linked")


@broker_router.post("/pause", summary="Per-account kill switch")
async def post_pause(body: PauseIn, identity: CurrentIdentity) -> dict[str, bool]:
    try:
        broker.set_paused(subject=identity.subject, paused=body.paused)
    except broker.ServiceError as exc:
        raise _translate(exc) from exc
    return {"paused": body.paused}


@broker_router.post("/proposals", summary="Turn a finished run into PENDING order proposals")
async def post_proposals(body: ProposeIn, identity: CurrentIdentity) -> list[dict[str, Any]]:
    try:
        return broker.propose(subject=identity.subject, run_id=body.run_id)
    except broker.ServiceError as exc:
        raise _translate(exc) from exc


@broker_router.get("/proposals")
async def get_proposals(identity: CurrentIdentity, status: str | None = None) -> list[dict[str, Any]]:
    return repo.list_proposals(subject=identity.subject, status=status)


@broker_router.post("/proposals/decide", summary="Approve (submit) or reject proposals")
async def post_decide(body: DecideIn, identity: CurrentIdentity) -> list[dict[str, Any]]:
    try:
        return broker.decide(subject=identity.subject, approve=body.approve, reject=body.reject)
    except broker.ServiceError as exc:
        raise _translate(exc) from exc


# ── /rl/alerts ──────────────────────────────────────────


@alerts_router.get("/ranked", summary="Recent alerts, ranked by learned usefulness")
async def get_ranked(identity: CurrentIdentity, limit: int = 50, explore: bool = False) -> dict[str, Any]:
    return alert_service.ranked_alerts(subject=identity.subject, limit=max(1, min(limit, 200)), explore=explore)


@alerts_router.get("/preferences", summary="What the prioritiser has learned")
async def get_preferences(identity: CurrentIdentity) -> dict[str, Any]:
    return alert_service.learned_preferences(subject=identity.subject)


@alerts_router.post("/{alert_id}/feedback", summary="Mark an alert useful / not useful (null clears)")
async def post_feedback(alert_id: str, body: FeedbackIn, identity: CurrentIdentity) -> dict[str, Any]:
    from src.persistence.repository import get_alert

    if get_alert(alert_id) is None:
        raise HTTPException(404, "Alert not found")
    if body.useful is None:
        repo.delete_alert_feedback(subject=identity.subject, alert_id=alert_id)
    else:
        repo.record_alert_feedback(subject=identity.subject, alert_id=alert_id, useful=body.useful)
    return {"alert_id": alert_id, "useful": body.useful}
