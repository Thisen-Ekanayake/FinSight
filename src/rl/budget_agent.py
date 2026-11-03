# ═══════════════════════════════════════════════════════
# FinSight — Budget & savings planner (contextual bandit)
# ═══════════════════════════════════════════════════════
#
# Purpose : Choose a monthly savings rate for one household, then turn it
#           into concrete category limits and goal contributions.
#
# Public API:
#   HouseholdProfile, profile_from_ledger(...), BudgetConfig,
#   train_budget_agent(profile, config, on_progress) -> dict
#
# ══ WHY A BANDIT, AND WHY A SIMULATOR ══
#   One month's budget barely changes next month's situation, so the decision
#   is (almost) a one-step problem: a contextual bandit, not a full MDP. But a
#   household produces ONE month of feedback per month — years of real data
#   for a bandit to learn anything. So the agent learns in a simulator fitted
#   to the household's own ledger (income level and volatility, how much of
#   its spending is fixed needs vs flexible wants, how urgent its goals are),
#   with the context perturbed around those values so the policy generalises
#   rather than memorising one point.
#
# ══ THE MONTH, SIMULATED ══
#   arm      a savings rate r ∈ {0%, 5%, …, 50%} of income, set aside first.
#   needs    drawn from the household's own needs distribution — inelastic.
#   left     income − savings − needs: the discretionary money.
#   shortfall  needs that did not fit after saving: paid back out of savings.
#   reward   log utility of discretionary money
#            + goal progress up to what keeps goals on schedule (worth more
#              while the emergency buffer is thin)
#            + log utility of saving beyond that
#            − a heavy penalty on shortfall. All in units of monthly income.
#
# ══ THE LEARNER ══
#   Linear Thompson sampling: one Bayesian linear regression per arm over the
#   context, an action chosen by sampling each arm's weights from its
#   posterior. It explores where it is uncertain and exploits where not.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import numpy as np

ProgressFn = Callable[[int, int, float], None]

RATES = [round(0.05 * i, 2) for i in range(11)]  # 0% … 50%
FALLBACK_INCOME = 4000.0


@dataclass(frozen=True)
class GoalNeed:
    name: str
    goal_id: int
    remaining: float
    months_left: int | None  # None: no deadline


@dataclass(frozen=True)
class HouseholdProfile:
    income_mu: float
    income_sd: float
    needs_mu: float
    needs_sd: float
    wants_mu: float
    needs_by_category: dict[str, float]
    wants_by_category: dict[str, float]
    savings_categories: list[str]
    goals: list[GoalNeed]
    emergency_buffer_months: float
    months_of_history: int
    assumed_income: bool = False

    @property
    def goal_required(self) -> float:
        """Monthly contribution that keeps every goal on schedule (24 months for undated ones)."""
        return sum(g.remaining / max(1, g.months_left if g.months_left is not None else 24) for g in self.goals)


def _months_until(deadline: str | None, today: date) -> int | None:
    if not deadline:
        return None
    d = date.fromisoformat(deadline)
    return max(1, (d.year - today.year) * 12 + d.month - today.month)


def profile_from_ledger(
    entries: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    *,
    today: date | None = None,
    months: int = 6,
) -> HouseholdProfile:
    """
    Summarise a ledger into the distributions the simulator draws from.

    Uses up to ``months`` complete-or-current months. With no income recorded,
    income is assumed from the budget limits (and flagged) so a brand-new
    account still gets a plan.
    """
    today = today or date.today()
    kinds = {c["name"]: c["kind"] for c in categories}

    by_month: dict[str, dict[str, float]] = {}
    income_by_month: dict[str, float] = {}
    for e in entries:
        month = e["occurred_on"][:7]
        if e["direction"] == "income":
            income_by_month[month] = income_by_month.get(month, 0.0) + e["amount"]
        else:
            cat = e["category"] or "Uncategorised"
            by_month.setdefault(month, {})
            by_month[month][cat] = by_month[month].get(cat, 0.0) + e["amount"]

    recent = sorted(set(by_month) | set(income_by_month))[-months:]
    n_months = len(recent)

    def per_category_mean(kind: str) -> dict[str, float]:
        out: dict[str, float] = {}
        names = {c["name"] for c in categories if c["kind"] == kind}
        if kind == "need":
            # Untagged spending is treated as a need: conservative.
            names |= {cat for m in by_month.values() for cat in m if cat not in kinds}
        for name in names:
            # Averaged over the months the category was actually used: a bill
            # logged once in four months is a monthly bill someone forgot to
            # log, not a quarter-sized one.
            spent = [v for v in (by_month.get(m, {}).get(name, 0.0) for m in recent) if v > 0]
            limit = next((c["monthly_limit"] for c in categories if c["name"] == name), 0.0)
            mean = float(np.mean(spent)) if spent else float(limit)
            if kind == "need":
                # A need's limit is the person's own statement of what the bill
                # costs. Sparse logging must not talk the plan into cutting it:
                # writing a smaller rent limit does not make rent cheaper.
                mean = max(mean, float(limit))
            if mean > 0:
                out[name] = mean
        return out

    needs = per_category_mean("need")
    wants = per_category_mean("want")
    needs_totals = [sum(by_month.get(m, {}).get(n, 0.0) for n in needs) for m in recent] or [sum(needs.values())]

    incomes = [income_by_month[m] for m in recent if m in income_by_month]
    assumed = not incomes
    if assumed:
        planned = sum(c["monthly_limit"] for c in categories)
        incomes = [planned * 1.15 if planned > 0 else FALLBACK_INCOME]
    income_mu = float(np.mean(incomes))
    income_sd = float(np.std(incomes)) if len(incomes) > 1 else 0.05 * income_mu

    needs_mu = sum(needs.values())
    needs_sd = max(float(np.std(needs_totals)) if len(needs_totals) > 1 else 0.0, 0.05 * needs_mu)

    goal_needs = [
        GoalNeed(
            name=g["name"],
            goal_id=g["id"],
            remaining=max(0.0, g["target"] - g["saved"]),
            months_left=_months_until(g.get("deadline"), today),
        )
        for g in goals
        if g["target"] > g["saved"]
    ]
    emergency = next((g for g in goals if "emergency" in g["name"].lower()), None)
    buffer_months = (emergency["saved"] / needs_mu) if emergency and needs_mu > 0 else 0.0

    return HouseholdProfile(
        income_mu=max(income_mu, 1.0),
        income_sd=max(income_sd, 0.01 * income_mu),
        needs_mu=needs_mu,
        needs_sd=needs_sd,
        wants_mu=sum(wants.values()),
        needs_by_category=needs,
        wants_by_category=wants,
        savings_categories=[c["name"] for c in categories if c["kind"] == "savings"],
        goals=goal_needs,
        emergency_buffer_months=buffer_months,
        months_of_history=n_months,
        assumed_income=assumed,
    )


# ── Simulator ───────────────────────────────────────────


@dataclass
class _Context:
    income_mu: float
    income_sd: float
    needs_mu: float
    needs_sd: float
    wants_mu: float
    goal_required: float
    buffer_months: float

    def features(self) -> np.ndarray:
        inc = self.income_mu
        base = np.array(
            [
                self.needs_mu / inc,
                self.wants_mu / inc,
                min(self.goal_required / inc, 1.5),
                min(self.buffer_months / 3.0, 2.0),
                self.income_sd / inc,
            ]
        )
        # Squares and pairwise products: the reward is curved in the context
        # (log utilities, a kink where goals are met), and a per-arm model that
        # is only linear in it lands an arm or two off the best one.
        pairs = np.outer(base, base)[np.triu_indices(len(base))]
        return np.concatenate([[1.0], base, pairs])


def _base_context(p: HouseholdProfile) -> _Context:
    return _Context(
        p.income_mu, p.income_sd, p.needs_mu, p.needs_sd, p.wants_mu, p.goal_required, p.emergency_buffer_months
    )


def _perturb(base: _Context, rng: np.random.Generator, spread: float = 1.0) -> _Context:
    """
    A nearby household. ``spread`` = 1 is the widest; it anneals toward 0 so
    practice starts broad (to generalise) and ends on this household's own
    situation (so the estimates that decide the recommendation are sharp).
    """

    def jitter(lo: float, hi: float) -> float:
        return 1.0 + spread * (rng.uniform(lo, hi) - 1.0)

    buffer = base.buffer_months * jitter(0.0, 2.0) if base.buffer_months else spread * rng.uniform(0.0, 4.0)
    return _Context(
        income_mu=base.income_mu * jitter(0.75, 1.25),
        income_sd=base.income_sd * jitter(0.5, 2.0),
        needs_mu=base.needs_mu * jitter(0.8, 1.2),
        needs_sd=base.needs_sd * jitter(0.5, 2.0),
        wants_mu=base.wants_mu * jitter(0.6, 1.4),
        goal_required=base.goal_required * jitter(0.4, 1.6),
        buffer_months=buffer,
    )


def simulate_month(ctx: _Context, rate: float, rng: np.random.Generator) -> dict[str, float]:
    """One month under savings rate ``rate``. Rewards are in units of mean income."""
    income = max(0.0, rng.normal(ctx.income_mu, ctx.income_sd))
    needs = max(0.0, rng.normal(ctx.needs_mu, ctx.needs_sd))
    target = rate * income
    left = income - target - needs
    shortfall = max(0.0, -left)
    wants = min(max(left, 0.0), ctx.wants_mu * rng.uniform(0.85, 1.15))
    saved = target - shortfall  # a shortfall is paid back out of what was set aside

    scale = ctx.income_mu
    urgency = 1.0 + max(0.0, 1.0 - ctx.buffer_months / 3.0)  # thin buffer: saving matters more
    on_schedule = min(max(saved, 0.0), ctx.goal_required)
    beyond = max(0.0, saved - ctx.goal_required)
    discretionary = max(left, 0.0)

    # Diminishing returns on both sides: the first dollars of discretionary
    # money and of extra saving are worth the most, so the optimum is interior
    # — around the familiar 20% for a comfortable household with no goals,
    # higher when goals are behind or the buffer is thin, lower when tight.
    reward = (
        0.4 * math.log1p(discretionary / (0.1 * scale))
        + 1.6 * urgency * on_schedule / scale
        + 0.15 * urgency * math.log1p(beyond / (0.05 * scale))
        - 3.0 * shortfall / scale
    )
    return {"reward": reward, "saved": saved, "shortfall": shortfall, "wants": wants}


def expected_rewards(ctx: _Context, rng: np.random.Generator, samples: int = 400) -> list[float]:
    """Monte-Carlo expected reward of every arm in one context — the simulator's ground truth."""
    return [float(np.mean([simulate_month(ctx, r, rng)["reward"] for _ in range(samples)])) for r in RATES]


# ── Learner ─────────────────────────────────────────────


@dataclass
class _LinTS:
    dim: int
    arms: int
    lam: float = 1.0
    noise: float = 0.25
    A: list[np.ndarray] = field(default_factory=list)
    b: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.A = [self.lam * np.eye(self.dim) for _ in range(self.arms)]
        self.b = [np.zeros(self.dim) for _ in range(self.arms)]

    def mean(self, arm: int) -> np.ndarray:
        return np.asarray(np.linalg.solve(self.A[arm], self.b[arm]))

    def choose(self, x: np.ndarray, rng: np.random.Generator) -> int:
        scores = []
        for arm in range(self.arms):
            cov = self.noise**2 * np.linalg.inv(self.A[arm])
            theta = rng.multivariate_normal(self.mean(arm), cov)
            scores.append(float(theta @ x))
        return int(np.argmax(scores))

    def greedy(self, x: np.ndarray) -> list[float]:
        return [float(self.mean(arm) @ x) for arm in range(self.arms)]

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x


@dataclass(frozen=True)
class BudgetConfig:
    rounds: int = 3000
    report_every: int = 50
    seed: int = 11


def _plan(profile: HouseholdProfile, rate: float) -> dict[str, Any]:
    """Turn a savings rate into category limits, goal contributions and projections."""
    income = profile.income_mu
    savings = rate * income

    # Goals with deadlines first, in proportion to what keeps each on schedule;
    # anything left over goes to undated goals, then to "investing".
    contributions: dict[int, float] = {}
    dated = [g for g in profile.goals if g.months_left is not None]
    undated = [g for g in profile.goals if g.months_left is None]
    need = {g.goal_id: g.remaining / max(1, g.months_left or 1) for g in dated}
    total_need = sum(need.values())
    pool = savings
    if total_need > 0:
        share = min(1.0, pool / total_need)
        for gid, amount in need.items():
            contributions[gid] = amount * share
        pool -= sum(contributions.values())
    if undated and pool > 0:
        each = pool / len(undated)
        for g in undated:
            contributions[g.goal_id] = min(each, g.remaining)
        pool -= sum(contributions[g.goal_id] for g in undated)
    unallocated = max(0.0, pool)

    goals_out = []
    for g in profile.goals:
        monthly = contributions.get(g.goal_id, 0.0)
        months = math.ceil(g.remaining / monthly) if monthly > 0 else None
        goals_out.append(
            {
                "goal_id": g.goal_id,
                "name": g.name,
                "monthly": round(monthly, 2),
                "remaining": round(g.remaining, 2),
                "months_to_complete": months,
                "deadline_months": g.months_left,
                "on_track": g.months_left is None or (months is not None and months <= g.months_left),
            }
        )

    limits: dict[str, float] = {}
    for name, mean in profile.needs_by_category.items():
        limits[name] = round(mean * 1.05 / 5) * 5  # needs: recent average + 5% headroom
    needs_total = sum(limits.values())
    room = max(0.0, income - savings - needs_total)
    wants_total = sum(profile.wants_by_category.values())
    # Wants get up to 30% more than recent spending; anything beyond that is
    # left unassigned rather than inflating a category nobody asked to grow.
    wants_budget = min(room, wants_total * 1.3) if wants_total else 0.0
    for name, mean in profile.wants_by_category.items():
        limits[name] = round((wants_budget * mean / wants_total) / 5) * 5
    if profile.savings_categories:
        each = savings / len(profile.savings_categories)
        for name in profile.savings_categories:
            limits[name] = round(each / 5) * 5

    return {
        "savings_rate": rate,
        "monthly_savings": round(savings, 2),
        "unallocated_savings": round(unallocated, 2),
        "needs_total": round(needs_total, 2),
        "wants_budget": round(wants_budget, 2),
        "unassigned": round(max(0.0, room - wants_budget), 2),
        "category_limits": limits,
        "goals": goals_out,
    }


def train_budget_agent(
    profile: HouseholdProfile,
    config: BudgetConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Train the bandit in the simulator and recommend a plan for the household's actual context."""
    cfg = config or BudgetConfig()
    rng = np.random.default_rng(cfg.seed)
    base = _base_context(profile)
    agent = _LinTS(dim=len(base.features()), arms=len(RATES))
    rule_arm = RATES.index(0.2)

    curve: list[float] = []
    rule_curve: list[float] = []
    window: list[float] = []
    rule_window: list[float] = []
    for t in range(1, cfg.rounds + 1):
        ctx = _perturb(base, rng, spread=max(0.1, 1.0 - t / (0.7 * cfg.rounds)))
        x = ctx.features()
        arm = agent.choose(x, rng)
        outcome = simulate_month(ctx, RATES[arm], rng)
        agent.update(arm, x, outcome["reward"])
        window.append(outcome["reward"])
        # The 50/30/20 rule's 20% on the same context, for comparison.
        rule_window.append(simulate_month(ctx, RATES[rule_arm], rng)["reward"])
        if t % cfg.report_every == 0 or t == cfg.rounds:
            curve.append(round(float(np.mean(window)), 5))
            rule_curve.append(round(float(np.mean(rule_window)), 5))
            window, rule_window = [], []
            if on_progress:
                on_progress(t, cfg.rounds, curve[-1])

    x_now = base.features()
    learned = agent.greedy(x_now)
    best = int(np.argmax(learned))
    truth = expected_rewards(base, np.random.default_rng(cfg.seed + 1))
    oracle = int(np.argmax(truth))

    return {
        "profile": {
            "income": round(profile.income_mu, 2),
            "income_volatility": round(profile.income_sd / profile.income_mu, 4),
            "needs": round(profile.needs_mu, 2),
            "wants": round(profile.wants_mu, 2),
            "goal_required": round(profile.goal_required, 2),
            "emergency_buffer_months": round(profile.emergency_buffer_months, 2),
            "months_of_history": profile.months_of_history,
            "assumed_income": profile.assumed_income,
        },
        "rates": RATES,
        "learned_value": [round(v, 4) for v in learned],
        "simulated_value": [round(v, 4) for v in truth],
        "recommended_rate": RATES[best],
        "best_simulated_rate": RATES[oracle],
        "rule_rate": RATES[rule_arm],
        "value_vs_rule": round(truth[best] - truth[rule_arm], 4),
        "learning_curve": curve,
        "rule_curve": rule_curve,
        "plan": _plan(profile, RATES[best]),
    }
