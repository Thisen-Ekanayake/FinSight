# ═══════════════════════════════════════════════════════
# FinSight — Tests: RL portfolio environment and agent
# ═══════════════════════════════════════════════════════
# Synthetic returns only: no network, no prices, deterministic seeds.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pytest

from src.rl.portfolio_agent import TrainConfig, train_portfolio_agent
from src.rl.portfolio_env import EnvConfig, PortfolioEnv, portfolio_metrics, softmax


def _synthetic(T: int = 800, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.normal(0.001, 0.01, T), rng.normal(-0.001, 0.01, T), rng.normal(0.0, 0.02, T)],
        axis=1,
    )


class TestEnv:
    def test_softmax_sums_to_one(self):
        w = softmax(np.array([1.0, 2.0, 3.0]))
        assert w.sum() == pytest.approx(1.0)
        assert np.argmax(w) == 2

    def test_starts_in_cash(self):
        env = PortfolioEnv(_synthetic(100))
        env.reset()
        assert env.weights[-1] == 1.0
        assert env.n_actions == 4

    def test_all_cash_earns_nothing_and_costs_nothing(self):
        env = PortfolioEnv(_synthetic(100), EnvConfig(risk_aversion=0, relative_reward=False))
        env.reset()
        _, reward, info = env.step(np.array([0, 0, 0, 1.0]))
        assert info["gross"] == 0 and info["cost"] == 0 and reward == pytest.approx(0)

    def test_relative_reward_scores_against_an_equal_weight_basket(self):
        returns = np.tile([0.02, -0.01], (100, 1))
        env = PortfolioEnv(returns, EnvConfig(risk_aversion=0, cost_bps=0, relative_reward=True))
        env.reset()
        # Holding the basket exactly earns zero relative reward…
        env.weights = np.array([0.5, 0.5, 0.0])
        _, basket, _ = env.step(np.array([0.5, 0.5, 0.0]))
        assert basket == pytest.approx(0.0, abs=1e-12)
        # …and tilting toward the winner earns a positive one.
        _, tilted, _ = env.step(np.array([0.9, 0.1, 0.0]))
        assert tilted > 0

    def test_turnover_is_charged(self):
        env = PortfolioEnv(np.zeros((100, 2)), EnvConfig(cost_bps=10, risk_aversion=0, relative_reward=False))
        env.reset()
        _, _, info = env.step(np.array([1.0, 0, 0]))
        # Cash → asset 1 moves two units of weight (1 out, 1 in).
        assert info["turnover"] == pytest.approx(2.0)
        assert info["cost"] == pytest.approx(0.002)

    def test_risk_aversion_penalises_volatility(self):
        returns = np.full((100, 1), 0.05)
        calm = PortfolioEnv(returns, EnvConfig(risk_aversion=0, cost_bps=0, relative_reward=False))
        nervous = PortfolioEnv(returns, EnvConfig(risk_aversion=5, cost_bps=0, relative_reward=False))
        calm.reset(), nervous.reset()
        _, r_calm, _ = calm.step(np.array([1.0, 0]))
        _, r_nervous, _ = nervous.step(np.array([1.0, 0]))
        assert r_nervous < r_calm

    def test_too_short_history_is_refused(self):
        with pytest.raises(ValueError):
            PortfolioEnv(np.zeros((10, 2)))

    def test_metrics_on_a_flat_series(self):
        m = portfolio_metrics(np.zeros(252))
        assert m["total_return"] == 0 and m["max_drawdown"] == 0 and m["sharpe"] == 0

    def test_metrics_drawdown(self):
        m = portfolio_metrics(np.array([0.1, -0.5, 0.0]))
        assert m["max_drawdown"] == pytest.approx(0.5)


class TestAgent:
    def test_learns_to_prefer_the_rising_asset(self):
        r = _synthetic()
        res = train_portfolio_agent(r, [f"d{i}" for i in range(len(r))], ["UP", "DOWN", "NOISE"], TrainConfig())
        avg = res["average_weights"]
        assert avg["UP"] > 3 * avg["DOWN"]
        assert avg["UP"] > avg["NOISE"]
        assert res["metrics"]["policy"]["total_return"] > res["metrics"]["buy_hold"]["total_return"]

    def test_is_deterministic_for_a_seed(self):
        r = _synthetic(400)
        dates = [f"d{i}" for i in range(len(r))]
        a = train_portfolio_agent(r, dates, ["A", "B", "C"], TrainConfig(episodes=80))
        b = train_portfolio_agent(r, dates, ["A", "B", "C"], TrainConfig(episodes=80))
        assert a["target_weights"] == b["target_weights"]

    def test_result_shape(self):
        r = _synthetic(400)
        dates = [f"d{i}" for i in range(len(r))]
        calls: list[tuple[int, int]] = []
        res = train_portfolio_agent(
            r, dates, ["A", "B", "C"], TrainConfig(episodes=40), on_progress=lambda d, t, _: calls.append((d, t))
        )
        assert calls[-1] == (40, 40)
        assert set(res["target_weights"]) == {"A", "B", "C", "CASH"}
        assert sum(res["target_weights"].values()) == pytest.approx(1.0, abs=1e-3)
        assert len(res["backtest"]["dates"]) == len(res["backtest"]["policy"])
        # Out of sample: the backtest starts on the first bar after training.
        assert res["train_span"]["end"] == dates[279]
        assert res["test_span"]["start"] == dates[280]

    def test_shape_mismatch_is_refused(self):
        with pytest.raises(ValueError):
            train_portfolio_agent(np.zeros((400, 2)), ["d"] * 400, ["A"], TrainConfig(episodes=10))


# ═══════════════════════════════════════════════════════
# Budget planner, trade timing, alert prioritiser
# ═══════════════════════════════════════════════════════

from datetime import date  # noqa: E402

from src.rl.alert_bandit import Label, fit, preferences, rank  # noqa: E402
from src.rl.budget_agent import (  # noqa: E402
    RATES,
    BudgetConfig,
    _base_context,
    expected_rewards,
    profile_from_ledger,
    simulate_month,
    train_budget_agent,
)
from src.rl.timing_agent import TimingConfig, ticker_features, train_timing_agent  # noqa: E402

_CATS = [
    {"name": "Rent", "monthly_limit": 1200, "kind": "need"},
    {"name": "Food", "monthly_limit": 400, "kind": "need"},
    {"name": "Fun", "monthly_limit": 250, "kind": "want"},
    {"name": "Invest", "monthly_limit": 300, "kind": "savings"},
]


def _entries(income: float = 5000) -> list[dict]:
    out = []
    for m in ("2026-08", "2026-09", "2026-10"):
        out += [
            {"occurred_on": f"{m}-01", "amount": income, "direction": "income", "category": "Income"},
            {"occurred_on": f"{m}-02", "amount": 1200, "direction": "expense", "category": "Rent"},
            {"occurred_on": f"{m}-03", "amount": 400, "direction": "expense", "category": "Food"},
            {"occurred_on": f"{m}-04", "amount": 250, "direction": "expense", "category": "Fun"},
        ]
    return out


class TestBudgetProfile:
    def test_reads_income_and_needs_from_the_ledger(self):
        p = profile_from_ledger(_entries(), _CATS, [], today=date(2026, 11, 3))
        assert p.income_mu == 5000 and not p.assumed_income
        assert p.needs_mu == 1600 and p.wants_mu == 250
        assert p.months_of_history == 3

    def test_needs_never_drop_below_the_stated_limit(self):
        sparse = [{"occurred_on": "2026-10-03", "amount": 30, "direction": "expense", "category": "Food"}]
        p = profile_from_ledger(sparse, _CATS, [], today=date(2026, 11, 3))
        assert p.needs_by_category["Food"] == 400  # logged 30 once; the stated bill is 400

    def test_assumes_income_when_none_is_recorded(self):
        p = profile_from_ledger([], _CATS, [], today=date(2026, 11, 3))
        assert p.assumed_income and p.income_mu > 0

    def test_goal_urgency_from_deadlines(self):
        goals = [{"id": 1, "name": "Trip", "target": 1200, "saved": 0, "deadline": "2027-05-01"}]
        p = profile_from_ledger(_entries(), _CATS, goals, today=date(2026, 11, 3))
        assert p.goal_required == pytest.approx(200)  # 1200 over 6 months


class TestBudgetSimulator:
    def test_saving_beyond_what_income_allows_is_penalised(self):
        p = profile_from_ledger(_entries(income=1700), _CATS, [], today=date(2026, 11, 3))
        values = expected_rewards(_base_context(p), np.random.default_rng(0), samples=200)
        assert values[RATES.index(0.5)] < values[RATES.index(0.05)]

    def test_urgent_goals_push_the_optimum_up(self):
        calm = profile_from_ledger(
            _entries(),
            _CATS,
            [{"id": 1, "name": "Emergency fund", "target": 30000, "saved": 29000, "deadline": None}],
            today=date(2026, 11, 3),
        )
        urgent = profile_from_ledger(
            _entries(),
            _CATS,
            [{"id": 1, "name": "Emergency fund", "target": 9000, "saved": 0, "deadline": "2027-03-01"}],
            today=date(2026, 11, 3),
        )
        rng = np.random.default_rng(0)
        best_calm = RATES[int(np.argmax(expected_rewards(_base_context(calm), rng, 200)))]
        best_urgent = RATES[int(np.argmax(expected_rewards(_base_context(urgent), rng, 200)))]
        assert best_urgent > best_calm

    def test_shortfall_is_paid_out_of_savings(self):
        ctx = _base_context(profile_from_ledger(_entries(income=1600), _CATS, [], today=date(2026, 11, 3)))
        out = simulate_month(ctx, 0.5, np.random.default_rng(1))
        assert out["shortfall"] > 0 and out["saved"] < 0.5 * 1600


class TestBudgetAgent:
    def test_recommendation_is_near_the_simulated_best(self):
        goals = [{"id": 1, "name": "Emergency fund", "target": 6000, "saved": 1500, "deadline": "2027-08-01"}]
        p = profile_from_ledger(_entries(), _CATS, goals, today=date(2026, 11, 3))
        res = train_budget_agent(p, BudgetConfig(rounds=1500))
        values = res["simulated_value"]
        chosen = values[RATES.index(res["recommended_rate"])]
        # Within 2% of the best achievable value in the simulator.
        assert chosen >= max(values) - 0.02 * abs(max(values))
        assert len(res["learning_curve"]) == len(res["rule_curve"])

    def test_plan_is_internally_consistent(self):
        goals = [{"id": 7, "name": "Trip", "target": 1200, "saved": 0, "deadline": "2027-05-01"}]
        p = profile_from_ledger(_entries(), _CATS, goals, today=date(2026, 11, 3))
        plan = train_budget_agent(p, BudgetConfig(rounds=600))["plan"]
        assert set(plan["category_limits"]) == {"Rent", "Food", "Fun", "Invest"}
        goal = plan["goals"][0]
        assert goal["goal_id"] == 7 and goal["monthly"] > 0
        allocated = sum(g["monthly"] for g in plan["goals"]) + plan["unallocated_savings"]
        assert allocated == pytest.approx(plan["monthly_savings"], abs=0.05)


def _regimes(seed: int = 4, T: int = 1300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r: list[float] = []
    while len(r) < T:
        n = int(rng.integers(60, 120))
        mu = rng.choice([0.002, -0.004], p=[0.6, 0.4])
        r += list(rng.normal(mu, 0.01, n))
    base = np.array(r[:T])
    return base[:, None] + rng.normal(0, 0.004, (T, 3))


class TestTimingAgent:
    def test_features_never_look_ahead(self):
        r = np.random.default_rng(0).normal(0, 0.01, 300)
        f1, scale = ticker_features(r)
        r2 = r.copy()
        r2[200:] += 0.5  # change the future
        f2, _ = ticker_features(r2, scale)
        assert np.allclose(f1[:201], f2[:201])

    def test_learns_to_step_aside_in_trending_regimes(self):
        syn = _regimes()
        res = train_timing_agent(syn, [f"d{i}" for i in range(len(syn))], ["A", "B", "C"], TimingConfig(episodes=600))
        beats = [s["metrics"]["agent"]["sharpe"] > s["metrics"]["buy_hold"]["sharpe"] for s in res["signals"]]
        assert sum(beats) >= 2
        assert res["feature_weights"]["mom_60d"] > 0  # it learned momentum, in the right direction
        assert all(0 < s["time_in_market"] < 1 for s in res["signals"])

    def test_stays_long_when_there_is_nothing_to_time(self):
        drift = np.full((800, 2), 0.001) + np.random.default_rng(2).normal(0, 0.002, (800, 2))
        res = train_timing_agent(drift, [f"d{i}" for i in range(800)], ["X", "Y"], TimingConfig(episodes=200))
        assert all(s["time_in_market"] > 0.9 for s in res["signals"])

    def test_signal_shape(self):
        syn = _regimes(T=500)
        res = train_timing_agent(syn, [f"d{i}" for i in range(500)], ["A", "B", "C"], TimingConfig(episodes=40))
        s = res["signals"][0]
        assert s["signal"] in ("LONG", "FLAT") and 0.5 <= s["confidence"] <= 1
        assert len(s["equity"]["agent"]) == len(res["dates"]) == len(s["equity"]["position"])


def _alert(kind: str, ticker: str, sev: str = "HIGH") -> dict:
    return {"alert_type": kind, "ticker": ticker, "severity": sev}


class TestAlertBandit:
    def test_ranks_the_kind_people_approve_first(self):
        labels = [Label(_alert("PRICE_MOVE", "AAPL"), True)] * 5 + [Label(_alert("NEWS_SENTIMENT", "TSLA"), False)] * 5
        post = fit(labels)
        ranked = rank([_alert("NEWS_SENTIMENT", "TSLA", "MED"), _alert("PRICE_MOVE", "AAPL", "MED")], post)
        assert ranked[0]["alert_type"] == "PRICE_MOVE"
        assert ranked[0]["priority"]["priority"] > 0.5 > ranked[1]["priority"]["priority"]

    def test_unseen_values_borrow_from_seen_features(self):
        post = fit([Label(_alert("PRICE_MOVE", "AAPL"), True)] * 6)
        new_ticker = rank([_alert("PRICE_MOVE", "MSFT")], post)[0]["priority"]["priority"]
        assert new_ticker > 0.5

    def test_no_labels_means_a_neutral_prior(self):
        p = rank([_alert("MACRO_EVENT", "SPY")], fit([]))[0]["priority"]
        assert p["priority"] == pytest.approx(0.5)

    def test_thompson_sampling_explores(self):
        post = fit([Label(_alert("PRICE_MOVE", "AAPL"), True)])
        draws = {
            rank([_alert("PRICE_MOVE", "AAPL")], post, explore=True, rng=np.random.default_rng(i))[0]["priority"][
                "sampled"
            ]
            for i in range(10)
        }
        assert len(draws) > 1

    def test_weights_count(self):
        post = fit([Label(_alert("PRICE_MOVE", "AAPL"), True, weight=2.0)])
        prefs = preferences(post)
        assert prefs["alert_type"][0]["useful"] == 2.0
