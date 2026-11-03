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
