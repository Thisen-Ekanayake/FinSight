# ═══════════════════════════════════════════════════════
# FinSight — Portfolio allocation environment
# ═══════════════════════════════════════════════════════
#
# Purpose : A small, deterministic market-replay environment for learning a
#           portfolio allocation policy over N assets plus cash.
#
# Public API:
#   EnvConfig, PortfolioEnv, softmax, portfolio_metrics
#
# ══ THE MDP ══
#   state   per-asset features over a trailing window (mean return, volatility,
#           short momentum — all standardised) plus the CURRENT weights, so the
#           policy can learn that moving costs money.
#   action  a target weight vector over [asset_1 … asset_N, cash], summing to 1.
#   reward  log growth of the portfolio over the next bar, net of the
#           transaction cost its turnover caused,
#             − log growth of an equal-weight basket (relative_reward)
#             − risk_aversion × (bar return)²
#           The squared term is a per-step variance penalty: with
#           risk_aversion = 0 the agent maximises growth alone, and raising it
#           trades return for a smoother equity curve. It is the "risk" slider
#           in the UI.
#
# ══ WHY REPLAY AND NOT SIMULATION ══
#   The environment walks real historical returns. It cannot model the agent's
#   own market impact, and an agent that does well here has learned something
#   about the PAST. Every result shown to a user is an out-of-sample backtest
#   on a held-out tail it never trained on — and is still not a forecast.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class EnvConfig:
    """Knobs for one environment instance."""

    window: int = 20
    episode_len: int = 60
    cost_bps: float = 10.0
    risk_aversion: float = 1.0
    # Score growth RELATIVE to holding the assets equally. Market-wide moves
    # then cancel out of the reward, so the learner sees the signal it can
    # actually act on (which asset, how much cash) instead of mostly noise.
    relative_reward: bool = True


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    out: np.ndarray = exp / np.sum(exp, axis=-1, keepdims=True)
    return out


class PortfolioEnv:
    """
    Replay environment over a (T, N) matrix of simple daily returns.

    Cash is the implicit (N+1)th asset and earns nothing — conservative, and it
    keeps "go to cash" an honest, cost-free-to-hold choice.
    """

    def __init__(self, returns: np.ndarray, config: EnvConfig | None = None, *, scale: float | None = None) -> None:
        if returns.ndim != 2:
            raise ValueError("returns must be a (T, N) matrix")
        self.config = config or EnvConfig()
        if returns.shape[0] <= self.config.window + 1:
            raise ValueError(f"need more than {self.config.window + 1} bars, got {returns.shape[0]}")

        self.returns = returns.astype(np.float64)
        self.n_assets = returns.shape[1]
        self.n_actions = self.n_assets + 1

        # One scale for every feature, so features are O(1) whatever the asset's
        # volatility. A backtest env passes the TRAINING env's scale in: deriving
        # it from the test span would leak the future into the features.
        self.scale = scale if scale is not None else (float(np.std(self.returns)) or 1e-3)

        self.t = self.config.window
        self.end = returns.shape[0] - 1
        self.weights = self._initial_weights()

    # ── geometry ──────────────────────────────────────

    @property
    def n_features(self) -> int:
        return int(3 * self.n_assets + self.n_actions + 1)

    def _initial_weights(self) -> np.ndarray:
        w = np.zeros(self.n_actions)
        w[-1] = 1.0  # start in cash
        return w

    def features(self, t: int | None = None, weights: np.ndarray | None = None) -> np.ndarray:
        """Observation at bar ``t`` (defaults to the current bar)."""
        t = self.t if t is None else t
        weights = self.weights if weights is None else weights
        window = self.returns[t - self.config.window : t]
        mean = window.mean(axis=0) / self.scale
        vol = window.std(axis=0) / self.scale
        momentum = window[-5:].sum(axis=0) / (self.scale * np.sqrt(5))
        return np.concatenate([mean, vol, momentum, weights, [1.0]])

    # ── episode ───────────────────────────────────────

    def reset(self, start: int | None = None) -> np.ndarray:
        """Start an episode at bar ``start`` (defaults to the first usable bar)."""
        lo = self.config.window
        self.t = lo if start is None else max(lo, min(start, self.end - 1))
        self.weights = self._initial_weights()
        return self.features()

    def done(self) -> bool:
        return bool(self.t >= self.end)

    def step(self, target: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        """
        Rebalance to ``target`` at the close of bar ``t`` and hold through bar ``t``.

        Returns
        -------
        tuple
            ``(next_observation, reward, info)`` where ``info`` carries the
            gross portfolio return and the cost actually paid.
        """
        target = np.clip(np.asarray(target, dtype=np.float64), 0.0, None)
        total = target.sum()
        target = target / total if total > 0 else self._initial_weights()

        turnover = float(np.abs(target - self.weights).sum())
        cost = turnover * self.config.cost_bps / 10_000.0

        asset_returns = np.append(self.returns[self.t], 0.0)
        gross = float(target @ asset_returns)
        net = gross - cost
        growth = np.log(max(1.0 + net, 1e-6))
        if self.config.relative_reward:
            growth -= np.log(max(1.0 + float(asset_returns[:-1].mean()), 1e-6))
        reward = growth - self.config.risk_aversion * gross**2

        # Weights drift with the market until the next rebalance.
        grown = target * (1.0 + asset_returns)
        self.weights = grown / grown.sum() if grown.sum() > 0 else self._initial_weights()
        self.t += 1

        return self.features(), float(reward), {"gross": gross, "net": net, "cost": cost, "turnover": turnover}


class Metrics(TypedDict):
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float


def portfolio_metrics(daily_returns: np.ndarray) -> Metrics:
    """Standard summary statistics for a series of daily net returns."""
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size == 0:
        return Metrics(total_return=0.0, annual_return=0.0, annual_volatility=0.0, sharpe=0.0, max_drawdown=0.0)
    equity = np.cumprod(1.0 + r)
    total = float(equity[-1] - 1.0)
    years = r.size / TRADING_DAYS
    annual = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    drawdown = float(np.max(1.0 - equity / peak))
    return Metrics(
        total_return=total,
        annual_return=annual,
        annual_volatility=vol,
        sharpe=sharpe,
        max_drawdown=drawdown,
    )
