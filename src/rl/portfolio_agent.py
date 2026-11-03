# ═══════════════════════════════════════════════════════
# FinSight — Portfolio allocation agent (policy gradient)
# ═══════════════════════════════════════════════════════
#
# Purpose : Train an allocation policy on PortfolioEnv and backtest it on a
#           held-out tail against two plain benchmarks.
#
# Public API:
#   TrainConfig, train_portfolio_agent(returns, dates, tickers, config, on_progress)
#
# ══ THE LEARNER ══
#   A linear-softmax policy trained with REINFORCE: logits = W·φ(s), Gaussian
#   exploration noise on the logits, softmax to a weight vector. The gradient
#   of log π w.r.t. the logits is ε/σ, so one update is an outer product per
#   step, scaled by a batch-normalised advantage, and fed to Adam.
#
#   Deliberately numpy-only. A deep policy (PPO + torch) is a later upgrade
#   behind the same train → backtest → target-weights contract; it would add
#   ~1 GB to the image to learn from a few hundred bars of daily data, where a
#   linear policy over good features is the honest model size.
#
# ══ WHAT THE NUMBERS MEAN ══
#   The returned backtest covers ONLY bars the policy never trained on. It is
#   still past data: a strong backtest is evidence the features carry signal,
#   not a promise about next month.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.rl.portfolio_env import EnvConfig, PortfolioEnv, portfolio_metrics, softmax

ProgressFn = Callable[[int, int, float], None]

MAX_SERIES_POINTS = 250


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 400
    batch: int = 8
    learning_rate: float = 0.05
    gamma: float = 0.9
    sigma_start: float = 1.0
    sigma_end: float = 0.1
    test_fraction: float = 0.3
    risk_aversion: float = 1.0
    cost_bps: float = 10.0
    window: int = 20
    episode_len: int = 60
    seed: int = 7


class _Adam:
    def __init__(self, shape: tuple[int, ...], lr: float) -> None:
        self.lr = lr
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Return the ascent step for ``grad``."""
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)
        return np.asarray(self.lr * m_hat / (np.sqrt(v_hat) + 1e-8))


def _downsample(values: list[Any], limit: int = MAX_SERIES_POINTS) -> list[Any]:
    if len(values) <= limit:
        return values
    idx = np.linspace(0, len(values) - 1, limit).round().astype(int)
    return [values[i] for i in idx]


def _discounted(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards))
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def _run_policy(env: PortfolioEnv, theta: np.ndarray) -> tuple[list[float], list[np.ndarray]]:
    """Deterministic rollout from the env's first bar to its last."""
    obs = env.reset()
    net: list[float] = []
    held: list[np.ndarray] = []
    while not env.done():
        weights = softmax(theta @ obs)
        held.append(weights)
        obs, _, info = env.step(weights)
        net.append(info["net"])
    return net, held


def train_portfolio_agent(
    returns: np.ndarray,
    dates: list[str],
    tickers: list[str],
    config: TrainConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Train on the head of ``returns``, backtest on the tail, and propose weights for today.

    Parameters
    ----------
    returns : ndarray
        ``(T, N)`` simple daily returns, oldest first. ``dates[i]`` is the date
        of ``returns[i]``.
    on_progress : callable, optional
        ``(episodes_done, episodes_total, mean_episode_reward)`` after each batch.

    Returns
    -------
    dict
        JSON-serialisable result: target weights, learning curve, backtest
        equity curves and metrics for the policy and both benchmarks.
    """
    cfg = config or TrainConfig()
    if returns.shape[1] != len(tickers) or returns.shape[0] != len(dates):
        raise ValueError("returns, dates and tickers disagree on shape")

    T = returns.shape[0]
    split = int(T * (1 - cfg.test_fraction))
    if split - cfg.window < cfg.episode_len + 5 or T - split < 20:
        raise ValueError(f"not enough history to train and backtest ({T} bars)")

    env_cfg = EnvConfig(
        window=cfg.window, episode_len=cfg.episode_len, cost_bps=cfg.cost_bps, risk_aversion=cfg.risk_aversion
    )
    train_env = PortfolioEnv(returns[:split], env_cfg)
    rng = np.random.default_rng(cfg.seed)

    theta = np.zeros((train_env.n_actions, train_env.n_features))
    adam = _Adam(theta.shape, cfg.learning_rate)

    curve: list[float] = []
    max_start = train_env.end - cfg.episode_len
    updates = max(1, int(np.ceil(cfg.episodes / cfg.batch)))
    done_episodes = 0

    for update in range(updates):
        frac = update / max(1, updates - 1)
        sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * frac

        grads: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (eps, phi, return-to-go) per episode
        episode_rewards: list[float] = []
        n_this = min(cfg.batch, cfg.episodes - done_episodes)
        for _ in range(n_this):
            obs = train_env.reset(int(rng.integers(cfg.window, max_start + 1)))
            eps_list, phi_list, rewards = [], [], []
            for _ in range(cfg.episode_len):
                if train_env.done():
                    break
                eps = rng.standard_normal(train_env.n_actions)
                weights = softmax(theta @ obs + sigma * eps)
                phi_list.append(obs)
                eps_list.append(eps)
                obs, reward, _ = train_env.step(weights)
                rewards.append(reward)
            grads.append((np.array(eps_list), np.array(phi_list), _discounted(rewards, cfg.gamma)))
            episode_rewards.append(float(np.sum(rewards)))

        all_returns = np.concatenate([g[2] for g in grads])
        mean, std = all_returns.mean(), all_returns.std() or 1.0
        grad = np.zeros_like(theta)
        for eps_arr, phi_arr, rtg in grads:
            adv = (rtg - mean) / std
            grad += ((eps_arr / sigma) * adv[:, None]).T @ phi_arr
        grad /= len(all_returns)
        theta += adam.step(grad)

        done_episodes += n_this
        curve.append(float(np.mean(episode_rewards)))
        if on_progress:
            on_progress(done_episodes, cfg.episodes, curve[-1])

    # ── out-of-sample backtest ──────────────────────
    test_returns = returns[split - cfg.window :]
    test_env = PortfolioEnv(test_returns, env_cfg, scale=train_env.scale)
    policy_net, held = _run_policy(test_env, theta)
    test_dates = dates[split : split + len(policy_net)]
    tail = returns[split : split + len(policy_net)]

    n = len(tickers)
    equal_weight = tail.mean(axis=1)  # daily-rebalanced 1/N, before costs
    buy_hold_equity = np.cumprod(1.0 + tail, axis=0).mean(axis=1)
    buy_hold = np.diff(np.concatenate([[1.0], buy_hold_equity])) / np.concatenate([[1.0], buy_hold_equity[:-1]])

    def equity(series: np.ndarray) -> list[float]:
        return [round(float(x), 6) for x in np.cumprod(1.0 + np.asarray(series))]

    # ── target for today ───────────────────────────
    # Features over the most recent window, with the weights the policy would
    # be holding after walking the test span.
    full_env = PortfolioEnv(returns, env_cfg, scale=train_env.scale)
    obs_now = full_env.features(t=T, weights=test_env.weights)
    target = softmax(theta @ obs_now)

    avg_held = np.mean(held, axis=0) if held else target
    labels = [*tickers, "CASH"]

    return {
        "tickers": tickers,
        "target_weights": {label: round(float(w), 4) for label, w in zip(labels, target)},
        "average_weights": {label: round(float(w), 4) for label, w in zip(labels, avg_held)},
        "learning_curve": [round(x, 6) for x in _downsample(curve)],
        "train_span": {"start": dates[0], "end": dates[split - 1], "bars": split},
        "test_span": {"start": test_dates[0], "end": test_dates[-1], "bars": len(test_dates)},
        "backtest": {
            "dates": _downsample(test_dates),
            "policy": _downsample(equity(np.array(policy_net))),
            "equal_weight": _downsample(equity(equal_weight)),
            "buy_hold": _downsample(equity(buy_hold)),
        },
        "metrics": {
            "policy": dict(portfolio_metrics(np.array(policy_net))),
            "equal_weight": dict(portfolio_metrics(equal_weight)),
            "buy_hold": dict(portfolio_metrics(buy_hold)),
        },
        "config": {
            "episodes": cfg.episodes,
            "risk_aversion": cfg.risk_aversion,
            "cost_bps": cfg.cost_bps,
            "window": cfg.window,
            "assets": n,
        },
    }
