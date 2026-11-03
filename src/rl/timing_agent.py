# ═══════════════════════════════════════════════════════
# FinSight — Trade timing agent (long / flat, backtest only)
# ═══════════════════════════════════════════════════════
#
# Purpose : Learn, per ticker, when to be invested and when to sit in cash,
#           from price-derived features. Produces SIGNALS — it never trades.
#
# Public API:
#   TimingConfig, ticker_features(returns), train_timing_agent(...)
#
# ══ THE MDP ══
#   state   standardised momentum (5 / 20 / 60 day), short/long volatility
#           ratio, distance from the 50-day average, and the current position.
#   action  long (1) or flat (0), sampled from a logistic policy.
#   reward  (position − 1) × next-day return − cost on every switch
#           − risk_aversion × (position × return)².
#           Measured RELATIVE TO BUY-AND-HOLD, for the same reason the
#           portfolio agent's reward is relative to equal weight: always-long
#           scores exactly zero, so the learner only gets credit for getting
#           OUT before falls and IN before rises, not for the market's drift.
#
# ══ ONE POLICY, MANY TICKERS ══
#   The weights are shared across the universe (features are standardised per
#   ticker), which multiplies the training data and stops a policy from
#   memorising one chart. The out-of-sample backtest is then run per ticker.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from src.rl.portfolio_agent import _Adam, _discounted, _downsample
from src.rl.portfolio_env import portfolio_metrics
from src.rl.ppo import Episode, PPOConfig, PPOLearner

ProgressFn = Callable[[int, int, float], None]

WARMUP = 60
N_FEATURES = 8


@dataclass(frozen=True)
class TimingConfig:
    algorithm: Literal["ppo", "reinforce"] = "ppo"
    ppo_actor_lr: float = 3e-3
    ppo_weight_decay: float = 8.0
    ppo_ent_coef: float = 0.01
    episodes: int = 600
    batch: int = 8
    episode_len: int = 60
    learning_rate: float = 0.05
    cost_bps: float = 10.0
    risk_aversion: float = 1.0
    test_fraction: float = 0.3
    l2: float = 0.05
    gamma: float = 0.9
    prior_bias: float = 0.5
    # Decide once every `hold` trading days and keep the position in between.
    # Exploring one day at a time pays the switching cost twice for one day of
    # move — the learner rightly concludes switching never pays. Real timing
    # decisions are about weeks, not days.
    hold: int = 5
    seed: int = 13


def ticker_features(returns: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """
    Per-bar features for ONE ticker, (T, N_FEATURES − 2) without position or bias.

    Row ``t`` uses returns strictly before ``t`` — no look-ahead. Rows before
    WARMUP are zero and never used.
    """
    r = np.asarray(returns, dtype=np.float64)
    T = r.size
    scale = scale or (float(np.std(r)) or 1e-3)
    price = np.concatenate([[1.0], np.cumprod(1.0 + r)])  # price[t] = level BEFORE return r[t]
    feats = np.zeros((T, N_FEATURES - 2))
    for t in range(WARMUP, T):
        past = r[t - 60 : t]
        vol20 = past[-20:].std() or 1e-6
        vol60 = past.std() or 1e-6
        ma50 = price[t - 50 : t].mean()
        feats[t] = [
            past[-5:].sum() / (scale * np.sqrt(5)),
            past[-20:].sum() / (scale * np.sqrt(20)),
            past.sum() / (scale * np.sqrt(60)),
            vol20 / vol60 - 1.0,
            (price[t] / ma50 - 1.0) / (scale * np.sqrt(20)),
            (vol20 / scale) - 1.0,
        ]
    return feats, scale


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def _step_reward(pos: int, prev: int, r: float, cfg: TimingConfig) -> float:
    cost = abs(pos - prev) * cfg.cost_bps / 10_000.0
    return (pos - 1) * r - cost - cfg.risk_aversion * (pos * r) ** 2


def _phi(feats: np.ndarray, t: int, pos: int) -> np.ndarray:
    return np.concatenate([feats[t], [float(pos), 1.0]])


def _rollout(
    theta: np.ndarray,
    feats: np.ndarray,
    returns: np.ndarray,
    start: int,
    cfg: TimingConfig,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """One stochastic training episode on one ticker: (phis, d log π, discounted returns), total reward."""
    pos = 1
    phis, grads_scale, rewards = [], [], []
    for t in range(start, start + cfg.episode_len, cfg.hold):
        phi = _phi(feats, t, pos)
        p = _sigmoid(float(theta @ phi))
        action = int(rng.random() < p)
        reward = 0.0
        for k in range(t, min(t + cfg.hold, returns.size)):
            reward += _step_reward(action, pos if k == t else action, float(returns[k]), cfg)
        rewards.append(reward)
        phis.append(phi)
        grads_scale.append(action - p)  # d log π / d logit
        pos = action
    return (np.array(phis), np.array(grads_scale), _discounted(rewards, cfg.gamma)), float(np.sum(rewards))


ProbFn = Callable[[np.ndarray], float]


def _backtest(
    prob: ProbFn,
    feats: np.ndarray,
    returns: np.ndarray,
    scale: float,
    split: int,
    ticker: str,
    cfg: TimingConfig,
) -> dict[str, Any]:
    """Deterministic out-of-sample run (long iff p > 0.5) plus today's signal."""
    T = returns.size
    pos = 1
    strat, held = [], []
    for t in range(split, T):
        if (t - split) % cfg.hold == 0:
            action = int(prob(_phi(feats, t, pos)) > 0.5)
        else:
            action = pos
        cost = abs(action - pos) * cfg.cost_bps / 10_000.0
        strat.append(action * float(returns[t]) - cost)
        held.append(action)
        pos = action
    bh = returns[split:]
    # Today's signal: features over the most recent bars, current position.
    last_feats, _ = ticker_features(np.append(returns, 0.0), scale)
    p_now = prob(_phi(last_feats, T, pos))
    return {
        "ticker": ticker,
        "signal": "LONG" if p_now > 0.5 else "FLAT",
        "confidence": round(p_now if p_now > 0.5 else 1 - p_now, 4),
        "prob_long": round(p_now, 4),
        "time_in_market": round(float(np.mean(held)), 4),
        "switches": int(np.sum(np.abs(np.diff([1, *held])))),
        "metrics": {
            "agent": dict(portfolio_metrics(np.array(strat))),
            "buy_hold": dict(portfolio_metrics(bh)),
        },
        "equity": {
            "agent": _downsample([round(float(x), 6) for x in np.cumprod(1.0 + np.array(strat))]),
            "buy_hold": _downsample([round(float(x), 6) for x in np.cumprod(1.0 + bh)]),
            # Long/flat per (downsampled) bar, for the shaded "in market" band.
            "position": _downsample(held),
        },
    }


FEATURE_NAMES = ["mom_5d", "mom_20d", "mom_60d", "vol_ratio", "vs_ma50", "vol_level", "position", "bias"]
Trained = tuple[ProbFn, list[float], dict[str, float]]


def _train_reinforce(
    feats: list[np.ndarray],
    returns: np.ndarray,
    split: int,
    cfg: TimingConfig,
    rng: np.random.Generator,
    on_progress: ProgressFn | None,
) -> Trained:
    """The logistic REINFORCE baseline."""
    N = returns.shape[1]
    # The prior leans "stay long" (prior_bias 0.5 → p ≈ 0.62). The L2 term below pulls toward THIS,
    # not toward zero — zero would be a coin flip, and a coin flip in a rising
    # market is an expensive default. The policy has to earn every exit.
    prior = np.zeros(N_FEATURES)
    prior[-1] = cfg.prior_bias
    theta = prior.copy()
    adam = _Adam(theta.shape, cfg.learning_rate)
    curve: list[float] = []
    done = 0
    updates = max(1, int(np.ceil(cfg.episodes / cfg.batch)))

    for _ in range(updates):
        n_this = min(cfg.batch, cfg.episodes - done)
        episodes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        totals = []
        for _ in range(n_this):
            j = int(rng.integers(N))
            start = int(rng.integers(WARMUP, split - cfg.episode_len))
            episode, total = _rollout(theta, feats[j], returns[:, j], start, cfg, rng)
            episodes.append(episode)
            totals.append(total)

        all_rtg = np.concatenate([e[2] for e in episodes])
        mean, std = all_rtg.mean(), all_rtg.std() or 1.0
        grad = np.zeros_like(theta)
        for phis_arr, g, rtg in episodes:
            grad += ((g * (rtg - mean) / std)[:, None] * phis_arr).sum(axis=0)
        grad = grad / len(all_rtg) - cfg.l2 * (theta - prior)
        theta += adam.step(grad)

        done += n_this
        curve.append(round(float(np.mean(totals)), 6))
        if on_progress:
            on_progress(done, cfg.episodes, curve[-1])

    return (
        (lambda phi: _sigmoid(float(theta @ phi))),
        curve,
        {n: round(float(w), 4) for n, w in zip(FEATURE_NAMES, theta)},
    )


def _train_ppo(
    feats: list[np.ndarray],
    returns: np.ndarray,
    split: int,
    cfg: TimingConfig,
    rng: np.random.Generator,
    on_progress: ProgressFn | None,
) -> Trained:
    """
    PPO with an MLP actor (Bernoulli long/flat) and critic. One decision per
    `hold` days; the reward for a decision is the sum over the days it holds.
    The actor starts at the same long-leaning prior as the baseline, via a
    fixed logit offset, and weight decay pulls the network back toward it.
    """
    N = returns.shape[1]
    learner = PPOLearner(
        "bernoulli",
        N_FEATURES,
        1,
        PPOConfig(
            gamma=cfg.gamma,
            actor_lr=cfg.ppo_actor_lr,
            weight_decay=cfg.ppo_weight_decay,
            ent_coef=cfg.ppo_ent_coef,
        ),
        rng,
    )
    learner.bias = cfg.prior_bias
    curve: list[float] = []
    done = 0
    states: list[np.ndarray] = []
    while done < cfg.episodes:
        n_this = min(cfg.batch, cfg.episodes - done)
        episodes: list[Episode] = []
        for _ in range(n_this):
            j = int(rng.integers(N))
            start = int(rng.integers(WARMUP, split - cfg.episode_len))
            ep, pos = Episode(), 1
            for t in range(start, start + cfg.episode_len, cfg.hold):
                phi = _phi(feats[j], t, pos)
                action, logp = learner.act(phi, rng)
                a = int(action[0])
                reward = sum(
                    _step_reward(a, pos if k == t else a, float(returns[k, j]), cfg)
                    for k in range(t, min(t + cfg.hold, returns.shape[0]))
                )
                ep.add(phi, action, logp, reward)
                pos = a
            ep.last_obs = _phi(feats[j], min(start + cfg.episode_len, returns.shape[0] - 1), pos)
            episodes.append(ep)
            states.extend(ep.obs)
        learner.learn(episodes, rng)
        done += n_this
        curve.append(round(float(np.mean([sum(ep.rewards) for ep in episodes])), 6))
        if on_progress:
            on_progress(done, cfg.episodes, curve[-1])

    # Sensitivity of the logit to each feature, averaged over visited states.
    sample = np.array(states[:: max(1, len(states) // 400)])
    explain: dict[str, float] = {}
    for k, name in enumerate(FEATURE_NAMES[:-1]):
        hi, lo = sample.copy(), sample.copy()
        hi[:, k] += 0.5
        lo[:, k] -= 0.5
        explain[name] = round(float(np.mean(learner.actor.forward(hi)[:, 0] - learner.actor.forward(lo)[:, 0])), 4)
    explain["bias"] = round(float(np.mean(learner.actor.forward(sample)[:, 0])) + learner.bias, 4)
    return learner.prob, curve, explain


def train_timing_agent(
    returns: np.ndarray,
    dates: list[str],
    tickers: list[str],
    config: TimingConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Train a shared long/flat policy on the head of each series; backtest each on the tail."""
    cfg = config or TimingConfig()
    T, N = returns.shape
    if N != len(tickers) or T != len(dates):
        raise ValueError("returns, dates and tickers disagree on shape")
    split = int(T * (1 - cfg.test_fraction))
    if split - WARMUP < cfg.episode_len + 5 or T - split < 20:
        raise ValueError(f"not enough history to train and backtest ({T} bars)")

    # Feature scale per ticker comes from the TRAINING span only.
    feats, scales = [], []
    for j in range(N):
        _, scale = ticker_features(returns[:split, j])
        f, _ = ticker_features(returns[:, j], scale)
        feats.append(f)
        scales.append(scale)

    rng = np.random.default_rng(cfg.seed)
    if cfg.algorithm == "ppo":
        prob, curve, explain = _train_ppo(feats, returns, split, cfg, rng, on_progress)
    else:
        prob, curve, explain = _train_reinforce(feats, returns, split, cfg, rng, on_progress)

    # ── out-of-sample backtest, deterministic (long iff p > 0.5) ──
    per_ticker = []
    test_dates = dates[split:]
    for j, ticker in enumerate(tickers):
        per_ticker.append(_backtest(prob, feats[j], returns[:, j], scales[j], split, ticker, cfg))

    return {
        "algorithm": cfg.algorithm,
        "tickers": tickers,
        "signals": per_ticker,
        "dates": _downsample(test_dates),
        "learning_curve": _downsample(curve),
        # Linear baseline: the weights themselves. PPO: the average sensitivity
        # of the policy's logit to each feature over the training states — the
        # nearest thing a network has to "what it weighs".
        "feature_weights": explain,
        "train_span": {"start": dates[0], "end": dates[split - 1], "bars": split},
        "test_span": {"start": test_dates[0], "end": test_dates[-1], "bars": len(test_dates)},
        "config": {
            "algorithm": cfg.algorithm,
            "episodes": cfg.episodes,
            "risk_aversion": cfg.risk_aversion,
            "cost_bps": cfg.cost_bps,
        },
    }
