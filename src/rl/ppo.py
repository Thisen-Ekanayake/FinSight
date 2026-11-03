# ═══════════════════════════════════════════════════════
# FinSight — Proximal Policy Optimisation (PPO)
# ═══════════════════════════════════════════════════════
#
# Purpose : An actor-critic learner shared by the portfolio and timing agents.
#
# Public API:
#   PPOConfig, Episode, PPOLearner(kind="gaussian"|"bernoulli", obs_dim, act_dim, config, rng)
#     .act(obs, rng)            -> (action, logp)      stochastic, for rollouts
#     .act_deterministic(obs)   -> action               mean / p > 0.5, for backtests
#     .learn(episodes)          -> stats dict
#
# ══ THE ALGORITHM (Schulman et al., 2017) ══
#   actor   an MLP mapping a state to a distribution over actions:
#             gaussian   mean of a diagonal Gaussian (state-independent
#                        log-std); the portfolio agent softmaxes the sample
#                        into weights.
#             bernoulli  a logit; the timing agent's long / flat.
#   critic  a second MLP estimating the state's value — the baseline that
#           replaces REINFORCE's batch-mean and cuts gradient variance.
#   GAE     advantages A_t = Σ (γλ)^k δ_{t+k}, δ_t = r_t + γV(s_{t+1}) − V(s_t);
#           the last state of each truncated episode is bootstrapped from V.
#   update  several epochs of minibatch Adam on the CLIPPED surrogate
#             L = −E[min(ρA, clip(ρ, 1±ε)A)] − c_ent·H,   ρ = π_new/π_old
#           which lets each batch be reused without one lucky step moving the
#           policy too far. Epochs stop early if the KL to the old policy
#           drifts past `target_kl`.
#   scale   rewards are divided by a running std of the discounted return, so
#           the critic learns O(1) targets instead of ±0.001 daily returns.
#
# Every gradient here is hand-derived; tests/test_ppo.py checks them all
# against finite differences.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.rl.nn import MLP, Adam, clip_grads

LOG_2PI = math.log(2 * math.pi)


@dataclass(frozen=True)
class PPOConfig:
    hidden: tuple[int, ...] = (64, 64)
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.95
    lam: float = 0.9
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 256
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    init_log_std: float = 0.0
    # False: the caller sets `log_std` on a schedule (annealed exploration)
    # instead of learning it — see the portfolio agent for why.
    learn_std: bool = True
    # Decoupled (AdamW-style) decay on the actor's weights. With a near-zero
    # output layer the untrained policy is "no preference"; decay keeps it
    # there unless the data keeps pushing.
    weight_decay: float = 0.0


@dataclass
class Episode:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    last_obs: np.ndarray | None = None  # state after the final step, for bootstrapping

    def add(self, obs: np.ndarray, action: np.ndarray, logp: float, reward: float) -> None:
        self.obs.append(obs)
        self.actions.append(np.atleast_1d(action).astype(np.float64))
        self.logp.append(float(logp))
        self.rewards.append(float(reward))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


# ── Distribution maths (each returns log-probs and the pieces of its gradient) ──


def gaussian_logp(a: np.ndarray, mu: np.ndarray, log_std: np.ndarray) -> np.ndarray:
    z = (a - mu) / np.exp(log_std)
    return np.asarray(np.sum(-0.5 * z * z - log_std - 0.5 * LOG_2PI, axis=-1))


def bernoulli_logp(a: np.ndarray, logit: np.ndarray) -> np.ndarray:
    # log σ(z) = −softplus(−z); log(1 − σ(z)) = −softplus(z). Stable for any z.
    z = logit[..., 0]
    act = a[..., 0]
    return np.asarray(-(act * np.logaddexp(0, -z) + (1 - act) * np.logaddexp(0, z)))


def surrogate_grad_logp(logp: np.ndarray, old_logp: np.ndarray, adv: np.ndarray, clip: float) -> np.ndarray:
    """
    d(L_clip)/d(logp) per sample, with L_clip = −mean(min(ρA, clip(ρ)A)).

    The gradient flows only where the unclipped term is the minimum — i.e.
    the policy has not yet moved past the trust region in the direction the
    advantage favours.
    """
    ratio = np.exp(logp - old_logp)
    active = np.where(adv >= 0, ratio <= 1 + clip, ratio >= 1 - clip)
    return np.asarray(-(ratio * adv * active) / len(adv))


class RunningStd:
    """Running variance of a scalar stream (Welford, batched)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, xs: np.ndarray) -> None:
        for x in np.ravel(xs):
            self.n += 1
            d = x - self.mean
            self.mean += d / self.n
            self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 1.0


class PPOLearner:
    def __init__(
        self,
        kind: Literal["gaussian", "bernoulli"],
        obs_dim: int,
        act_dim: int,
        config: PPOConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.kind = kind
        self.cfg = config or PPOConfig()
        rng = rng or np.random.default_rng(0)
        out = act_dim if kind == "gaussian" else 1
        self.actor = MLP([obs_dim, *self.cfg.hidden, out], rng, out_scale=0.01)
        self.critic = MLP([obs_dim, *self.cfg.hidden, 1], rng, out_scale=1.0)
        self.log_std = np.full(out, self.cfg.init_log_std) if kind == "gaussian" else np.zeros(0)
        self.actor_opt = Adam(self._actor_params(), self.cfg.actor_lr)
        self.critic_opt = Adam(self.critic.params, self.cfg.critic_lr)
        self.ret_std = RunningStd()
        self.bias = 0.0  # initial logit offset for bernoulli policies (a prior lean)

    def _actor_params(self) -> list[np.ndarray]:
        if self.kind == "gaussian" and self.cfg.learn_std:
            return [*self.actor.params, self.log_std]
        return self.actor.params

    # ── acting ────────────────────────────────────────

    def act(self, obs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        out = self.actor.forward(obs[None])[0]
        if self.kind == "gaussian":
            a = out + np.exp(self.log_std) * rng.standard_normal(out.shape)
            return a, float(gaussian_logp(a[None], out[None], self.log_std)[0])
        logit = out + self.bias
        a = np.array([float(rng.random() < _sigmoid(logit)[0])])
        return a, float(bernoulli_logp(a[None], logit[None])[0])

    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        out = self.actor.forward(obs[None])[0]
        if self.kind == "gaussian":
            return np.asarray(out)
        return np.array([float(_sigmoid(out + self.bias)[0] > 0.5)])

    def prob(self, obs: np.ndarray) -> float:
        """P(action = 1) for a bernoulli policy."""
        return float(_sigmoid(self.actor.forward(obs[None])[0] + self.bias)[0])

    def value(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self.critic.forward(np.atleast_2d(obs))[:, 0])

    # ── learning ──────────────────────────────────────

    def _advantages(self, episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        # Reward scaling by the running std of discounted returns.
        for ep in episodes:
            running, rets = 0.0, []
            for reward in ep.rewards:
                running = running * cfg.gamma + reward
                rets.append(running)
            self.ret_std.update(np.array(rets))
        scale = self.ret_std.std or 1.0

        advs, targets = [], []
        for ep in episodes:
            scaled = np.array(ep.rewards) / scale
            v = self.value(np.array(ep.obs))
            v_next = np.append(v[1:], self.value(ep.last_obs)[0] if ep.last_obs is not None else 0.0)
            delta = scaled + cfg.gamma * v_next - v
            adv = np.zeros_like(delta)
            running = 0.0
            for t in range(len(delta) - 1, -1, -1):
                running = delta[t] + cfg.gamma * cfg.lam * running
                adv[t] = running
            advs.append(adv)
            targets.append(adv + v)
        return np.concatenate(advs), np.concatenate(targets)

    def actor_loss_and_grads(
        self, obs: np.ndarray, actions: np.ndarray, old_logp: np.ndarray, adv: np.ndarray
    ) -> tuple[float, list[np.ndarray], np.ndarray]:
        """Clipped-surrogate loss (with entropy bonus), its gradients, and the new log-probs."""
        cfg = self.cfg
        out = self.actor.forward(obs)
        n = len(adv)
        if self.kind == "gaussian":
            std = np.exp(self.log_std)
            logp = gaussian_logp(actions, out, self.log_std)
            g_logp = surrogate_grad_logp(logp, old_logp, adv, cfg.clip)
            diff = actions - out
            g_out = g_logp[:, None] * diff / std**2
            g_log_std = (g_logp[:, None] * (diff**2 / std**2 - 1.0)).sum(axis=0) - cfg.ent_coef
            entropy = float(np.sum(self.log_std + 0.5 * (1 + LOG_2PI)))
            grads = self.actor.backward(g_out)
            if cfg.learn_std:
                grads.append(g_log_std)
        else:
            logit = out + self.bias
            p = _sigmoid(logit[:, 0])
            logp = bernoulli_logp(actions, logit)
            g_logp = surrogate_grad_logp(logp, old_logp, adv, cfg.clip)
            z = logit[:, 0]
            # dH/dz = −z·p(1−p); the loss carries −c_ent·mean(H).
            g_z = g_logp * (actions[:, 0] - p) + cfg.ent_coef * z * p * (1 - p) / n
            entropy = float(np.mean(-(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))))
            grads = self.actor.backward(g_z[:, None])
        ratio = np.exp(logp - old_logp)
        surr = -float(np.mean(np.minimum(ratio * adv, np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv)))
        return surr - cfg.ent_coef * entropy, grads, logp

    def critic_loss_and_grads(self, obs: np.ndarray, targets: np.ndarray) -> tuple[float, list[np.ndarray]]:
        v = self.critic.forward(obs)[:, 0]
        err = v - targets
        return float(0.5 * np.mean(err**2)), self.critic.backward((err / len(err))[:, None])

    def learn(self, episodes: list[Episode], rng: np.random.Generator | None = None) -> dict[str, Any]:
        cfg = self.cfg
        rng = rng or np.random.default_rng(0)
        obs = np.array([o for ep in episodes for o in ep.obs])
        actions = np.array([a for ep in episodes for a in ep.actions])
        old_logp = np.array([lp for ep in episodes for lp in ep.logp])
        adv, targets = self._advantages(episodes)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        mb = min(cfg.minibatch, n)
        epochs_run, kl = 0, 0.0
        actor_loss = critic_loss = 0.0
        for _ in range(cfg.epochs):
            order = rng.permutation(n)
            for start in range(0, n, mb):
                idx = order[start : start + mb]
                actor_loss, a_grads, _ = self.actor_loss_and_grads(obs[idx], actions[idx], old_logp[idx], adv[idx])
                self.actor_opt.step(self._actor_params(), clip_grads(a_grads, cfg.max_grad_norm))
                if cfg.weight_decay:
                    for w in self.actor.W:
                        w *= 1.0 - cfg.actor_lr * cfg.weight_decay
                critic_loss, c_grads = self.critic_loss_and_grads(obs[idx], targets[idx])
                self.critic_opt.step(self.critic.params, clip_grads(c_grads, cfg.max_grad_norm))
            epochs_run += 1
            _, _, new_logp = self.actor_loss_and_grads(obs, actions, old_logp, adv)
            kl = float(np.mean(old_logp - new_logp))
            if kl > cfg.target_kl:
                break
        if self.kind == "gaussian":
            # Keep exploration from collapsing to nothing or exploding.
            np.clip(self.log_std, -2.5, 0.5, out=self.log_std)
        return {
            "actor_loss": round(actor_loss, 5),
            "critic_loss": round(critic_loss, 5),
            "kl": round(kl, 5),
            "epochs": epochs_run,
            "std": float(np.exp(self.log_std).mean()) if self.kind == "gaussian" else None,
        }
