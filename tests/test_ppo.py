# ═══════════════════════════════════════════════════════
# FinSight — Tests: numpy MLP and PPO
# ═══════════════════════════════════════════════════════
# Every hand-written gradient is checked against central finite differences.
# If backprop is wrong, PPO "trains" anyway and just learns nothing — so this
# file is the guarantee that a flat learning curve means "no signal", not "bug".
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np
import pytest

from src.rl.nn import MLP, Adam, clip_grads
from src.rl.ppo import Episode, PPOConfig, PPOLearner, bernoulli_logp, gaussian_logp


def _numeric_grad(f, param: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    g = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        hi = f()
        param[i] = old - eps
        lo = f()
        param[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(1e-8, np.max(np.abs(a)) + np.max(np.abs(b))))


class TestMLP:
    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        net = MLP([5, 7, 6, 3], rng, out_scale=1.0)
        x = rng.standard_normal((4, 5))
        target = rng.standard_normal((4, 3))

        def loss() -> float:
            return float(0.5 * np.sum((net.forward(x) - target) ** 2))

        out = net.forward(x)
        grads = net.backward(out - target)
        for p, g in zip(net.params, grads):
            assert _rel_err(g, _numeric_grad(loss, p)) < 1e-6

    def test_output_layer_starts_small(self):
        net = MLP([4, 8, 2], np.random.default_rng(1), out_scale=0.01)
        assert np.abs(net.forward(np.ones((1, 4)))).max() < 0.1

    def test_adam_descends_a_quadratic(self):
        p = [np.array([3.0, -2.0])]
        opt = Adam(p, lr=0.1)
        for _ in range(300):
            opt.step(p, [2 * p[0]])
        assert np.allclose(p[0], 0, atol=1e-2)

    def test_clip_grads(self):
        g = clip_grads([np.array([3.0, 4.0])], max_norm=1.0)
        assert np.linalg.norm(g[0]) == pytest.approx(1.0)


class TestDistributions:
    def test_gaussian_logp_matches_closed_form(self):
        a, mu, ls = np.array([[0.3, -1.0]]), np.array([[0.0, 0.5]]), np.array([0.1, -0.4])
        std = np.exp(ls)
        expected = np.sum(-0.5 * ((a - mu) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi)))
        assert gaussian_logp(a, mu, ls)[0] == pytest.approx(expected)

    def test_bernoulli_logp_is_stable(self):
        lp = bernoulli_logp(np.array([[1.0], [0.0]]), np.array([[50.0], [50.0]]))
        assert lp[0] == pytest.approx(0.0, abs=1e-12)
        assert lp[1] == pytest.approx(-50.0)


@pytest.mark.parametrize("kind,act_dim", [("gaussian", 3), ("bernoulli", 1)])
@pytest.mark.parametrize("ent_coef", [0.0, 0.05])
class TestActorGradients:
    def test_surrogate_gradient_matches_finite_differences(self, kind, act_dim, ent_coef):
        rng = np.random.default_rng(3)
        learner = PPOLearner(kind, 4, act_dim, PPOConfig(hidden=(6,), ent_coef=ent_coef), rng)
        for p in learner.actor.params:  # bigger weights: a non-trivial policy
            p *= 30
        obs = rng.standard_normal((16, 4))
        if kind == "gaussian":
            actions = rng.standard_normal((16, act_dim))
            learner.log_std[:] = rng.normal(0, 0.3, act_dim)
        else:
            actions = (rng.random((16, 1)) < 0.5).astype(float)
        adv = rng.standard_normal(16)
        _, _, logp = learner.actor_loss_and_grads(obs, actions, np.zeros(16), adv)
        # Old log-probs near the current ones, some outside the clip range, so
        # both the active and the clipped branches are exercised.
        old_logp = logp + rng.normal(0, 0.15, 16)

        def loss() -> float:
            return learner.actor_loss_and_grads(obs, actions, old_logp, adv)[0]

        _, grads, _ = learner.actor_loss_and_grads(obs, actions, old_logp, adv)
        params = learner._actor_params()
        for p, g in zip(params, grads):
            assert _rel_err(g, _numeric_grad(loss, p, eps=1e-6)) < 1e-4


class TestCritic:
    def test_value_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(4)
        learner = PPOLearner("gaussian", 3, 2, PPOConfig(hidden=(5,)), rng)
        obs, targets = rng.standard_normal((8, 3)), rng.standard_normal(8)

        def loss() -> float:
            return learner.critic_loss_and_grads(obs, targets)[0]

        _, grads = learner.critic_loss_and_grads(obs, targets)
        for p, g in zip(learner.critic.params, grads):
            assert _rel_err(g, _numeric_grad(loss, p)) < 1e-6


class TestLearning:
    def test_solves_a_contextual_bandit(self):
        """Reward 1 for action = sign of the first feature. PPO must learn it."""
        rng = np.random.default_rng(5)
        learner = PPOLearner("bernoulli", 2, 1, PPOConfig(hidden=(16,), actor_lr=3e-3, ent_coef=0.01), rng)
        for _ in range(40):
            eps = []
            for _ in range(8):
                ep = Episode()
                for _ in range(20):
                    o = rng.standard_normal(2)
                    a, lp = learner.act(o, rng)
                    ep.add(o, a, lp, float(a[0] == float(o[0] > 0)))
                eps.append(ep)
            learner.learn(eps, rng)
        test = rng.standard_normal((200, 2))
        acc = np.mean([learner.act_deterministic(o)[0] == float(o[0] > 0) for o in test])
        assert acc > 0.9

    def test_gaussian_policy_moves_toward_reward(self):
        """Reward is −(a − 2)²: the mean action must move from 0 toward 2."""
        rng = np.random.default_rng(6)
        learner = PPOLearner("gaussian", 1, 1, PPOConfig(hidden=(8,), actor_lr=3e-3), rng)
        for _ in range(60):
            eps = []
            for _ in range(8):
                ep = Episode()
                for _ in range(16):
                    o = np.array([1.0])
                    a, lp = learner.act(o, rng)
                    ep.add(o, a, lp, -float((a[0] - 2.0) ** 2))
                eps.append(ep)
            learner.learn(eps, rng)
        assert learner.act_deterministic(np.array([1.0]))[0] > 1.0

    def test_learn_reports_stats(self):
        rng = np.random.default_rng(7)
        learner = PPOLearner("gaussian", 2, 2, PPOConfig(hidden=(4,)), rng)
        ep = Episode()
        for _ in range(10):
            o = rng.standard_normal(2)
            a, lp = learner.act(o, rng)
            ep.add(o, a, lp, 0.1)
        ep.last_obs = rng.standard_normal(2)
        stats = learner.learn([ep], rng)
        assert {"actor_loss", "critic_loss", "kl", "epochs"} <= set(stats)
