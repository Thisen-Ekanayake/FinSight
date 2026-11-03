# ═══════════════════════════════════════════════════════
# FinSight — A small neural network, in numpy
# ═══════════════════════════════════════════════════════
#
# Purpose : The multi-layer perceptron PPO's actor and critic are built from,
#           with hand-written backprop and an Adam optimiser.
#
# Public API:
#   MLP(sizes, rng, out_scale), Adam(params, lr), clip_grads(grads, max_norm)
#
# ══ WHY NOT PYTORCH ══
#   These networks have a few thousand weights and learn from a few thousand
#   daily bars. PyTorch would add ~700 MB to the image and to every CI run to
#   do arithmetic a laptop CPU finishes in seconds in numpy. Backprop for a
#   tanh MLP is thirty lines, and tests/test_ppo.py checks every gradient
#   against finite differences — so the saving costs no trust. If these
#   models ever need a GPU, PPO in src/rl/ppo.py only talks to `forward`,
#   `backward` and `params`, and a torch module can stand in behind those.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import numpy as np


class MLP:
    """
    ``sizes = [in, h1, …, out]``: tanh hidden layers, a linear output layer.

    ``forward`` caches what ``backward`` needs, so call them in pairs on the
    same batch.
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator, out_scale: float = 0.01) -> None:
        self.sizes = sizes
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            # Orthogonal-ish init, small on the output layer: a fresh policy
            # starts near-uniform and a fresh critic near zero, the standard
            # PPO initialisation.
            gain = out_scale if last else np.sqrt(2.0)
            q, _ = np.linalg.qr(rng.standard_normal((max(fan_in, fan_out), min(fan_in, fan_out))))
            w = q if fan_in >= fan_out else q.T
            self.W.append(gain * w[:fan_in, :fan_out])
            self.b.append(np.zeros(fan_out))
        self._cache: list[np.ndarray] = []

    @property
    def params(self) -> list[np.ndarray]:
        return [*self.W, *self.b]

    def set_params(self, values: list[np.ndarray]) -> None:
        n = len(self.W)
        self.W = [v.copy() for v in values[:n]]
        self.b = [v.copy() for v in values[n:]]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``x`` is ``(batch, in)``; returns ``(batch, out)``."""
        h = np.atleast_2d(x)
        self._cache = [h]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            self._cache.append(h)
        return h

    def backward(self, grad_out: np.ndarray) -> list[np.ndarray]:
        """d(loss)/d(params) given d(loss)/d(output), in ``params`` order."""
        g = np.atleast_2d(grad_out)
        gW: list[np.ndarray] = [np.empty(0)] * len(self.W)
        gb: list[np.ndarray] = [np.empty(0)] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            h_in = self._cache[i]
            gW[i] = h_in.T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (1.0 - self._cache[i] ** 2)  # tanh'
        return [*gW, *gb]


class Adam:
    """Adam over a list of arrays, updated IN PLACE. ``step`` DESCENDS."""

    def __init__(self, params: list[np.ndarray], lr: float = 3e-4) -> None:
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            m_hat = m / (1 - 0.9**self.t)
            v_hat = v / (1 - 0.999**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def clip_grads(grads: list[np.ndarray], max_norm: float) -> list[np.ndarray]:
    """Scale ``grads`` so their global L2 norm is at most ``max_norm``."""
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    if norm <= max_norm or norm == 0.0:
        return grads
    return [g * (max_norm / norm) for g in grads]
