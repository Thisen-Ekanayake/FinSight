# ═══════════════════════════════════════════════════════
# FinSight — Alert prioritiser (Thompson-sampling bandit)
# ═══════════════════════════════════════════════════════
#
# Purpose : Learn which monitoring alerts a person finds worth their time,
#           from what they already do — approve or reject HIGH alerts at the
#           approval gate — plus explicit 👍 / 👎 feedback, and rank new
#           alerts by it.
#
# Public API:
#   Label, fit(labels) -> Posterior, rank(alerts, posterior, explore, rng)
#
# ══ THE MODEL ══
#   One Beta-Bernoulli arm per feature value — alert_type, severity, ticker —
#   each counting "useful" vs "not useful". An alert's usefulness is the
#   average of its features' log-odds (a naive-Bayes style combination), so a
#   never-seen ticker still borrows strength from its type and severity.
#
#   Ranking is Thompson sampling: draw from each Beta, combine, sort. Alerts
#   the model is unsure about are sometimes lifted to the top, which is how it
#   keeps learning instead of burying a whole category after two thumbs-down.
#   `explore=False` ranks by posterior mean for a stable, repeatable view.
#
# ══ WHAT IT NEVER DOES ══
#   It reorders. It never hides, suppresses or auto-decides an alert — the
#   human approval gate stays exactly where it is.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FEATURES = ("alert_type", "severity", "ticker")
PRIOR = (1.0, 1.0)


@dataclass(frozen=True)
class Label:
    alert: dict[str, Any]
    useful: bool
    weight: float = 1.0
    source: str = "explicit"  # explicit 👍/👎, or "approval" from the HIGH-alert gate


@dataclass
class Posterior:
    counts: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    n_labels: int = 0
    n_explicit: int = 0
    n_implicit: int = 0

    def beta(self, feature: str, value: str) -> tuple[float, float]:
        a, b = self.counts.get(feature, {}).get(value, [0.0, 0.0])
        return PRIOR[0] + a, PRIOR[1] + b


def _key(alert: dict[str, Any], feature: str) -> str:
    return str(alert.get(feature) or "?").upper()


def fit(labels: list[Label]) -> Posterior:
    post = Posterior()
    for label in labels:
        post.n_labels += 1
        if label.source == "explicit":
            post.n_explicit += 1
        else:
            post.n_implicit += 1
        for feature in FEATURES:
            slot = post.counts.setdefault(feature, {}).setdefault(_key(label.alert, feature), [0.0, 0.0])
            slot[0 if label.useful else 1] += label.weight
    return post


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _combine(ps: list[float]) -> float:
    z = sum(_logit(p) for p in ps) / len(ps)
    return 1.0 / (1.0 + math.exp(-z))


def score(alert: dict[str, Any], post: Posterior, rng: np.random.Generator | None = None) -> dict[str, Any]:
    """Posterior-mean usefulness, a Thompson draw if ``rng`` is given, and per-feature evidence."""
    means, draws, parts, sds = [], [], [], []
    for feature in FEATURES:
        a, b = post.beta(feature, _key(alert, feature))
        mean = a / (a + b)
        means.append(mean)
        sds.append(math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
        if rng is not None:
            draws.append(float(rng.beta(a, b)))
        parts.append(
            {"feature": feature, "value": _key(alert, feature), "mean": round(mean, 4), "n": round(a + b - 2, 2)}
        )
    mean = _combine(means)
    return {
        "priority": round(mean, 4),
        "sampled": round(_combine(draws), 4) if draws else round(mean, 4),
        "uncertainty": round(float(np.mean(sds)), 4),
        "evidence": parts,
    }


def rank(
    alerts: list[dict[str, Any]],
    post: Posterior,
    *,
    explore: bool = False,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Return ``alerts`` with a ``priority`` block, most useful first."""
    rng = rng if rng is not None else np.random.default_rng()
    scored = [{**a, "priority": score(a, post, rng if explore else None)} for a in alerts]
    key = "sampled" if explore else "priority"
    return sorted(scored, key=lambda a: a["priority"][key], reverse=True)


def preferences(post: Posterior) -> dict[str, list[dict[str, Any]]]:
    """What the model has learned, per feature: posterior mean with a 90% credible band."""
    out: dict[str, list[dict[str, Any]]] = {}
    for feature in FEATURES:
        rows = []
        for value in post.counts.get(feature, {}):
            a, b = post.beta(feature, value)
            draws = np.random.default_rng(0).beta(a, b, 4000)
            rows.append(
                {
                    "value": value,
                    "mean": round(a / (a + b), 4),
                    "low": round(float(np.quantile(draws, 0.05)), 4),
                    "high": round(float(np.quantile(draws, 0.95)), 4),
                    "useful": round(a - PRIOR[0], 2),
                    "not_useful": round(b - PRIOR[1], 2),
                }
            )
        out[feature] = sorted(rows, key=lambda r: r["mean"], reverse=True)
    return out
