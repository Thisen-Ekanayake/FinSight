# ═══════════════════════════════════════════════════════
# FinSight — Alert prioritiser service
# ═══════════════════════════════════════════════════════
#
# Purpose : Assemble the prioritiser's labels and rank recent alerts.
#
# ══ TWO SOURCES OF LABELS ══
#   implicit   every HIGH alert passed the human approval gate. Delivered
#              (FIRED) means a person approved it; REJECTED means they did not.
#              That is a decision people are already making — weight 1.
#   explicit   this account's 👍 / 👎 on any alert — weight 2, and it replaces
#              the implicit label for the same alert.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any

import numpy as np

from src.finance import repository as finance_repo
from src.persistence import repository as monitor_repo
from src.rl.alert_bandit import Label, Posterior, fit, preferences, rank

EXPLICIT_WEIGHT = 2.0
IMPLICIT_WEIGHT = 1.0


def _labels(subject: str) -> tuple[list[Label], dict[str, bool]]:
    explicit = finance_repo.list_alert_feedback(subject=subject)
    labels: list[Label] = []
    for alert_id, useful in explicit.items():
        alert = monitor_repo.get_alert(alert_id)
        if alert is not None:
            labels.append(Label(dict(alert), useful, EXPLICIT_WEIGHT, "explicit"))

    for alert in monitor_repo.list_alerts(limit=1000, severity="HIGH"):
        if alert["alert_id"] in explicit:
            continue
        if alert["status"] in ("FIRED", "REJECTED"):
            labels.append(Label(dict(alert), alert["status"] == "FIRED", IMPLICIT_WEIGHT, "approval"))
    return labels, explicit


def posterior(subject: str) -> tuple[Posterior, dict[str, bool]]:
    labels, explicit = _labels(subject)
    return fit(labels), explicit


def ranked_alerts(*, subject: str, limit: int = 50, explore: bool = False) -> dict[str, Any]:
    post, explicit = posterior(subject)
    alerts: list[dict[str, Any]] = [dict(a) for a in monitor_repo.list_alerts(limit=limit)]
    for alert in alerts:
        alert["feedback"] = explicit.get(alert["alert_id"])
        # Heavy fields the ranking view does not need.
        alert.pop("canonical_text", None)
        alert.pop("metrics", None)
    return {
        "alerts": rank(alerts, post, explore=explore, rng=np.random.default_rng()),
        "labels": {"total": post.n_labels, "explicit": post.n_explicit, "from_approvals": post.n_implicit},
    }


def learned_preferences(*, subject: str) -> dict[str, Any]:
    post, _ = posterior(subject)
    return {
        "preferences": preferences(post),
        "labels": {"total": post.n_labels, "explicit": post.n_explicit, "from_approvals": post.n_implicit},
    }
