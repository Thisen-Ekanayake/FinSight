# ═══════════════════════════════════════════════════════
# FinSight — Alert Dispatch
# ═══════════════════════════════════════════════════════
#
# Purpose : Deliver a resolved alert (FIRED or human-approved) to the sinks
#           configured in NOTIFICATION_SINKS.
#
# Public API:
#   dispatch_alert(alert) -> bool     True if at least one sink delivered it
#
# ══ WHY "AT LEAST ONE" RATHER THAN "ALL" ══
#   The sinks are independent and best-effort: an SMTP outage must not turn
#   into a silently un-dispatched HIGH alert whose console/file record would
#   otherwise have made it visible. Each sink is tried, each failure is
#   logged with which sink and which alert, and dispatched is the OR of all
#   of them — the same asymmetric-cost stance dedup.py takes: a missed
#   channel is a nuisance, a missed alert is the failure mode that matters.
#
# ══ WHY THERE IS NO RETRY QUEUE ══
#   A retry queue needs its own durability story — which is exactly what the
#   alert already has. It is sitting in SQLite with whatever status
#   persist_cycle_node wrote, and `GET /monitor/alerts?status=FIRED` finds
#   anything the console/file/email sinks failed to deliver. Building a
#   second persistence layer for "alerts dispatch failed to send" would
#   duplicate the first one for no benefit a human re-running --decisions
#   does not already get for free.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
from typing import Callable

from src.core.config import DATA_DIR, ensure_data_dirs
from src.monitor.config import (
    ALERTS_LOG_PATH_NAME,
    EMAIL_APP_PASSWORD,
    EMAIL_ENABLED,
    EMAIL_FROM,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PORT,
    EMAIL_TO,
    NOTIFICATION_SINKS,
)
from src.monitor.state import Alert

logger = logging.getLogger(__name__)

ALERTS_LOG_PATH = DATA_DIR / ALERTS_LOG_PATH_NAME


def _dispatch_console(alert: Alert) -> None:
    """Log the alert at WARNING so it surfaces even with INFO-level noise filtered."""
    marker = "!!" if alert["severity"] == "HIGH" else "  "
    logger.warning(
        "ALERT %s [%s] %s %s — %s",
        marker,
        alert["severity"],
        alert["ticker"] or "MACRO",
        alert["headline"],
        alert["detail"],
    )


def _dispatch_file(alert: Alert) -> None:
    """
    Append one JSON line to data/alerts.log.

    A line per alert, not a rewritten file: the log is meant to be tailed
    (``tail -f data/alerts.log``), and rewriting the whole file on every
    dispatch would make that impossible and would not scale past a handful of
    alerts anyway.
    """
    ensure_data_dirs()
    record = {
        "alert_id": alert["alert_id"],
        "ticker": alert["ticker"],
        "alert_type": alert["alert_type"],
        "severity": alert["severity"],
        "headline": alert["headline"],
        "detail": alert["detail"],
        "fired_at": alert["fired_at"],
    }
    with ALERTS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _dispatch_email(alert: Alert) -> None:
    """
    Send one email via Gmail SMTP with an app password.

    Raises if EMAIL_ENABLED is false or the sender/recipient are unset. That
    is deliberate rather than a quiet no-op: a misconfigured sink that
    silently "succeeds" would let dispatch_alert report the alert as
    delivered when nothing was actually sent — the exact false-confidence
    failure this whole subsystem exists to avoid elsewhere. Logged loudly by
    dispatch_alert's exception handler instead, once per alert, until it is
    either configured or removed from NOTIFICATION_SINKS.
    """
    if not EMAIL_ENABLED:
        raise RuntimeError("email is in NOTIFICATION_SINKS but EMAIL_ENABLED=false — see docs/api_keys.md")

    if not (EMAIL_FROM and EMAIL_TO and EMAIL_APP_PASSWORD):
        raise RuntimeError("EMAIL_ENABLED=true but EMAIL_FROM/EMAIL_TO/EMAIL_APP_PASSWORD is not fully set")

    import smtplib
    from email.message import EmailMessage

    message = EmailMessage()
    message["Subject"] = f"[FinSight {alert['severity']}] {alert['ticker'] or 'MACRO'} — {alert['headline']}"
    message["From"] = EMAIL_FROM
    message["To"] = EMAIL_TO
    message.set_content(
        f"{alert['detail']}\n\n"
        f"ticker:      {alert['ticker'] or 'MACRO'}\n"
        f"type:        {alert['alert_type']}\n"
        f"severity:    {alert['severity']}\n"
        f"occurrences: {alert['occurrence_count']}\n"
        f"alert_id:    {alert['alert_id']}\n"
    )

    with smtplib.SMTP_SSL(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as smtp:
        smtp.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        smtp.send_message(message)


_SINKS: dict[str, Callable[[Alert], None]] = {
    "console": _dispatch_console,
    "file": _dispatch_file,
    "email": _dispatch_email,
}


def dispatch_alert(alert: Alert) -> bool:
    """
    Deliver one alert to every sink in NOTIFICATION_SINKS.

    Parameters
    ----------
    alert : Alert
        Must already be resolved — status FIRED. A caller dispatching a
        PENDING_APPROVAL or REJECTED alert is a bug upstream, not something
        this function should paper over.

    Returns
    -------
    bool
        True if at least one sink delivered it. False means every configured
        sink failed AND is worth investigating — the caller logs it into
        `dispatched` either way, so a False here is visible in the cycle
        report rather than silently dropped.
    """
    delivered = False

    for sink in NOTIFICATION_SINKS:
        func = _SINKS.get(sink)
        if func is None:
            logger.warning("Unknown notification sink %r in NOTIFICATION_SINKS — skipping", sink)
            continue
        try:
            func(alert)
            delivered = True
        except Exception:  # noqa: BLE001 - one sink's failure must not block the others
            logger.exception("Notification sink %r failed for alert %s", sink, alert["alert_id"])

    return delivered
