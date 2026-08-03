# ═══════════════════════════════════════════════════════
# FinSight — Tests: Alert Dispatch
# ═══════════════════════════════════════════════════════
#
# Offline only. No network, no SMTP, no LLM.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.monitor import dispatch as dispatch_module
from src.monitor.dispatch import dispatch_alert


def alert(**overrides):
    base = {
        "alert_id": "a1",
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "alert_type": "NEW_FILING",
        "severity": "HIGH",
        "status": "FIRED",
        "headline": "Apple filed an 8-K",
        "detail": "non-reliance on previously issued financials",
        "canonical_text": "c",
        "dedup_key": "k",
        "metrics": {},
        "evidence": [],
        "occurrence_count": 1,
        "first_seen_at": "2026-08-03T00:00:00+00:00",
        "last_seen_at": "2026-08-03T00:00:00+00:00",
        "fired_at": "2026-08-03T00:00:00+00:00",
        "parent_alert_id": None,
    }
    base.update(overrides)
    return base


class TestConsoleSink:
    def test_console_alone_delivers(self, caplog):
        with patch.object(dispatch_module, "NOTIFICATION_SINKS", ("console",)):
            assert dispatch_alert(alert()) is True

    def test_the_headline_reaches_the_log(self, caplog):
        import logging

        with patch.object(dispatch_module, "NOTIFICATION_SINKS", ("console",)), caplog.at_level(logging.WARNING):
            dispatch_alert(alert(headline="Apple filed an 8-K"))
        assert "Apple filed an 8-K" in caplog.text


class TestFileSink:
    def test_appends_one_json_line(self, tmp_path):
        log_path = tmp_path / "alerts.log"

        with (
            patch.object(dispatch_module, "NOTIFICATION_SINKS", ("file",)),
            patch.object(dispatch_module, "ALERTS_LOG_PATH", log_path),
        ):
            assert dispatch_alert(alert(alert_id="a1")) is True
            assert dispatch_alert(alert(alert_id="a2")) is True

        lines = log_path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["alert_id"] == "a1"
        assert json.loads(lines[1])["alert_id"] == "a2"

    def test_creates_the_data_dir_if_missing(self, tmp_path):
        log_path = tmp_path / "nested" / "alerts.log"

        with (
            patch.object(dispatch_module, "NOTIFICATION_SINKS", ("file",)),
            patch.object(dispatch_module, "ALERTS_LOG_PATH", log_path),
            patch("src.core.config.DATA_DIR", tmp_path / "nested"),
        ):
            assert dispatch_alert(alert()) is True
        assert log_path.exists()


class TestEmailSink:
    def test_disabled_email_does_not_report_delivered(self):
        """
        A misconfigured-but-listed sink must not make dispatch_alert claim
        success — that would be the exact false-confidence failure this
        subsystem exists to prevent elsewhere in the codebase.
        """
        with (
            patch.object(dispatch_module, "NOTIFICATION_SINKS", ("email",)),
            patch.object(dispatch_module, "EMAIL_ENABLED", False),
        ):
            assert dispatch_alert(alert()) is False

    def test_enabled_but_missing_recipient_does_not_report_delivered(self):
        with (
            patch.object(dispatch_module, "NOTIFICATION_SINKS", ("email",)),
            patch.object(dispatch_module, "EMAIL_ENABLED", True),
            patch.object(dispatch_module, "EMAIL_FROM", "me@example.com"),
            patch.object(dispatch_module, "EMAIL_TO", ""),
            patch.object(dispatch_module, "EMAIL_APP_PASSWORD", "x"),
        ):
            assert dispatch_alert(alert()) is False

    def test_a_fully_configured_send_calls_smtp(self):
        sent = {}

        class FakeSMTP:
            def __init__(self, host, port):
                sent["host"] = host
                sent["port"] = port

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def login(self, user, password):
                sent["login"] = (user, password)

            def send_message(self, message):
                sent["subject"] = message["Subject"]

        with (
            patch.object(dispatch_module, "NOTIFICATION_SINKS", ("email",)),
            patch.object(dispatch_module, "EMAIL_ENABLED", True),
            patch.object(dispatch_module, "EMAIL_FROM", "me@example.com"),
            patch.object(dispatch_module, "EMAIL_TO", "you@example.com"),
            patch.object(dispatch_module, "EMAIL_APP_PASSWORD", "app-password"),
            patch("smtplib.SMTP_SSL", FakeSMTP),
        ):
            assert dispatch_alert(alert(headline="Apple filed an 8-K")) is True

        assert sent["login"] == ("me@example.com", "app-password")
        assert "Apple filed an 8-K" in sent["subject"]


class TestPartialFailure:
    def test_one_sink_failing_does_not_block_another(self, tmp_path):
        log_path = tmp_path / "alerts.log"

        with (
            patch.object(dispatch_module, "NOTIFICATION_SINKS", ("email", "file")),
            patch.object(dispatch_module, "EMAIL_ENABLED", False),
            patch.object(dispatch_module, "ALERTS_LOG_PATH", log_path),
        ):
            assert dispatch_alert(alert()) is True  # file still delivered

        assert log_path.exists()

    def test_an_unknown_sink_name_is_skipped_not_fatal(self):
        with patch.object(dispatch_module, "NOTIFICATION_SINKS", ("carrier-pigeon", "console")):
            assert dispatch_alert(alert()) is True


@pytest.mark.parametrize(
    "sinks",
    [
        (),
    ],
)
class TestNoSinks:
    def test_no_configured_sinks_reports_not_delivered(self, sinks):
        with patch.object(dispatch_module, "NOTIFICATION_SINKS", sinks):
            assert dispatch_alert(alert()) is False
