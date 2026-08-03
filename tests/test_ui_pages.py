# ═══════════════════════════════════════════════════════
# FinSight — Tests: Dashboard Pages
# ═══════════════════════════════════════════════════════
#
# Each page is run for real through Streamlit's own AppTest harness — the
# only thing substituted is src.ui.client, so this catches the class of bug
# a plain import can't: a widget reading a key that a mocked response
# doesn't have, a loop over the wrong field name, a form that never wires
# its submit button. Offline only.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

HEALTH = {
    "status": "ok",
    "environment": "development",
    "llm_backend": "vertex",
    "database": True,
    "checkpointer": True,
    "qdrant": True,
    "qdrant_detail": "connected",
    "scheduler_enabled": False,
    "scheduler_next_run_at": None,
}


def alert(**overrides):
    base = {
        "alert_id": "a1",
        "cycle_id": "c1",
        "ticker": "AAPL",
        "alert_type": "NEW_FILING",
        "severity": "HIGH",
        "status": "PENDING_APPROVAL",
        "headline": "AAPL filed an 8-K",
        "detail": "non-reliance on previously issued financials",
        "canonical_text": "c",
        "occurrence_count": 1,
        "first_seen_at": "2026-08-03T00:00:00+00:00",
        "last_seen_at": "2026-08-03T00:00:00+00:00",
        "fired_at": "2026-08-03T00:00:00+00:00",
        "parent_alert_id": None,
        "evidence": [],
    }
    base.update(overrides)
    return base


class TestHome:
    def test_renders_with_no_pending_cycles(self):
        with (
            patch("src.ui.client.health", return_value=HEALTH),
            patch("src.ui.client.list_cycles", return_value=[]),
        ):
            at = AppTest.from_file("src/ui/pages/home.py").run()
        assert not at.exception
        assert at.title[0].value == "FinSight"

    def test_warns_when_the_api_is_unreachable(self):
        from src.ui.client import ApiError

        with patch("src.ui.client.health", side_effect=ApiError("connection refused")):
            at = AppTest.from_file("src/ui/pages/home.py").run()
        assert not at.exception
        assert at.warning

    def test_flags_pending_approvals(self):
        with (
            patch("src.ui.client.health", return_value=HEALTH),
            patch("src.ui.client.list_cycles", return_value=[{"cycle_id": "c1"}]),
        ):
            at = AppTest.from_file("src/ui/pages/home.py").run()
        assert not at.exception
        assert any("paused" in w.value for w in at.warning)


class TestResearch:
    def test_renders_with_no_runs_yet(self):
        with patch("src.ui.client.list_runs", return_value=[]):
            at = AppTest.from_file("src/ui/pages/research.py").run()
        assert not at.exception

    def test_asking_a_question_shows_the_answer(self):
        response = {
            "thread_id": "research:1",
            "query": "q",
            "answer": "Apple's gross margin rose.",
            "citations": [
                {"source_type": "EDGAR", "source_id": "0000320193-26-000010", "url": "https://x", "as_of": "2026-08-03"}
            ],
            "conflicts": [],
            "verification": {
                "citation_coverage": 0.95,
                "passed": True,
                "verified_count": 3,
                "unsupported_claims": [],
                "invalid_source_ids": [],
            },
            "agents_used": ["fundamentals"],
            "tickers": ["AAPL"],
            "branch_count": 1,
            "repair_count": 0,
            "errors": [],
            "latency_ms": 4200,
        }
        with (
            patch("src.ui.client.list_runs", return_value=[]),
            patch("src.ui.client.ask", return_value=response) as mock_ask,
        ):
            at = AppTest.from_file("src/ui/pages/research.py").run()
            at.text_area[0].input("How did Apple's margin trend?")
            at.button[0].click().run()

        assert not at.exception
        mock_ask.assert_called_once()
        assert "gross margin rose" in at.session_state["last_result"]["answer"]


class TestMonitor:
    def test_renders_all_three_tabs_with_empty_data(self):
        with (
            patch("src.ui.client.list_cycles", return_value=[]),
            patch("src.ui.client.list_alerts", return_value=[]),
            patch("src.ui.client.list_decisions", return_value=[]),
        ):
            at = AppTest.from_file("src/ui/pages/monitor.py").run()
        assert not at.exception

    def test_running_a_cycle_shows_the_result(self):
        result = {
            "cycle_id": "c1",
            "status": "COMPLETE",
            "warmup": False,
            "candidate_count": 2,
            "fired": [alert(status="FIRED")],
            "merged": [],
            "suppressed": [],
            "pending_approval": [],
            "errors": [],
            "duration_ms": 100,
        }
        with (
            patch("src.ui.client.list_cycles", return_value=[]),
            patch("src.ui.client.list_alerts", return_value=[]),
            patch("src.ui.client.list_decisions", return_value=[]),
            patch("src.ui.client.run_cycle", return_value=result),
        ):
            at = AppTest.from_file("src/ui/pages/monitor.py").run()
            at.button[0].click().run()

        assert not at.exception
        assert at.success

    def test_a_paused_cycle_warns_instead_of_claiming_success(self):
        result = {
            "cycle_id": "c1",
            "status": "PENDING_APPROVAL",
            "warmup": False,
            "candidate_count": 1,
            "fired": [alert()],
            "merged": [],
            "suppressed": [],
            "pending_approval": [alert()],
            "errors": [],
            "duration_ms": 100,
        }
        with (
            patch("src.ui.client.list_cycles", return_value=[]),
            patch("src.ui.client.list_alerts", return_value=[]),
            patch("src.ui.client.list_decisions", return_value=[]),
            patch("src.ui.client.run_cycle", return_value=result),
        ):
            at = AppTest.from_file("src/ui/pages/monitor.py").run()
            at.button[0].click().run()

        assert not at.exception
        assert at.warning
        assert not at.success


class TestApprovals:
    def test_nothing_pending_says_so(self):
        with patch("src.ui.client.list_cycles", return_value=[]):
            at = AppTest.from_file("src/ui/pages/approvals.py").run()
        assert not at.exception
        assert at.success

    def test_a_pending_cycle_shows_its_alert_and_a_reject_default(self):
        with (
            patch(
                "src.ui.client.list_cycles",
                return_value=[{"cycle_id": "c1", "tickers": ["AAPL"], "started_at": "2026-08-03T12:00:00+00:00"}],
            ),
            patch("src.ui.client.get_pending_alerts", return_value=[alert()]),
        ):
            at = AppTest.from_file("src/ui/pages/approvals.py").run()

        assert not at.exception
        assert len(at.radio) == 1
        assert at.radio[0].value == "Reject"  # the safe default: no decision means not dispatched

    def test_submitting_without_changing_anything_rejects_everything(self):
        with (
            patch(
                "src.ui.client.list_cycles",
                return_value=[{"cycle_id": "c1", "tickers": ["AAPL"], "started_at": "2026-08-03T12:00:00+00:00"}],
            ),
            patch("src.ui.client.get_pending_alerts", return_value=[alert(alert_id="a1")]),
            patch("src.ui.client.resume_cycle", return_value={"status": "COMPLETE"}) as mock_resume,
        ):
            at = AppTest.from_file("src/ui/pages/approvals.py").run()
            at.button[0].click().run()

        assert not at.exception
        mock_resume.assert_called_once_with("c1", {"a1": "reject"})

    def test_choosing_approve_sends_approve(self):
        with (
            patch(
                "src.ui.client.list_cycles",
                return_value=[{"cycle_id": "c1", "tickers": ["AAPL"], "started_at": "2026-08-03T12:00:00+00:00"}],
            ),
            patch("src.ui.client.get_pending_alerts", return_value=[alert(alert_id="a1")]),
            patch("src.ui.client.resume_cycle", return_value={"status": "COMPLETE"}) as mock_resume,
        ):
            at = AppTest.from_file("src/ui/pages/approvals.py").run()
            at.radio[0].set_value("Approve").run()
            at.button[0].click().run()

        assert not at.exception
        mock_resume.assert_called_once_with("c1", {"a1": "approve"})


class TestWatchlist:
    def test_renders_with_an_empty_watchlist(self):
        with patch("src.ui.client.get_watchlist", return_value=[]):
            at = AppTest.from_file("src/ui/pages/watchlist.py").run()
        assert not at.exception

    def test_renders_a_watched_ticker(self):
        items = [
            {
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "warmed_up": True,
                "added_at": "2026-08-01T00:00:00+00:00",
                "last_checked": {"filing": "2026-08-03T09:00:00+00:00"},
            }
        ]
        with patch("src.ui.client.get_watchlist", return_value=items):
            at = AppTest.from_file("src/ui/pages/watchlist.py").run()
        assert not at.exception


class TestAdmin:
    def test_renders_with_typical_data(self):
        budgets = [
            {
                "provider": "finnhub",
                "day": "2026-08-03",
                "used": 10,
                "limit": 60,
                "remaining": 50,
                "soft_limit_reached": False,
                "exhausted": False,
            }
        ]
        with (
            patch("src.ui.client.health", return_value=HEALTH),
            patch("src.ui.client.budgets", return_value=budgets),
            patch("src.ui.client.config", return_value={"environment": "development"}),
        ):
            at = AppTest.from_file("src/ui/pages/admin.py").run()
        assert not at.exception
