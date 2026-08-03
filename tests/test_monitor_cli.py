# ═══════════════════════════════════════════════════════
# FinSight — Tests: Monitoring CLI
# ═══════════════════════════════════════════════════════
#
# Offline only. run_cycle/resume_cycle/pending_alerts_for are all mocked —
# this file is about the CLI's own argument handling and return codes, not
# about the graph, which is tested in test_monitor_graph.py.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.monitor.cli import main


@pytest.fixture(autouse=True)
def _no_side_effects():
    """Every CLI path touches init_db(); keep it offline and fast."""
    with (
        patch("src.core.logging_setup.configure_logging"),
        patch("src.core.tracing.configure_tracing"),
        patch("src.persistence.db.init_db"),
    ):
        yield


class TestArgumentValidation:
    def test_approve_without_resume_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--approve", "a1"])
        assert exc.value.code == 2

    def test_reject_without_resume_is_a_usage_error(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--reject", "a1"])
        assert exc.value.code == 2

    def test_no_flags_prints_help_and_exits_zero(self, capsys):
        assert main([]) == 0
        assert "usage" in capsys.readouterr().out.lower()


class TestOnce:
    def test_a_completed_cycle_returns_zero(self, capsys):
        state = {"cycle_id": "c1", "warmup": False, "fired": [], "merged": [], "suppressed": [], "candidates": []}

        with (
            patch("src.vectorstore.collections.ensure_collections"),
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.run_cycle", return_value=state) as mock_run,
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            assert main(["--once"]) == 0

        assert mock_run.call_args.kwargs["checkpointer"] is not None

    def test_a_warmup_cycle_says_so(self, capsys):
        state = {"cycle_id": "c1", "warmup": True, "fired": [], "merged": [], "suppressed": [], "candidates": []}

        with (
            patch("src.vectorstore.collections.ensure_collections"),
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.run_cycle", return_value=state),
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            assert main(["--once", "--warmup"]) == 0

        assert "Warmup complete" in capsys.readouterr().out

    def test_monitor_errors_return_one(self, capsys):
        state = {
            "cycle_id": "c1",
            "warmup": False,
            "fired": [],
            "merged": [],
            "suppressed": [],
            "candidates": [],
            "monitor_errors": ["news_monitor(AAPL): timeout"],
        }

        with (
            patch("src.vectorstore.collections.ensure_collections"),
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.run_cycle", return_value=state),
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            assert main(["--once"]) == 1

    def test_a_paused_cycle_returns_two(self, capsys):
        state = {
            "cycle_id": "c1",
            "warmup": False,
            "fired": [],
            "merged": [],
            "suppressed": [],
            "candidates": [],
            "pending_approval": [{"alert_id": "a1", "ticker": "AAPL", "headline": "h", "detail": "d"}],
            "__interrupt__": ["anything"],
        }

        with (
            patch("src.vectorstore.collections.ensure_collections"),
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.run_cycle", return_value=state),
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            assert main(["--once"]) == 2


class TestPending:
    def test_no_pending_cycles_says_so(self, capsys):
        with patch("src.persistence.repository.list_cycles", return_value=[]):
            assert main(["--pending"]) == 0
        assert "No cycles awaiting approval" in capsys.readouterr().out

    def test_pending_cycles_list_their_alerts(self, capsys):
        rows = [{"cycle_id": "c1", "status": "PENDING_APPROVAL"}]
        alerts = [{"alert_id": "a1", "ticker": "AAPL", "headline": "Apple filed an 8-K", "detail": "d"}]

        with (
            patch("src.persistence.repository.list_cycles", return_value=rows),
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.pending_alerts_for", return_value=alerts),
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            assert main(["--pending"]) == 0

        out = capsys.readouterr().out
        assert "c1" in out
        assert "Apple filed an 8-K" in out
        assert "a1" in out


class TestResume:
    def test_approve_and_reject_build_the_decisions_dict(self, capsys):
        state = {"cycle_id": "c1", "warmup": False, "fired": [], "merged": [], "suppressed": [], "candidates": []}

        with (
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.resume_cycle", return_value=state) as mock_resume,
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            code = main(["--resume", "c1", "--approve", "a1", "--reject", "a2"])

        assert code == 0
        args, kwargs = mock_resume.call_args
        assert args[0] == "c1"
        assert args[1] == {"a1": "approve", "a2": "reject"}

    def test_a_decisions_mismatch_prints_the_error_and_returns_one(self, capsys):
        with (
            patch("src.persistence.checkpointer.sync_checkpointer") as mock_ckpt,
            patch("src.monitor.graph.resume_cycle", side_effect=ValueError("decisions must cover exactly [...]")),
        ):
            mock_ckpt.return_value.__enter__.return_value = object()
            code = main(["--resume", "c1", "--approve", "wrong-id"])

        assert code == 1
        assert "decisions must cover exactly" in capsys.readouterr().out
