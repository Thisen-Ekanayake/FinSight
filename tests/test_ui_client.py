# ═══════════════════════════════════════════════════════
# FinSight — Tests: UI HTTP Client
# ═══════════════════════════════════════════════════════
#
# Offline only. requests.request itself is mocked — this is about the
# client's own contract (paths, params, error mapping), not about a real
# API being up.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.ui import client


def fake_response(*, status=200, json_body=None, text="", ok=None):
    response = MagicMock()
    response.status_code = status
    response.ok = ok if ok is not None else 200 <= status < 300
    response.json.return_value = json_body if json_body is not None else {}
    response.text = text
    response.content = b"x" if json_body is not None else b""
    return response


class TestSuccess:
    def test_a_get_returns_the_parsed_body(self):
        with patch("requests.request", return_value=fake_response(json_body={"status": "ok"})) as mock_req:
            assert client.health() == {"status": "ok"}
        args, kwargs = mock_req.call_args
        assert args[0] == "GET"
        assert args[1].endswith("/health")

    def test_ask_posts_the_query(self):
        with patch("requests.request", return_value=fake_response(json_body={"answer": "..."})) as mock_req:
            client.ask("How did Apple's margin trend?")
        _, kwargs = mock_req.call_args
        assert kwargs["json"] == {"query": "How did Apple's margin trend?"}

    def test_ask_includes_thread_id_only_when_given(self):
        with patch("requests.request", return_value=fake_response(json_body={})) as mock_req:
            client.ask("q", thread_id="research:abc")
        _, kwargs = mock_req.call_args
        assert kwargs["json"]["thread_id"] == "research:abc"

    def test_resume_cycle_posts_decisions(self):
        with patch("requests.request", return_value=fake_response(json_body={"status": "COMPLETE"})) as mock_req:
            client.resume_cycle("c1", {"a1": "approve"})
        args, kwargs = mock_req.call_args
        assert args[1].endswith("/monitor/cycles/c1/resume")
        assert kwargs["json"] == {"decisions": {"a1": "approve"}}

    def test_list_alerts_drops_unset_filters(self):
        with patch("requests.request", return_value=fake_response(json_body=[])) as mock_req:
            client.list_alerts(ticker="AAPL")
        _, kwargs = mock_req.call_args
        assert kwargs["params"] == {"limit": 50, "ticker": "AAPL"}

    def test_a_delete_with_no_body_returns_none(self):
        with patch("requests.request", return_value=fake_response(status=204, json_body=None, text="")):
            assert client.remove_ticker("AAPL") is None


class TestErrors:
    def test_a_4xx_raises_with_the_servers_detail(self):
        response = fake_response(status=404, json_body={"detail": "No cycle nope"})
        with patch("requests.request", return_value=response):
            with pytest.raises(client.ApiError, match="No cycle nope"):
                client.get_thread("nope")

    def test_a_5xx_with_no_json_body_falls_back_to_raw_text(self):
        response = fake_response(status=500, text="Internal Server Error")
        response.json.side_effect = ValueError("not json")
        with patch("requests.request", return_value=response):
            with pytest.raises(client.ApiError, match="Internal Server Error"):
                client.health()

    def test_the_api_being_unreachable_is_an_apierror_not_a_traceback(self):
        with patch("requests.request", side_effect=requests.ConnectionError("refused")):
            with pytest.raises(client.ApiError, match="Could not reach the API"):
                client.health()
