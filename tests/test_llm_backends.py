# ═══════════════════════════════════════════════════════
# FinSight — Tests: LLM Backend Selection & Credentials
# ═══════════════════════════════════════════════════════
#
# Offline only. Constructs no real model and makes no API call, so these
# consume zero Gemini quota and zero spend.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core import config, llm
from src.core.errors import ConfigurationError, MissingCredentialError


class TestValidateLlmCredentials:
    """Fail fast with an actionable message, not a 401 from inside a graph run."""

    def test_rejects_unknown_backend(self):
        with patch.object(config, "GEMINI_BACKEND", "openai"):
            with pytest.raises(ConfigurationError, match="must be 'vertex' or 'aistudio'"):
                config.validate_llm_credentials()

    def test_vertex_requires_a_project(self):
        with patch.object(config, "GEMINI_BACKEND", "vertex"), patch.object(config, "GCP_PROJECT", ""):
            with pytest.raises(MissingCredentialError, match="no GCP project"):
                config.validate_llm_credentials()

    def test_vertex_requires_adc_or_explicit_credentials(self):
        import google.auth.exceptions

        with (
            patch.object(config, "GEMINI_BACKEND", "vertex"),
            patch.object(config, "GCP_PROJECT", "some-project"),
            patch("google.auth.default", side_effect=google.auth.exceptions.DefaultCredentialsError("no creds")),
        ):
            with pytest.raises(MissingCredentialError, match="application-default login"):
                config.validate_llm_credentials()

    def test_vertex_passes_when_credentials_resolve(self):
        with (
            patch.object(config, "GEMINI_BACKEND", "vertex"),
            patch.object(config, "GCP_PROJECT", "some-project"),
            patch("google.auth.default", return_value=(MagicMock(), "some-project")),
        ):
            config.validate_llm_credentials()

    def test_aistudio_requires_an_api_key(self):
        with patch.object(config, "GEMINI_BACKEND", "aistudio"), patch.dict("os.environ", {}, clear=True):
            with pytest.raises(MissingCredentialError, match="GOOGLE_API_KEY"):
                config.validate_llm_credentials()

    def test_aistudio_passes_with_a_key(self):
        with (
            patch.object(config, "GEMINI_BACKEND", "aistudio"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "fake"}),
        ):
            config.validate_llm_credentials()


class TestRateLimiter:
    """One shared limiter per tier — a per-call limiter would defeat the purpose."""

    def setup_method(self):
        llm.reset_llm_cache()

    def teardown_method(self):
        llm.reset_llm_cache()

    def test_same_tier_returns_the_same_limiter(self):
        # Parallel Send() branches must contend for ONE bucket, otherwise each
        # branch fires at the full rate and the throttle does nothing.
        assert llm._get_rate_limiter("flash") is llm._get_rate_limiter("flash")

    def test_tiers_get_separate_limiters(self):
        assert llm._get_rate_limiter("flash") is not llm._get_rate_limiter("pro")

    def test_rate_matches_configured_rpm(self):
        limiter = llm._get_rate_limiter("flash")
        assert limiter.requests_per_second == pytest.approx(config.GEMINI_RPM["flash"] / 60.0)


class TestGetLlm:
    """Backend dispatch and caching, with the model class mocked out."""

    def setup_method(self):
        llm.reset_llm_cache()

    def teardown_method(self):
        llm.reset_llm_cache()

    def test_vertex_backend_sets_vertexai_flag(self):
        fake_cls = MagicMock()
        with (
            patch.object(llm, "GEMINI_BACKEND", "vertex"),
            patch.object(llm, "GCP_PROJECT", "proj-1"),
            patch.object(llm, "GCP_LOCATION", "us-central1"),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", fake_cls),
            patch.object(config, "GEMINI_BACKEND", "vertex"),
            patch.object(config, "GCP_PROJECT", "proj-1"),
            patch("google.auth.default", return_value=(MagicMock(), "proj-1")),
        ):
            llm.get_llm("flash")

        kwargs = fake_cls.call_args.kwargs
        assert kwargs["vertexai"] is True
        assert kwargs["project"] == "proj-1"
        assert kwargs["location"] == "us-central1"
        assert "google_api_key" not in kwargs

    def test_aistudio_backend_passes_api_key_and_no_vertex_flag(self):
        fake_cls = MagicMock()
        with (
            patch.object(llm, "GEMINI_BACKEND", "aistudio"),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", fake_cls),
            patch.object(config, "GEMINI_BACKEND", "aistudio"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}),
        ):
            llm.get_llm("flash")

        kwargs = fake_cls.call_args.kwargs
        assert kwargs["google_api_key"] == "fake-key"
        assert "vertexai" not in kwargs

    def test_caches_per_backend_tier_and_temperature(self):
        # side_effect so each construction yields a DISTINCT object — a plain
        # MagicMock returns the same instance every call, which would make the
        # identity assertions below vacuous.
        fake_cls = MagicMock(side_effect=lambda **kw: MagicMock())
        with (
            patch.object(llm, "GEMINI_BACKEND", "aistudio"),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", fake_cls),
            patch.object(config, "GEMINI_BACKEND", "aistudio"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}),
        ):
            first = llm.get_llm("flash")
            second = llm.get_llm("flash")
            different_temp = llm.get_llm("flash", temperature=0.9)

        assert first is second
        assert first is not different_temp
        assert fake_cls.call_count == 2

    def test_every_tier_is_constructible(self):
        fake_cls = MagicMock()
        with (
            patch.object(llm, "GEMINI_BACKEND", "aistudio"),
            patch("langchain_google_genai.ChatGoogleGenerativeAI", fake_cls),
            patch.object(config, "GEMINI_BACKEND", "aistudio"),
            patch.dict("os.environ", {"GOOGLE_API_KEY": "fake-key"}),
        ):
            for tier in config.MODEL_BY_TIER:
                llm.get_llm(tier)  # type: ignore[arg-type]

        models = [c.kwargs["model"] for c in fake_cls.call_args_list]
        assert models == [config.GEMINI_MODEL_FLASH, config.GEMINI_MODEL_PRO]


@pytest.mark.llm
class TestLiveGemini:
    """Real API call. Costs money on vertex — deselected by default."""

    def test_flash_round_trips(self):
        response = llm.get_llm("flash", max_output_tokens=64).invoke("Reply with exactly: OK")
        assert "OK" in str(response.content)
