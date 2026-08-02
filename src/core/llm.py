# ═══════════════════════════════════════════════════════
# FinSight — LLM Factory
# ═══════════════════════════════════════════════════════
#
# Purpose : The ONLY place a Gemini chat model is constructed. Every node and
#           agent gets its model through get_llm(tier) so that backend choice,
#           rate limiting, retries, and model naming stay centralised.
#
# Public API:
#   get_llm(tier="flash") -> BaseChatModel
#   reset_llm_cache()
#
# Two backends, selected by GEMINI_BACKEND:
#   "vertex"   Vertex AI via Application Default Credentials. Needs a GCP
#              project with aiplatform.googleapis.com enabled. Billed per
#              token — there is no free tier.
#   "aistudio" Google AI Studio via GOOGLE_API_KEY. Free, but the free tier
#              meters requests-per-minute.
#
# Why the rate limiter matters in BOTH cases:
#   On aistudio it stops 429s. On vertex, quota is high enough that nothing
#   stops a runaway parallel fan-out from spending real money — so there the
#   limiter is a COST control. A shared per-tier limiter throttles every call
#   in-process, including the concurrent Send() branches that would otherwise
#   all fire at once.
#
# Usage:
#   from src.core.llm import get_llm
#   llm = get_llm("flash")
#   structured = get_llm("pro").with_structured_output(MySchema)
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.config import (
    GCP_LOCATION,
    GCP_PROJECT,
    GEMINI_BACKEND,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_RPM,
    GEMINI_TEMPERATURE,
    MODEL_BY_TIER,
    ModelTier,
    require_key,
    validate_llm_credentials,
)

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Cache: one model instance (and one rate limiter) per (backend, tier, temp).
# Sharing the limiter across callers is the entire point — a per-call limiter
# would let N parallel branches each fire at the full rate.
_LLM_CACHE: dict[tuple[str, str, float], "BaseChatModel"] = {}
_RATE_LIMITERS: dict[str, Any] = {}


def _get_rate_limiter(tier: str) -> Any:
    """Return the process-wide rate limiter for a model tier, creating it once."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    if tier not in _RATE_LIMITERS:
        rpm = GEMINI_RPM.get(tier, 10)
        # check_every_n_seconds is polling granularity, not the rate itself.
        # 0.1s keeps latency low while the bucket refills.
        _RATE_LIMITERS[tier] = InMemoryRateLimiter(
            requests_per_second=rpm / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=max(1, rpm // 2),
        )
        logger.info("Rate limiter tier=%s: %d req/min (%.3f req/s)", tier, rpm, rpm / 60.0)
    return _RATE_LIMITERS[tier]


def _build_vertex(model: str, temp: float, max_tokens: int, tier: str) -> BaseChatModel:
    """
    Construct a Gemini model routed through Vertex AI.

    Auth comes from Application Default Credentials — no key is passed. The
    separate ``langchain-google-vertexai`` package is deliberately NOT used:
    its ``ChatVertexAI`` is deprecated in favour of this class, which reaches
    both backends through the unified google-genai SDK.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
        temperature=temp,
        max_output_tokens=max_tokens,
        max_retries=GEMINI_MAX_RETRIES,
        rate_limiter=_get_rate_limiter(tier),
    )


def _build_aistudio(model: str, temp: float, max_tokens: int, tier: str) -> BaseChatModel:
    """Construct a Gemini model routed through AI Studio, using GOOGLE_API_KEY."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=require_key("GOOGLE_API_KEY"),
        temperature=temp,
        max_output_tokens=max_tokens,
        max_retries=GEMINI_MAX_RETRIES,
        rate_limiter=_get_rate_limiter(tier),
    )


def get_llm(
    tier: ModelTier = "flash",
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> BaseChatModel:
    """
    Build (or reuse) a rate-limited Gemini chat model.

    Tier guidance
    -------------
    ``flash``
        Routing, monitors, extraction, alert summarisation. The default —
        use it unless there is a measured reason not to.
    ``pro``
        Final synthesis, the citation verifier's qualitative stage, and
        LLM-judge evaluators. Several times the cost per token; use sparingly.

    Parameters
    ----------
    tier : {"flash", "pro"}, default "flash"
        Which model tier to use.
    temperature : float, optional
        Overrides GEMINI_TEMPERATURE. Use 0.0 for anything parsed downstream.
    max_output_tokens : int, optional
        Overrides GEMINI_MAX_OUTPUT_TOKENS.

    Returns
    -------
    BaseChatModel
        A cached, rate-limited chat model for the configured backend.

    Raises
    ------
    MissingCredentialError
        If the selected backend's credentials are absent.
    ConfigurationError
        If GEMINI_BACKEND is not a recognised value.
    """
    temp = GEMINI_TEMPERATURE if temperature is None else temperature
    max_tokens = max_output_tokens or GEMINI_MAX_OUTPUT_TOKENS
    cache_key = (GEMINI_BACKEND, tier, temp)

    if cache_key not in _LLM_CACHE:
        validate_llm_credentials()
        model = MODEL_BY_TIER[tier]

        if GEMINI_BACKEND == "vertex":
            _LLM_CACHE[cache_key] = _build_vertex(model, temp, max_tokens, tier)
            logger.info(
                "Built LLM backend=vertex tier=%s model=%s project=%s location=%s temp=%.2f",
                tier,
                model,
                GCP_PROJECT,
                GCP_LOCATION,
                temp,
            )
        else:
            _LLM_CACHE[cache_key] = _build_aistudio(model, temp, max_tokens, tier)
            logger.info("Built LLM backend=aistudio tier=%s model=%s temp=%.2f", tier, model, temp)

    return _LLM_CACHE[cache_key]


def reset_llm_cache() -> None:
    """Drop cached models and limiters. For tests that patch configuration."""
    _LLM_CACHE.clear()
    _RATE_LIMITERS.clear()
