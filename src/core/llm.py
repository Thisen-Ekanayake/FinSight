# ═══════════════════════════════════════════════════════
# FinSight — LLM Factory
# ═══════════════════════════════════════════════════════
#
# Purpose : The ONLY place ChatGoogleGenerativeAI is constructed. Every node
#           and agent gets its model through get_llm(tier) so that rate
#           limiting, retries, and model choice stay centralised.
#
# Public API:
#   get_llm(tier="flash") -> BaseChatModel
#   reset_llm_cache()
#
# Why the rate limiter:
#   Gemini's free tier meters REQUESTS PER MINUTE, not dollars. Blowing the
#   quota surfaces as 429s mid-graph, which is far more annoying than a bill.
#   A shared per-tier InMemoryRateLimiter throttles every call in-process,
#   including the parallel fan-out branches, which are exactly what would
#   otherwise burst past the limit.
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
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_RPM,
    GEMINI_TEMPERATURE,
    MODEL_BY_TIER,
    ModelTier,
    require_key,
)

if TYPE_CHECKING:  # pragma: no cover
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Cache: one model instance (and one rate limiter) per (tier, temperature).
# Sharing the limiter across callers is the entire point — a per-call limiter
# would let N parallel branches each fire at the full rate.
_LLM_CACHE: dict[tuple[str, float], "BaseChatModel"] = {}
_RATE_LIMITERS: dict[str, Any] = {}


def _get_rate_limiter(tier: str) -> Any:
    """Return the process-wide rate limiter for a model tier, creating it once."""
    from langchain_core.rate_limiters import InMemoryRateLimiter

    if tier not in _RATE_LIMITERS:
        rpm = GEMINI_RPM.get(tier, 10)
        # check_every_n_seconds is the polling granularity, not the rate.
        # 0.1s keeps latency low while the bucket refills.
        _RATE_LIMITERS[tier] = InMemoryRateLimiter(
            requests_per_second=rpm / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=max(1, rpm // 2),
        )
        logger.info("Rate limiter for tier %r: %d req/min (%.3f req/s)", tier, rpm, rpm / 60.0)
    return _RATE_LIMITERS[tier]


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
        LLM-judge evaluators. Meaningfully scarcer quota; use sparingly.

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
        A cached, rate-limited ``ChatGoogleGenerativeAI``.

    Raises
    ------
    MissingCredentialError
        If GOOGLE_API_KEY is unset.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    temp = GEMINI_TEMPERATURE if temperature is None else temperature
    cache_key = (tier, temp)

    if cache_key not in _LLM_CACHE:
        api_key = require_key("GOOGLE_API_KEY")
        model = MODEL_BY_TIER[tier]

        _LLM_CACHE[cache_key] = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temp,
            max_output_tokens=max_output_tokens or GEMINI_MAX_OUTPUT_TOKENS,
            max_retries=GEMINI_MAX_RETRIES,
            rate_limiter=_get_rate_limiter(tier),
        )
        logger.info("Built LLM tier=%s model=%s temperature=%.2f", tier, model, temp)

    return _LLM_CACHE[cache_key]


def reset_llm_cache() -> None:
    """Drop cached models and limiters. For tests that patch configuration."""
    _LLM_CACHE.clear()
    _RATE_LIMITERS.clear()
