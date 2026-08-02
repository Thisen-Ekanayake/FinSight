# ═══════════════════════════════════════════════════════
# FinSight — Logging Setup
# ═══════════════════════════════════════════════════════
#
# Purpose : One-time root logger configuration. Modules themselves just do
#           `logger = logging.getLogger(__name__)` and log lazily with %s.
#
# Public API:
#   configure_logging()
#
# Usage:
#   from src.core.logging_setup import configure_logging
#   configure_logging()          # call once, at an entrypoint only
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import sys

from src.core.config import LOG_LEVEL

_CONFIGURED = False

# Third-party loggers that are noisy at INFO and drown out our own output.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "qdrant_client",
    "apscheduler.executors.default",
    "yfinance",
    "peewee",
)


def configure_logging(*, level: str | None = None, force: bool = False) -> None:
    """
    Configure root logging for a FinSight entrypoint.

    Idempotent: repeat calls are no-ops unless ``force`` is set. Call this from
    entrypoints (CLI, FastAPI lifespan, scheduler) only — never from library
    modules, which would steal logging config from whoever imports them.

    Parameters
    ----------
    level : str, optional
        Override the LOG_LEVEL from the environment, e.g. ``"DEBUG"``.
    force : bool, default False
        Reconfigure even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logging.basicConfig(
        level=getattr(logging, (level or LOG_LEVEL), logging.INFO),
        format="%(asctime)s  %(levelname)-8s %(name)-32s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=force,
    )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
