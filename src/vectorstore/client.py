# ═══════════════════════════════════════════════════════
# FinSight — Qdrant Client Factory
# ═══════════════════════════════════════════════════════
#
# Purpose : Build the Qdrant client, and REFUSE to proceed if the configured
#           instance belongs to another project.
#
# Public API:
#   get_qdrant_client(url=None, *, verify=True) -> QdrantClient
#   assert_not_foreign_instance(client) -> None
#   reset_client_cache()
#
# Why the guard exists:
#   This machine runs two Qdrants. Another project owns :6333 (collections
#   athena_content, image_embeddings); FinSight owns :6335. A stale QDRANT_URL
#   in .env — easy to do, since 6333 is the documented default everywhere —
#   would have FinSight creating collections inside someone else's database,
#   and `make clean` would then delete their volume.
#
#   Cheap insurance: on connect, list collections. If any belong to the other
#   project, raise instead of writing a single point.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.config import QDRANT_API_KEY, QDRANT_TIMEOUT, QDRANT_URL
from src.core.errors import InfrastructureError, QdrantIsolationError
from src.vectorstore.config import FORBIDDEN_COLLECTIONS

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_CLIENT_CACHE: dict[str, QdrantClient] = {}


def assert_not_foreign_instance(client: QdrantClient) -> None:
    """
    Verify the connected Qdrant is FinSight's own instance.

    Parameters
    ----------
    client : QdrantClient
        A connected client.

    Raises
    ------
    QdrantIsolationError
        If any collection belonging to another project is present.
    InfrastructureError
        If the instance cannot be reached at all.
    """
    try:
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        raise InfrastructureError(
            f"Cannot reach Qdrant at {QDRANT_URL}: {exc}\n"
            f"  Start it with:  make qdrant\n"
            f"  Then verify:    make qdrant-check"
        ) from exc

    trespass = existing & FORBIDDEN_COLLECTIONS
    if trespass:
        raise QdrantIsolationError(
            f"REFUSING to use the Qdrant at {QDRANT_URL} — it belongs to another project.\n"
            f"  Found foreign collections: {sorted(trespass)}\n"
            f"  FinSight runs its OWN instance on port 6335, not 6333.\n"
            f"  Fix: set QDRANT_URL=http://localhost:6335 in .env, then run `make qdrant`."
        )

    logger.debug("Qdrant isolation verified at %s (collections: %s)", QDRANT_URL, sorted(existing) or "none")


def get_qdrant_client(url: str | None = None, *, verify: bool = True) -> QdrantClient:
    """
    Build (or reuse) a Qdrant client for FinSight's instance.

    Parameters
    ----------
    url : str, optional
        Overrides QDRANT_URL. Mainly for tests and eval runs.
    verify : bool, default True
        Run the isolation assertion. Only disable when connecting to a known
        throwaway instance in tests.

    Returns
    -------
    QdrantClient

    Raises
    ------
    QdrantIsolationError
        If the instance belongs to another project.
    InfrastructureError
        If the instance is unreachable.
    """
    from qdrant_client import QdrantClient

    target = url or QDRANT_URL

    if target not in _CLIENT_CACHE:
        client = QdrantClient(
            url=target,
            api_key=QDRANT_API_KEY or None,
            timeout=QDRANT_TIMEOUT,
        )
        if verify:
            assert_not_foreign_instance(client)

        _CLIENT_CACHE[target] = client
        logger.info("Connected to Qdrant at %s", target)

    return _CLIENT_CACHE[target]


def reset_client_cache() -> None:
    """Close and drop cached clients. For tests and clean shutdown."""
    for client in _CLIENT_CACHE.values():
        try:
            client.close()
        except Exception:  # pragma: no cover - best effort
            pass
    _CLIENT_CACHE.clear()
