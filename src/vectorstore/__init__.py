# ═══════════════════════════════════════════════════════
# FinSight — Vector Store Package
# ═══════════════════════════════════════════════════════
#
# Qdrant access: client factory with the cross-project isolation guard,
# collection definitions, chunking, ingest, and search.
#
# Public API:
#   get_qdrant_client, assert_not_foreign_instance
#   COLLECTION_FILINGS, COLLECTION_ALERTS
#
# Phases 2+ add: embeddings, collections, chunking, ingest, search.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from src.vectorstore.client import (
    assert_not_foreign_instance,
    get_qdrant_client,
    reset_client_cache,
)
from src.vectorstore.config import (
    COLLECTION_ALERTS,
    COLLECTION_FILINGS,
    FORBIDDEN_COLLECTIONS,
    VECTOR_SIZE,
)

__all__ = [
    "get_qdrant_client",
    "assert_not_foreign_instance",
    "reset_client_cache",
    "COLLECTION_FILINGS",
    "COLLECTION_ALERTS",
    "FORBIDDEN_COLLECTIONS",
    "VECTOR_SIZE",
]
