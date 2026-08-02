# ═══════════════════════════════════════════════════════
# FinSight — Qdrant Collection Management
# ═══════════════════════════════════════════════════════
#
# Purpose : Create both collections with correct vector params AND explicit
#           payload indexes.
#
# Public API:
#   ensure_collections(recreate=False)
#   ensure_collection(name, indexes, recreate=False)
#   collection_stats(name)
#   drop_collection(name)
#
# ══ WHY PAYLOAD INDEXES MATTER ══
#   Qdrant will happily filter on an unindexed payload field — by scanning
#   every point in the collection. Nothing errors; it just gets slower as the
#   collection grows, which is the worst kind of bug because it passes every
#   test on small data.
#
#   EVERY research query filters `ticker == X`, and every dedup lookup filters
#   ticker + alert_type + status + a fired_at range. So those fields are
#   indexed explicitly here. This is the Qdrant lesson most tutorials skip.
#
# Usage:
#   python -m src.vectorstore.collections --ensure
#   python -m src.vectorstore.collections --stats
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
from typing import Any, TypedDict

from src.vectorstore.client import get_qdrant_client
from src.vectorstore.config import (
    ALERTS_PAYLOAD_INDEXES,
    COLLECTION_ALERTS,
    COLLECTION_FILINGS,
    DISTANCE,
    FILINGS_PAYLOAD_INDEXES,
    HNSW_EF_CONSTRUCT,
    HNSW_M,
    ON_DISK_VECTORS,
    VECTOR_SIZE,
)

logger = logging.getLogger(__name__)


class CollectionStats(TypedDict):
    """Snapshot of one collection's state."""

    name: str
    exists: bool
    points: int
    vectors: int
    indexed_fields: list[str]
    status: str


def _schema_type(kind: str) -> Any:
    """Map a config type name onto a Qdrant PayloadSchemaType."""
    from qdrant_client.models import PayloadSchemaType

    return {
        "keyword": PayloadSchemaType.KEYWORD,
        "integer": PayloadSchemaType.INTEGER,
        "float": PayloadSchemaType.FLOAT,
        "bool": PayloadSchemaType.BOOL,
        "text": PayloadSchemaType.TEXT,
    }[kind]


def ensure_collection(
    name: str,
    indexes: dict[str, str],
    *,
    recreate: bool = False,
) -> bool:
    """
    Create a collection and its payload indexes if absent.

    Idempotent: safe to call on every startup. Existing collections are left
    alone unless ``recreate`` is set, but missing payload indexes are always
    added — so an index introduced later is picked up without a full re-ingest.

    Parameters
    ----------
    name : str
        Collection name.
    indexes : dict
        ``{payload_field: type_name}`` to index. Types are the keys of
        _schema_type.
    recreate : bool, default False
        DESTRUCTIVE. Drop and rebuild, discarding all points.

    Returns
    -------
    bool
        True if the collection was created, False if it already existed.
    """
    from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

    client = get_qdrant_client()

    if recreate and client.collection_exists(name):
        logger.warning("Dropping collection %s and all its points", name)
        client.delete_collection(name)

    if client.collection_exists(name):
        _ensure_indexes(name, indexes)
        return False

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance[DISTANCE.upper()],
            on_disk=ON_DISK_VECTORS,
        ),
        hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
    )
    logger.info("Created collection %s (%d-d, %s, m=%d)", name, VECTOR_SIZE, DISTANCE, HNSW_M)

    _ensure_indexes(name, indexes)
    return True


def _ensure_indexes(name: str, indexes: dict[str, str]) -> None:
    """Create any payload indexes that do not already exist."""
    client = get_qdrant_client()

    info = client.get_collection(name)
    existing = set((info.payload_schema or {}).keys())

    for field, kind in indexes.items():
        if field in existing:
            continue
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=_schema_type(kind),
        )
        logger.info("  payload index: %s.%s (%s)", name, field, kind)


def ensure_collections(*, recreate: bool = False) -> dict[str, bool]:
    """
    Create both FinSight collections.

    ``finsight_filings``
        SEC filing narrative chunks. Written in bulk, read with a ticker
        filter on essentially every query.
    ``finsight_alerts``
        Fired alerts, used as the dedup index. A completely different access
        pattern: frequent small writes, payload mutation, and range filters
        on timestamps.

    Returns
    -------
    dict
        ``{collection_name: was_created}``.
    """
    return {
        COLLECTION_FILINGS: ensure_collection(COLLECTION_FILINGS, FILINGS_PAYLOAD_INDEXES, recreate=recreate),
        COLLECTION_ALERTS: ensure_collection(COLLECTION_ALERTS, ALERTS_PAYLOAD_INDEXES, recreate=recreate),
    }


def collection_stats(name: str) -> CollectionStats:
    """Return a snapshot of a collection, tolerating its absence."""
    client = get_qdrant_client()

    if not client.collection_exists(name):
        return CollectionStats(name=name, exists=False, points=0, vectors=0, indexed_fields=[], status="missing")

    info = client.get_collection(name)
    return CollectionStats(
        name=name,
        exists=True,
        points=info.points_count or 0,
        # indexed_vectors_count trails points_count while HNSW builds in the
        # background, so a gap right after ingest is expected, not a fault.
        vectors=info.indexed_vectors_count or 0,
        indexed_fields=sorted((info.payload_schema or {}).keys()),
        status=str(info.status),
    )


def drop_collection(name: str) -> bool:
    """Delete a collection and every point in it. Returns True if it existed."""
    client = get_qdrant_client()
    if not client.collection_exists(name):
        return False
    client.delete_collection(name)
    logger.warning("Dropped collection %s", name)
    return True


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Create or inspect FinSight's collections."""
    parser = argparse.ArgumentParser(description="Manage Qdrant collections")
    parser.add_argument("--ensure", action="store_true", help="create collections and indexes if absent")
    parser.add_argument("--stats", action="store_true", help="show collection statistics")
    parser.add_argument("--recreate", action="store_true", help="DESTRUCTIVE: drop and rebuild")
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    if args.recreate:
        confirm = input("This DELETES all indexed data. Type 'yes' to continue: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1

    if args.ensure or args.recreate:
        for name, created in ensure_collections(recreate=args.recreate).items():
            print(f"  {name:22s} {'created' if created else 'already existed'}")

    if args.stats or not (args.ensure or args.recreate):
        print()
        for name in (COLLECTION_FILINGS, COLLECTION_ALERTS):
            stats = collection_stats(name)
            print(f"  {stats['name']:22s} {stats['status']:10s} points={stats['points']:,}")
            print(f"  {'':22s} indexed: {', '.join(stats['indexed_fields']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
