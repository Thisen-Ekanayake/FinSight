# ═══════════════════════════════════════════════════════
# FinSight — Alert Store (Qdrant as a working data structure)
# ═══════════════════════════════════════════════════════
#
# Purpose : Every Qdrant operation the dedup engine performs, in one place.
#
# Public API:
#   find_exact(dedup_key)                O(1) point fetch — the free fast path
#   search_similar(vector, ...)          filtered vector search
#   upsert_alert(alert, vector)          write or overwrite one point
#   bump_occurrence(alert_id, ...)       payload mutation, no re-embed
#   update_centroid(alert_id, vector)    running mean of the cluster
#   prune_expired(now)                   retention sweep
#   alert_stats()                        collection snapshot
#
# ══ THIS IS THE PHASE-2 COLLECTION'S OPPOSITE ══
#   finsight_filings is written once in bulk and then only read. This one is
#   written constantly, in ones and twos, and mutated in place:
#
#     set_payload   increment a counter without touching the vector
#     upsert        overwrite a vector without re-writing the payload
#     retrieve      fetch by id, no search, no filter
#     scroll        exhaustive listing for the retention sweep
#     query_points  filtered similarity, the actual dedup decision
#
#   Which is the point of Phase 6: Qdrant stops being "a place to put
#   embeddings" and becomes a data structure with an access pattern.
#
# ══ THE POINT ID IS DERIVED, NOT RANDOM ══
#   alert_id = uuid5(dedup_key), so the exact-match check is a retrieve() by
#   id — no filter, no payload index consulted, no scan. A random id would
#   force a filtered scroll on every candidate to answer a question the id
#   itself can answer.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.monitor.config import DEDUP_ACTIVE_STATUSES
from src.monitor.state import Alert
from src.vectorstore.config import (
    ALERT_RETENTION_SECONDS,
    COLLECTION_ALERTS,
    DEDUP_SEARCH_LIMIT,
    DEDUP_WINDOW_SECONDS,
)

logger = logging.getLogger(__name__)

# Which collection the engine reads and writes. A module constant rather than a
# literal so an eval run can point at a throwaway collection — the Phase 7
# cycle-replay suite builds finsight_alerts_eval_{uuid} and drops it after,
# which is only possible if this is patchable in one place.
ALERT_COLLECTION: str = COLLECTION_ALERTS


def _epoch(value: str | datetime) -> int:
    """Parse an ISO timestamp into a UTC epoch second, for range filters."""
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return int(stamp.timestamp())


def to_payload(alert: Alert) -> dict[str, Any]:
    """
    Flatten an Alert into a Qdrant payload.

    ``fired_at`` is stored as an INTEGER epoch second alongside the ISO string,
    because Qdrant's Range filter compares numbers and the payload index on
    that field is declared as an integer. Storing only the ISO string would
    make the per-type time window silently unfilterable — and an unfiltered
    dedup search matches alerts from months ago.
    """
    return {
        "alert_id": alert["alert_id"],
        "ticker": alert["ticker"],
        "alert_type": alert["alert_type"],
        "severity": alert["severity"],
        "status": alert["status"],
        "dedup_key": alert["dedup_key"],
        "fired_at": _epoch(alert["fired_at"]),
        "fired_at_iso": alert["fired_at"],
        "first_seen_at": alert["first_seen_at"],
        "last_seen_at": alert["last_seen_at"],
        "occurrence_count": alert["occurrence_count"],
        "canonical_text": alert["canonical_text"],
        "headline": alert["headline"],
        "company_name": alert["company_name"],
        "evidence_refs": [c.get("source_id", "") for c in alert.get("evidence") or []],
        "parent_alert_id": alert.get("parent_alert_id"),
    }


def find_exact(dedup_key: str, *, alert_id: str) -> dict[str, Any] | None:
    """
    Look up an alert by its derived point id.

    The free path: no embedding, no LLM, no vector search, no filter. Around
    90% of duplicates in a steady-state cycle resolve here — the same filing
    seen twice, the same article re-served by the feed, the same price move
    re-measured an hour later.

    Parameters
    ----------
    dedup_key : str
        Only used to confirm the fetched point is really the same event; the
        id is what does the lookup.
    alert_id : str
        ``alert_id_for(dedup_key)``.

    Returns
    -------
    dict or None
        The stored payload, or None if this event has not been seen.
    """
    from src.vectorstore.client import get_qdrant_client

    client = get_qdrant_client()

    points = client.retrieve(collection_name=ALERT_COLLECTION, ids=[alert_id], with_payload=True)
    if not points:
        return None

    payload = points[0].payload or {}

    # A uuid5 collision is not a practical concern, but a stale point written
    # under a previous ALERT_NAMESPACE would be — and silently treating an
    # unrelated event as a duplicate is the one failure mode this whole module
    # is built to avoid. Cheap to check, so check.
    if payload.get("dedup_key") and payload["dedup_key"] != dedup_key:
        logger.error(
            "Point %s holds dedup_key %s, expected %s — refusing to treat it as a duplicate",
            alert_id,
            payload["dedup_key"],
            dedup_key,
        )
        return None

    if payload.get("status") not in DEDUP_ACTIVE_STATUSES:
        # Present but retired (dismissed, expired). The event is genuinely new
        # again as far as alerting is concerned.
        return None

    return payload


def search_similar(
    vector: list[float],
    *,
    ticker: str,
    alert_type: str,
    now: datetime | None = None,
    limit: int = DEDUP_SEARCH_LIMIT,
    score_threshold: float | None = None,
) -> list[tuple[float, dict[str, Any]]]:
    """
    Find previously-fired alerts semantically close to this candidate.

    ══ THE FILTERS ARE HARD, THE SIMILARITY IS SOFT ══
    ticker and alert_type are payload filters, not embedded terms. "AAPL fell"
    and "MSFT fell" are lexically almost identical and completely unrelated, so
    asking cosine similarity to separate them is asking the wrong question of
    the wrong tool. The filter answers it exactly and for free — and every
    field used here is payload-indexed, so it stays free as the collection
    grows.

    The time window is per alert_type, because the right dedup horizon is a
    property of the underlying event's natural frequency: a 5% drop tomorrow is
    a NEW event, while a 10-K filed today is the same 10-K in three weeks.

    Parameters
    ----------
    vector : list of float
        The candidate's symmetric embedding.
    ticker : str
        Empty for macro alerts, which is a valid scope and filters exactly.
    alert_type : str
        One of the AlertType values.
    now : datetime, optional
        Injected for tests.
    limit : int
        Neighbours to consider.
    score_threshold : float, optional
        Floor passed to Qdrant, so obviously-unrelated points are never
        returned at all.

    Returns
    -------
    list of tuple
        ``(score, payload)``, highest first.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

    from src.vectorstore.client import get_qdrant_client

    moment = now or datetime.now(timezone.utc)
    window = DEDUP_WINDOW_SECONDS.get(alert_type, max(DEDUP_WINDOW_SECONDS.values()))
    floor = int(moment.timestamp()) - window

    query_filter = Filter(
        must=[
            FieldCondition(key="ticker", match=MatchValue(value=ticker.upper())),
            FieldCondition(key="alert_type", match=MatchValue(value=alert_type)),
            FieldCondition(key="status", match=MatchAny(any=list(DEDUP_ACTIVE_STATUSES))),
            FieldCondition(key="fired_at", range=Range(gte=floor)),
        ]
    )

    client = get_qdrant_client()
    response = client.query_points(
        collection_name=ALERT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )

    return [(float(point.score), point.payload or {}) for point in response.points]


def upsert_alert(alert: Alert, vector: list[float]) -> None:
    """
    Write one alert point, overwriting any point with the same id.

    Idempotent by construction: the id is derived from the dedup key, so
    re-observing an event rewrites its point rather than adding a competing
    near-duplicate vector that would then dilute future searches.
    """
    from qdrant_client.models import PointStruct

    from src.vectorstore.client import get_qdrant_client

    get_qdrant_client().upsert(
        collection_name=ALERT_COLLECTION,
        points=[PointStruct(id=alert["alert_id"], vector=vector, payload=to_payload(alert))],
        # wait=True: the very next candidate in this same cycle may need to
        # dedup against this point. The bulk-ingest path can afford eventual
        # visibility; a within-cycle read-after-write cannot.
        wait=True,
    )
    logger.debug("Upserted alert point %s (%s %s)", alert["alert_id"], alert["ticker"], alert["alert_type"])


def bump_occurrence(alert_id: str, *, count: int, last_seen_at: str) -> None:
    """
    Record another sighting WITHOUT touching the vector.

    ``set_payload`` rather than ``upsert``: re-sending the vector would mean
    re-embedding text that has not changed, and would overwrite a centroid that
    earlier merges may have already moved.
    """
    from src.vectorstore.client import get_qdrant_client

    get_qdrant_client().set_payload(
        collection_name=ALERT_COLLECTION,
        payload={"occurrence_count": count, "last_seen_at": last_seen_at},
        points=[alert_id],
        wait=True,
    )


def update_centroid(alert_id: str, vector: list[float]) -> bool:
    """
    Move a cluster's vector toward a newly merged member.

    ══ A RUNNING CENTROID, NOT A REPLACEMENT ══
    When a second article about the same event arrives, the cluster's meaning
    is better described by the average of both descriptions than by whichever
    one happened to arrive first. So the stored vector becomes the normalised
    mean, and the cluster drifts toward the event's semantic centre as coverage
    accumulates.

    The alternative — keeping the first vector forever — makes the cluster's
    identity an accident of arrival order, and later coverage that paraphrases
    heavily drifts out of range and fires a duplicate.

    Normalisation is required, not cosmetic: the mean of two unit vectors is
    shorter than either, and Qdrant's Cosine distance normalises at INDEX time
    only. Storing un-normalised means would leave the vector's magnitude
    drifting downward with every merge.

    Returns
    -------
    bool
        False if the point has disappeared (pruned between search and update),
        which is a race to log rather than an error to raise.
    """
    import math

    from qdrant_client.models import PointStruct

    from src.vectorstore.client import get_qdrant_client

    client = get_qdrant_client()
    points = client.retrieve(collection_name=ALERT_COLLECTION, ids=[alert_id], with_vectors=True, with_payload=True)
    if not points:
        logger.warning("Centroid update skipped — point %s no longer exists", alert_id)
        return False

    stored = points[0].vector
    # Qdrant's `vector` field also covers named, sparse, and multi-vector
    # points, none of which this collection uses. Anything but a flat list of
    # numbers means the point was written by something other than
    # upsert_alert, and averaging into it would corrupt whatever it really is.
    existing: list[float] = []
    for value in stored if isinstance(stored, list) else []:
        if not isinstance(value, (int, float)):
            existing = []
            break
        existing.append(float(value))

    if not existing:
        logger.warning("Centroid update skipped — point %s has no plain dense vector", alert_id)
        return False

    mean = [(a + b) / 2.0 for a, b in zip(existing, vector)]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm == 0:
        return False
    centroid = [value / norm for value in mean]

    client.upsert(
        collection_name=ALERT_COLLECTION,
        points=[PointStruct(id=alert_id, vector=centroid, payload=points[0].payload or {})],
        wait=True,
    )
    return True


def prune_expired(*, now: datetime | None = None, retention_seconds: int = ALERT_RETENTION_SECONDS) -> int:
    """
    Delete alert points older than the retention window.

    Only the Qdrant INDEX is pruned. The SQLite record is permanent — the two
    stores have different jobs, and losing the history to keep the index small
    would be the worst of both.

    Retention is twice the longest dedup window, so a point is only ever
    removed well after it could still have matched anything.

    Returns
    -------
    int
        Points deleted.
    """
    from qdrant_client.models import FieldCondition, Filter, Range

    from src.vectorstore.client import get_qdrant_client

    moment = now or datetime.now(timezone.utc)
    cutoff = int(moment.timestamp()) - retention_seconds

    client = get_qdrant_client()
    selector = Filter(must=[FieldCondition(key="fired_at", range=Range(lt=cutoff))])

    before = client.count(collection_name=ALERT_COLLECTION, count_filter=selector, exact=True).count
    if before:
        client.delete(collection_name=ALERT_COLLECTION, points_selector=selector, wait=True)
        logger.info("Pruned %d alert points older than %d days", before, retention_seconds // 86400)

    return int(before)


def alert_stats() -> dict[str, Any]:
    """Snapshot of the alert index, for /health and the cycle report."""
    from src.vectorstore.collections import collection_stats

    return dict(collection_stats(ALERT_COLLECTION))
