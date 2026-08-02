# ═══════════════════════════════════════════════════════
# FinSight — Vector Store Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : Collection names, vector params, chunking sizes, and the dedup
#           similarity thresholds. The single tuning surface for retrieval
#           and deduplication behaviour.
#
# Public API:
#   COLLECTION_FILINGS, COLLECTION_ALERTS
#   FILINGS_PAYLOAD_INDEXES, ALERTS_PAYLOAD_INDEXES
#   CHUNK_SIZE, CHUNK_OVERLAP, NARRATIVE_ITEMS
#   TAU_HIGH, TAU_LOW, DEDUP_WINDOW_SECONDS
#   FORBIDDEN_COLLECTIONS
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from src.core.config import EMBEDDING_DIM

# ── Isolation guard ─────────────────────────────────────
# Another project on this machine runs its own Qdrant on :6333 with these
# collections. If the client ever sees them, QDRANT_URL is pointing at the
# wrong instance and we must refuse to write. See src/vectorstore/client.py.
FORBIDDEN_COLLECTIONS: frozenset[str] = frozenset({"athena_content", "image_embeddings"})

# ── Collections ─────────────────────────────────────────
COLLECTION_FILINGS: str = "finsight_filings"
COLLECTION_ALERTS: str = "finsight_alerts"

VECTOR_SIZE: int = EMBEDDING_DIM
DISTANCE: str = "Cosine"

# HNSW defaults are correct at this scale (~40k chunks). Raising m/ef_construct
# would buy recall we do not need and cost index build time we would notice.
HNSW_M: int = 16
HNSW_EF_CONSTRUCT: int = 100

# ~40k chunks x 384 dims x 4 bytes is roughly 60MB — comfortably in RAM.
ON_DISK_VECTORS: bool = False

# Payload fields that get an explicit index. Without these, every filtered
# search degrades to a full scan — and EVERY research query filters on ticker.
# This is the Qdrant lesson most tutorials skip.
FILINGS_PAYLOAD_INDEXES: dict[str, str] = {
    "ticker": "keyword",
    "cik": "keyword",
    "accession_no": "keyword",
    "form_type": "keyword",
    "filing_date": "integer",
    "fiscal_year": "integer",
    "item_section": "keyword",
}

ALERTS_PAYLOAD_INDEXES: dict[str, str] = {
    "alert_id": "keyword",
    "ticker": "keyword",
    "alert_type": "keyword",
    "severity": "keyword",
    "status": "keyword",
    "fired_at": "integer",
    "dedup_key": "keyword",
}

# ── Chunking (10-K / 10-Q narrative) ────────────────────
# 1200 chars is roughly 300 tokens — comfortably inside bge-small's 512-token
# window, with room for the contextual header prepended at embed time.
CHUNK_SIZE: int = 1200
CHUNK_OVERLAP: int = 200
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]
MIN_CHUNK_CHARS: int = 200

# Narrative items worth embedding. Item 8 (financial statements) is
# deliberately absent: its numbers come from XBRL companyfacts, which is exact
# and self-citing. Asking an LLM to read a figure out of mangled table HTML is
# precisely where hallucinated numbers come from.
NARRATIVE_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}

# bge models are ASYMMETRIC: this prefix goes on QUERIES ONLY, never on stored
# documents. Alert-vs-alert dedup is symmetric and uses no prefix at all.
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

DEFAULT_SEARCH_LIMIT: int = 8

# ── Dedup thresholds ────────────────────────────────────
# ⚠️  MEASURED PRIORS, not final tuned values. Calibrate properly in Phase 7
#     via evals/sweep_thresholds.py against ~60 hand-labelled REAL pairs.
#
# Measured 2026-08-02 on BAAI/bge-small-en-v1.5, over hand-written canonical
# alert text in the exact format dedup.py produces:
#
#     band                          min     mean    max
#     DUPLICATE (same event)        0.895   0.914   0.929
#     RELATED   (event + new info)  0.801   0.844   0.887
#     DISTINCT  (different event)   0.728   0.735   0.742
#
# THE TRAP, and why these numbers are not the tutorial ones:
#   The negatives that matter are NOT random sentences. The Qdrant payload
#   filter already constrains candidates to the same ticker AND alert type,
#   so the hard case is two genuinely different events sharing both — and
#   those still score ~0.73. A 0.7 threshold would suppress real events.
#
# Chosen to sit between the observed bands, biased for SUPPRESSION PRECISION
# rather than max F1, because the cost is asymmetric: a false suppress is a
# missed real event, a false fire is mild annoyance.
#   TAU_HIGH  above the RELATED ceiling (0.887), below the DUPLICATE floor (0.895)
#   TAU_LOW   above the DISTINCT ceiling (0.742), below the RELATED floor (0.801)
#
# Note this corrects an earlier guess of 0.92/0.82: at 0.92, real duplicates
# measured at 0.895 and 0.918 would have slipped through and fired twice.
#
# RE-TUNE FROM SCRATCH if EMBEDDING_MODEL changes — these numbers are
# meaningless against different vectors.
TAU_HIGH: float = 0.89  # >= this  -> SUPPRESS as duplicate
TAU_LOW: float = 0.78  # in [LOW, HIGH) -> MERGE as same event, new information

# Never suppress a HIGH-severity alert below this similarity, whatever the
# thresholds say. Missing a real HIGH event costs far more than a duplicate ping.
TAU_HIGH_SEVERITY_FORCE_FIRE: float = 0.96

# The right dedup horizon is a property of the underlying event's natural
# frequency, not a global constant.
DEDUP_WINDOW_SECONDS: dict[str, int] = {
    "PRICE_MOVE": 24 * 3600,  # a 5% drop tomorrow IS a new event
    "NEWS_SENTIMENT": 72 * 3600,  # a story gets recycled for ~3 days
    "NEW_FILING": 30 * 86400,  # a 10-K is filed once; a long window is safe
    "MACRO_EVENT": 7 * 86400,  # CPI is monthly; a week covers the news cycle
}

# Prune alert points older than twice the longest window so the collection
# never grows unbounded and stale vectors do not skew search.
ALERT_RETENTION_SECONDS: int = max(DEDUP_WINDOW_SECONDS.values()) * 2

DEDUP_SEARCH_LIMIT: int = 5
