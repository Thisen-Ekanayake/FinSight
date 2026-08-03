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

# Ablation twin of COLLECTION_FILINGS: identical filings, identical chunking,
# identical payloads — embedded WITHOUT the contextual header.
#
# It has to be a separate collection rather than a flag on the same one. The
# header changes the vector, not the payload, and a collection cannot hold two
# vectors per point under one name. Comparing header on/off therefore means
# comparing two indexes over the same corpus, which is also the only way to run
# the A/B without destroying the production index.
#
# Not created by ensure_collections(); it exists only when the ablation is
# being run. See evals/variants.py "no-headers".
COLLECTION_FILINGS_ABLATION: str = "finsight_filings_noheader"

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

# ══ ITEM NUMBERING DIFFERS BY FORM ══
# The same "Item N" means different things in a 10-K and a 10-Q, so the map of
# narrative sections MUST be selected by form type. Getting this wrong is not
# a cosmetic error: in a 10-Q, Item 1 is FINANCIAL STATEMENTS, so treating it
# as "Business" ingests exactly the mangled table soup this design exists to
# avoid.
#
# In both maps, financial statements are deliberately absent — 10-K Item 8 and
# 10-Q Item 1. Their numbers come from XBRL companyfacts, which is exact and
# self-citing.
NARRATIVE_ITEMS_10K: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}

# 10-Q layout:
#   Part I   Item 1  Financial Statements          <- EXCLUDED (XBRL covers it)
#            Item 2  MD&A                          <- included
#            Item 3  Market Risk                   <- included
#            Item 4  Controls and Procedures       <- excluded (boilerplate)
#   Part II  Item 1  Legal Proceedings
#            Item 1A Risk Factors                  <- included
#
# Note Part I and Part II reuse numbers: "Item 1" is Financial Statements in
# Part I and Legal Proceedings in Part II, "Item 3" is Market Risk then
# Defaults. find_item_sections resolves each code to whichever occurrence has
# the most content, which lands on the substantive Part I sections.
NARRATIVE_ITEMS_10Q: dict[str, str] = {
    "1A": "Risk Factors",
    "2": "Management's Discussion and Analysis",
    "3": "Quantitative and Qualitative Disclosures About Market Risk",
}

# Default for callers that do not specify a form. Kept as the 10-K map for
# backwards compatibility; chunk_filing selects properly via items_for_form.
NARRATIVE_ITEMS: dict[str, str] = NARRATIVE_ITEMS_10K

NARRATIVE_ITEMS_BY_FORM: dict[str, dict[str, str]] = {
    "10-K": NARRATIVE_ITEMS_10K,
    "10-K/A": NARRATIVE_ITEMS_10K,
    "20-F": NARRATIVE_ITEMS_10K,
    "10-Q": NARRATIVE_ITEMS_10Q,
    "10-Q/A": NARRATIVE_ITEMS_10Q,
}


def items_for_form(form_type: str) -> dict[str, str]:
    """
    Return the narrative item map for a form type.

    Parameters
    ----------
    form_type : str
        e.g. ``"10-K"``, ``"10-Q"``.

    Returns
    -------
    dict
        ``{item_code: section_name}``. Unknown forms fall back to the 10-K
        map, which is the more common shape.
    """
    return NARRATIVE_ITEMS_BY_FORM.get(form_type.upper(), NARRATIVE_ITEMS_10K)


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

# ══ THE HIGH-SEVERITY GUARDRAIL IS NOT A SIMILARITY FLOOR ══
#   The plan specified one: "never suppress a HIGH alert below 0.96 similarity,
#   whatever the thresholds say". It was a prior, not a measurement, and the
#   first live run of the semantic path refuted it.
#
#   Three outlets covering one DOJ probe of Apple, canonicalized and embedded
#   for real, scored 0.898 and 0.913 against the first report. Both are
#   comfortably above TAU_HIGH and both sit far below 0.96 — so a 0.96 floor
#   does not protect against a missed event, it guarantees ONE PAGE PER OUTLET
#   for every HIGH-severity story. That is precisely the noise the engine
#   exists to remove, and an alert stream nobody reads has a false-negative
#   rate of 100%.
#
#   Any floor high enough to be a meaningful extra margin also sits above the
#   band where genuine paraphrases live, so the whole idea is unworkable here.
#
#   What replaced it is a rule about INFORMATION rather than similarity:
#
#       A HIGH candidate fires unless the matched parent already represents a
#       reported event of HIGH severity.
#
#   The guarantee the guardrail was reaching for is "the reader learns about
#   this HIGH event". If the parent was HIGH and fired, they already did. If
#   the parent was MED or LOW, they were told a milder version and the
#   escalation must reach them — which it does, either as an ESCALATE in the
#   merge band or as a forced FIRE in the suppress band.
#
#   See src/monitor/dedup.py and docs/dedup_algorithm.md.

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
