# ═══════════════════════════════════════════════════════
# FinSight — Tests: Qdrant Collection Management
# ═══════════════════════════════════════════════════════
#
# Unit tests mock the client. Integration tests run against the real
# container on :6335 using throwaway collection names, so they never touch
# finsight_filings or finsight_alerts.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import uuid

import pytest

from src.vectorstore import collections as coll
from src.vectorstore.config import (
    ALERTS_PAYLOAD_INDEXES,
    COLLECTION_ALERTS,
    COLLECTION_FILINGS,
    FILINGS_PAYLOAD_INDEXES,
    VECTOR_SIZE,
)


class TestIndexConfiguration:
    """
    Every field a query filters on must be indexed. An unindexed filter does
    not error — Qdrant just scans the whole collection, which passes tests on
    small data and degrades silently as it grows.
    """

    def test_filings_index_the_fields_every_query_filters_on(self):
        # Research queries are essentially always scoped to a ticker.
        assert "ticker" in FILINGS_PAYLOAD_INDEXES
        assert "form_type" in FILINGS_PAYLOAD_INDEXES
        assert "item_section" in FILINGS_PAYLOAD_INDEXES

    def test_filings_index_filing_date_as_a_range_filterable_type(self):
        assert FILINGS_PAYLOAD_INDEXES["filing_date"] == "integer"

    def test_alerts_index_the_full_dedup_filter(self):
        # dedup.py filters on ticker + alert_type + status + a fired_at range.
        for field in ("ticker", "alert_type", "status", "fired_at"):
            assert field in ALERTS_PAYLOAD_INDEXES, f"{field} must be indexed for dedup"

    def test_alerts_index_the_exact_match_fast_path(self):
        # dedup_key is the ~90% no-embedding, no-LLM path.
        assert ALERTS_PAYLOAD_INDEXES["dedup_key"] == "keyword"

    def test_fired_at_is_range_filterable(self):
        # Time-window dedup needs Range, which requires an integer index.
        assert ALERTS_PAYLOAD_INDEXES["fired_at"] == "integer"

    def test_index_types_are_all_supported(self):
        supported = {"keyword", "integer", "float", "bool", "text"}
        for mapping in (FILINGS_PAYLOAD_INDEXES, ALERTS_PAYLOAD_INDEXES):
            assert set(mapping.values()) <= supported


class TestSchemaTypeMapping:
    def test_maps_every_configured_type(self):
        for kind in {"keyword", "integer", "float", "bool", "text"}:
            assert coll._schema_type(kind) is not None

    def test_unknown_type_raises(self):
        with pytest.raises(KeyError):
            coll._schema_type("nonexistent")


@pytest.mark.integration
class TestAgainstRealQdrant:
    """Throwaway collections on the real container — never the production ones."""

    def setup_method(self):
        self.name = f"finsight_test_{uuid.uuid4().hex[:8]}"

    def teardown_method(self):
        coll.drop_collection(self.name)

    def test_creates_a_collection(self):
        assert coll.ensure_collection(self.name, {"ticker": "keyword"}) is True
        assert coll.collection_stats(self.name)["exists"] is True

    def test_is_idempotent(self):
        coll.ensure_collection(self.name, {"ticker": "keyword"})
        # Second call must report "already existed", not recreate and wipe.
        assert coll.ensure_collection(self.name, {"ticker": "keyword"}) is False

    def test_creates_the_requested_payload_indexes(self):
        coll.ensure_collection(self.name, {"ticker": "keyword", "filing_date": "integer"})
        indexed = coll.collection_stats(self.name)["indexed_fields"]
        assert "ticker" in indexed
        assert "filing_date" in indexed

    def test_adds_indexes_introduced_after_creation(self):
        # An index added to config later must be picked up without a full
        # re-ingest of the collection.
        coll.ensure_collection(self.name, {"ticker": "keyword"})
        coll.ensure_collection(self.name, {"ticker": "keyword", "form_type": "keyword"})
        assert "form_type" in coll.collection_stats(self.name)["indexed_fields"]

    def test_vector_params_match_the_embedding_model(self):
        coll.ensure_collection(self.name, {})
        from src.vectorstore.client import get_qdrant_client

        params = get_qdrant_client().get_collection(self.name).config.params.vectors
        assert params.size == VECTOR_SIZE
        assert params.distance.value.lower() == "cosine"

    def test_stats_for_a_missing_collection_do_not_raise(self):
        stats = coll.collection_stats("finsight_definitely_not_here")
        assert stats["exists"] is False
        assert stats["status"] == "missing"

    def test_drop_reports_whether_it_existed(self):
        coll.ensure_collection(self.name, {})
        assert coll.drop_collection(self.name) is True
        assert coll.drop_collection(self.name) is False


@pytest.mark.integration
class TestProductionCollections:
    """The real collections, as created by `make qdrant` + --ensure."""

    def test_both_exist_with_their_indexes(self):
        coll.ensure_collections()

        filings = coll.collection_stats(COLLECTION_FILINGS)
        assert filings["exists"]
        assert set(FILINGS_PAYLOAD_INDEXES) <= set(filings["indexed_fields"])

        alerts = coll.collection_stats(COLLECTION_ALERTS)
        assert alerts["exists"]
        assert set(ALERTS_PAYLOAD_INDEXES) <= set(alerts["indexed_fields"])
