# ═══════════════════════════════════════════════════════
# FinSight — Tests: Filing Ingest
# ═══════════════════════════════════════════════════════
#
# Unit tests are pure. Integration tests use a throwaway collection so they
# never disturb finsight_filings.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import uuid

import pytest

from src.vectorstore.chunking import FilingChunk, contextual_header
from src.vectorstore.ingest import POINT_NAMESPACE, _to_payload, point_id

CHUNK = FilingChunk(
    ticker="AAPL",
    cik="0000320193",
    accession_no="0000320193-25-000079",
    form_type="10-K",
    filing_date="2025-10-31",
    fiscal_year=2025,
    period_of_report="2025-09-27",
    item_section="1A",
    section_title="Risk Factors",
    chunk_index=7,
    text="The Company's business is subject to supply chain concentration risk.",
    char_start=1000,
    char_end=1069,
    source_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl-20250927.htm",
)


class TestPointId:
    """
    Deterministic IDs are what make re-ingest an overwrite instead of a
    duplicate insert. A filing is immutable once accepted, so the same input
    genuinely should produce the same point.
    """

    def test_is_stable_across_calls(self):
        assert point_id("0000320193-25-000079", 7) == point_id("0000320193-25-000079", 7)

    def test_differs_per_chunk_index(self):
        assert point_id("0000320193-25-000079", 7) != point_id("0000320193-25-000079", 8)

    def test_differs_per_filing(self):
        assert point_id("0000320193-25-000079", 7) != point_id("0000320193-24-000123", 7)

    def test_is_a_valid_uuid(self):
        assert uuid.UUID(point_id("0000320193-25-000079", 0)).version == 5

    def test_matches_an_independently_computed_uuid5(self):
        # Pin the exact derivation: changing it would orphan every existing
        # point and silently double the collection on the next ingest.
        expected = str(uuid.uuid5(POINT_NAMESPACE, "0000320193-25-000079:7"))
        assert point_id("0000320193-25-000079", 7) == expected

    def test_namespace_is_fixed_not_random(self):
        # Must be stable across runs, machines, and re-clones.
        assert POINT_NAMESPACE == uuid.NAMESPACE_URL


class TestPayload:
    def test_stores_raw_text_not_the_contextual_header(self):
        """
        The header is an embed-time device. Storing it would make citations
        quote a synthetic preamble the filing does not contain.
        """
        payload = _to_payload(CHUNK)
        assert payload["text"] == CHUNK["text"]
        assert not payload["text"].startswith(contextual_header(CHUNK))

    def test_filing_date_is_an_integer_timestamp(self):
        # Qdrant Range filters need a numeric field; ISO strings cannot be
        # range-filtered, so "filings since X" would silently fail.
        payload = _to_payload(CHUNK)
        assert isinstance(payload["filing_date"], int)

    def test_keeps_the_iso_date_for_display(self):
        assert _to_payload(CHUNK)["filing_date_iso"] == "2025-10-31"

    def test_timestamp_ordering_matches_date_ordering(self):
        older = dict(CHUNK, filing_date="2024-11-01")
        assert _to_payload(older)["filing_date"] < _to_payload(CHUNK)["filing_date"]  # type: ignore[arg-type]

    def test_carries_every_indexed_field(self):
        from src.vectorstore.config import FILINGS_PAYLOAD_INDEXES

        payload = _to_payload(CHUNK)
        for field in FILINGS_PAYLOAD_INDEXES:
            assert field in payload, f"{field} is indexed but absent from the payload"

    def test_carries_the_citation_identifiers(self):
        payload = _to_payload(CHUNK)
        assert payload["accession_no"] == CHUNK["accession_no"]
        assert payload["source_url"].startswith("https://www.sec.gov/")

    def test_carries_offsets_for_excerpt_highlighting(self):
        payload = _to_payload(CHUNK)
        assert payload["char_start"] < payload["char_end"]


@pytest.mark.integration
@pytest.mark.slow
class TestIngestAgainstQdrant:
    """Real Qdrant, throwaway collection."""

    def setup_method(self):
        import src.vectorstore.ingest as ing
        from src.vectorstore.collections import ensure_collection
        from src.vectorstore.config import FILINGS_PAYLOAD_INDEXES

        self.collection = f"finsight_ingest_test_{uuid.uuid4().hex[:8]}"
        self._original = ing.COLLECTION_FILINGS
        ing.COLLECTION_FILINGS = self.collection
        ensure_collection(self.collection, FILINGS_PAYLOAD_INDEXES)

    def teardown_method(self):
        import src.vectorstore.ingest as ing
        from src.vectorstore.collections import drop_collection

        ing.COLLECTION_FILINGS = self._original
        drop_collection(self.collection)

    def _count(self) -> int:
        from src.vectorstore.client import get_qdrant_client

        return get_qdrant_client().get_collection(self.collection).points_count or 0

    def test_ingest_then_reingest_does_not_duplicate(self):
        """
        THE IDEMPOTENCY GUARANTEE. Without deterministic point IDs the
        collection silently doubles on every re-run of run_ingest.sh.
        """
        from src.data.edgar import get_filing_index
        from src.vectorstore.ingest import ingest_filing

        filing = get_filing_index("AAPL", forms=["10-K"], limit=1)[0]

        first = ingest_filing(filing)
        assert first["chunks"] > 0
        count_after_first = self._count()

        second = ingest_filing(filing)
        assert second["chunks"] == first["chunks"]
        assert self._count() == count_after_first

    def test_skip_existing_avoids_rework(self):
        from src.data.edgar import get_filing_index
        from src.vectorstore.ingest import ingest_filing

        filing = get_filing_index("AAPL", forms=["10-K"], limit=1)[0]
        ingest_filing(filing)

        report = ingest_filing(filing, skip_existing=True)
        assert report["skipped"] is True
        assert report["reason"] == "already ingested"

    def test_stored_payload_round_trips(self):
        from src.data.edgar import get_filing_index
        from src.vectorstore.client import get_qdrant_client
        from src.vectorstore.ingest import ingest_filing

        filing = get_filing_index("AAPL", forms=["10-K"], limit=1)[0]
        ingest_filing(filing)

        points, _ = get_qdrant_client().scroll(self.collection, limit=1, with_payload=True)
        payload = points[0].payload or {}
        assert payload["ticker"] == "AAPL"
        assert payload["accession_no"] == filing["accession_no"]
        assert payload["item_section"] in {"1", "1A", "3", "7", "7A"}
        assert len(payload["text"]) > 0
