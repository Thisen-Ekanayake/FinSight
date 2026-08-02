# ═══════════════════════════════════════════════════════
# FinSight — Tests: Filing Search
# ═══════════════════════════════════════════════════════
#
# Filter construction is tested in isolation. Retrieval quality is tested
# against the real collection and needs `make ingest` to have been run.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from datetime import date

import pytest

from src.vectorstore.search import FilingHit, build_filter, to_citation


def _keys(qdrant_filter) -> list[str]:
    return [c.key for c in qdrant_filter.must]


class TestBuildFilter:
    """
    Filters are HARD constraints. "Apple's supply chain risks" must never
    return an Exxon chunk, however semantically similar — so scoping is a
    payload filter, not a hope about embedding similarity.
    """

    def test_no_constraints_yields_no_filter(self):
        assert build_filter() is None

    def test_single_ticker_uses_exact_match(self):
        from qdrant_client.models import MatchValue

        f = build_filter(ticker="AAPL")
        assert _keys(f) == ["ticker"]
        assert isinstance(f.must[0].match, MatchValue)

    def test_ticker_is_uppercased(self):
        assert build_filter(ticker="aapl").must[0].match.value == "AAPL"

    def test_multiple_tickers_use_match_any(self):
        from qdrant_client.models import MatchAny

        f = build_filter(ticker=["aapl", "msft"])
        assert isinstance(f.must[0].match, MatchAny)
        assert f.must[0].match.any == ["AAPL", "MSFT"]

    def test_item_section_filter(self):
        f = build_filter(item_sections=["1a"])
        assert _keys(f) == ["item_section"]
        assert f.must[0].match.any == ["1A"]

    def test_form_type_filter(self):
        f = build_filter(form_types=["10-k"])
        assert f.must[0].match.any == ["10-K"]

    def test_fiscal_year_filter_keeps_integers(self):
        f = build_filter(fiscal_years=[2024, 2025])
        assert f.must[0].match.any == [2024, 2025]

    def test_date_bounds_become_a_range(self):
        f = build_filter(since=date(2024, 1, 1), until=date(2025, 1, 1))
        assert _keys(f) == ["filing_date"]
        assert f.must[0].range.gte is not None
        assert f.must[0].range.lte is not None

    def test_open_ended_since_leaves_upper_bound_unset(self):
        f = build_filter(since=date(2024, 1, 1))
        assert f.must[0].range.gte is not None
        assert f.must[0].range.lte is None

    def test_constraints_combine_as_must(self):
        f = build_filter(ticker="AAPL", form_types=["10-K"], item_sections=["1A"])
        assert set(_keys(f)) == {"ticker", "form_type", "item_section"}

    def test_every_filtered_field_is_payload_indexed(self):
        """
        Filtering an unindexed field does not error — Qdrant just scans the
        whole collection. This test is the guard against that regression.
        """
        from src.vectorstore.config import FILINGS_PAYLOAD_INDEXES

        f = build_filter(
            ticker="AAPL",
            form_types=["10-K"],
            item_sections=["1A"],
            fiscal_years=[2025],
            since=date(2024, 1, 1),
        )
        for key in _keys(f):
            assert key in FILINGS_PAYLOAD_INDEXES, f"{key} is filtered but not indexed"


class TestToCitation:
    HIT = FilingHit(
        score=0.81,
        text="The Company obtains certain components from single or limited sources." * 12,
        ticker="AAPL",
        accession_no="0000320193-25-000079",
        form_type="10-K",
        filing_date="2025-10-31",
        fiscal_year=2025,
        item_section="1A",
        section_title="Risk Factors",
        chunk_index=12,
        char_start=100,
        char_end=900,
        source_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl-20250927.htm",
    )

    def test_source_id_is_the_accession_number(self):
        assert to_citation(self.HIT)["source_id"] == "0000320193-25-000079"

    def test_source_type_is_edgar(self):
        assert to_citation(self.HIT)["source_type"] == "EDGAR"

    def test_excerpt_is_truncated(self):
        assert len(to_citation(self.HIT, excerpt_chars=50)["excerpt"]) == 50

    def test_excerpt_comes_from_the_real_document_text(self):
        # Never the embed-time contextual header — a quoted citation must match
        # what the filing actually says.
        excerpt = to_citation(self.HIT)["excerpt"]
        assert excerpt in self.HIT["text"]
        assert not excerpt.startswith("AAPL | 10-K")

    def test_as_of_is_the_filing_date(self):
        assert to_citation(self.HIT)["as_of"] == "2025-10-31"


@pytest.mark.integration
@pytest.mark.slow
class TestSearchAgainstRealCollection:
    """Requires ingested filings: `python -m src.vectorstore.ingest --ticker AAPL --form 10-K`."""

    QUERY = "supply chain concentration and single-source component risk"

    def _skip_if_empty(self):
        from src.vectorstore.collections import collection_stats
        from src.vectorstore.config import COLLECTION_FILINGS

        if collection_stats(COLLECTION_FILINGS)["points"] == 0:
            pytest.skip("no filings ingested; run `make ingest` first")

    def test_returns_relevant_hits(self):
        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        hits = search_filings(self.QUERY, ticker="AAPL", limit=5)
        assert hits
        assert hits[0]["score"] > 0.6

    def test_hits_are_ordered_by_score(self):
        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        scores = [h["score"] for h in search_filings(self.QUERY, ticker="AAPL", limit=5)]
        assert scores == sorted(scores, reverse=True)

    def test_ticker_filter_is_absolute(self):
        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        hits = search_filings(self.QUERY, ticker="AAPL", limit=10)
        assert {h["ticker"] for h in hits} == {"AAPL"}

    def test_section_filter_is_absolute(self):
        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        hits = search_filings(self.QUERY, ticker="AAPL", item_sections=["1A"], limit=10)
        assert {h["item_section"] for h in hits} == {"1A"}

    def test_section_filter_changes_the_result_set(self):
        # Demonstrates the filter is doing real work rather than being a no-op.
        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        unfiltered = search_filings("revenue growth drivers", ticker="AAPL", limit=10)
        mdna_only = search_filings("revenue growth drivers", ticker="AAPL", item_sections=["7"], limit=10)

        if {h["item_section"] for h in unfiltered} != {"7"}:
            assert [h["accession_no"] for h in unfiltered] != [h["accession_no"] for h in mdna_only]

    def test_every_hit_carries_a_citable_accession_number(self):
        import re

        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        for hit in search_filings(self.QUERY, ticker="AAPL", limit=5):
            assert re.fullmatch(r"\d{10}-\d{2}-\d{6}", hit["accession_no"])
            assert hit["source_url"].startswith("https://www.sec.gov/")

    def test_impossible_filter_returns_nothing(self):
        from src.vectorstore.search import search_filings

        self._skip_if_empty()
        assert search_filings(self.QUERY, ticker="ZZZZ_NOT_A_TICKER", limit=5) == []
