# ═══════════════════════════════════════════════════════
# FinSight — Tests: SEC Filing Chunking
# ═══════════════════════════════════════════════════════
#
# Synthetic HTML exercises the parsing edge cases precisely. One integration
# test runs against a real cached 10-K.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schemas import FilingRef
from src.vectorstore.chunking import (
    chunk_filing,
    contextual_header,
    extract_text,
    find_item_sections,
)
from src.vectorstore.config import MIN_CHUNK_CHARS, NARRATIVE_ITEMS

FILING = FilingRef(
    ticker="TEST",
    cik="0000000001",
    accession_no="0000000001-25-000001",
    form_type="10-K",
    filing_date="2025-11-01",
    period_of_report="2025-09-27",
    primary_document="test-10k.htm",
    url="https://www.sec.gov/Archives/edgar/data/1/000000000125000001/test-10k.htm",
    items=[],
)


def _body(item: str, words: int = 400) -> str:
    """A section body long enough to survive the minimum-chunk filter."""
    return " ".join(f"{item}-content-{i}" for i in range(words))


def _quarterly(**overrides) -> FilingRef:
    """A 10-Q FilingRef, where Item numbering differs from a 10-K."""
    return FilingRef(**{**FILING, "form_type": "10-Q", **overrides})  # type: ignore[typeddict-item]


def _make_html(*, with_toc: bool = True, with_part_iii_cluster: bool = True) -> str:
    """Build a 10-K-shaped document with the structures that trip naive parsers."""
    parts = ["<html><body>"]

    if with_toc:
        # The trap: every heading appears here first, tightly packed.
        parts.append("<p>TABLE OF CONTENTS</p>")
        for code, title in [
            ("1", "Business"),
            ("1A", "Risk Factors"),
            ("1B", "Unresolved Staff Comments"),
            ("2", "Properties"),
            ("3", "Legal Proceedings"),
            ("7", "Management's Discussion and Analysis"),
            ("7A", "Quantitative and Qualitative Disclosures"),
            ("8", "Financial Statements"),
        ]:
            parts.append(f"<p>Item {code}. {title}</p>")

    for code, title in [
        ("1", "Business"),
        ("1A", "Risk Factors"),
        ("3", "Legal Proceedings"),
        ("7", "Management's Discussion and Analysis"),
        ("7A", "Quantitative and Qualitative Disclosures"),
        ("8", "Financial Statements"),
    ]:
        parts.append(f"<p>Item {code}. {title}</p>")
        parts.append(f"<p>{_body(code)}</p>")

    if with_part_iii_cluster:
        # Part III items are each a couple of lines incorporated by reference,
        # so they form a SECOND dense cluster near the end of the document.
        for code, title in [
            ("10", "Directors"),
            ("11", "Executive Compensation"),
            ("12", "Security Ownership"),
            ("13", "Related Transactions"),
            ("14", "Accountant Fees"),
            ("15", "Exhibits"),
        ]:
            parts.append(f"<p>Item {code}. {title}</p><p>Incorporated by reference.</p>")

    parts.append("</body></html>")
    return "\n".join(parts)


@pytest.fixture
def filing_path(tmp_path: Path) -> Path:
    path = tmp_path / "test-10k.htm"
    path.write_text(_make_html())
    return path


class TestExtractText:
    def test_strips_markup(self, filing_path):
        text = extract_text(filing_path)
        assert "<p>" not in text
        assert "Business" in text

    def test_drops_scripts_and_styles(self, tmp_path):
        path = tmp_path / "f.htm"
        path.write_text("<html><body><script>evil()</script><style>a{}</style><p>Real text</p></body></html>")
        text = extract_text(path)
        assert "evil" not in text
        assert "a{}" not in text
        assert "Real text" in text

    def test_collapses_nbsp_and_runs_of_space(self, tmp_path):
        path = tmp_path / "f.htm"
        path.write_text("<html><body><p>a&nbsp;&nbsp;&nbsp;b     c</p></body></html>")
        assert "a b c" in extract_text(path)

    def test_preserves_line_structure_for_the_item_regex(self, filing_path):
        # Item headings must remain at line starts or detection fails.
        assert "\n" in extract_text(filing_path)


class TestTableOfContentsTrap:
    """
    A 10-K contains every Item heading twice. Measured on Apple's FY2025 10-K,
    the TOC packs 23 markers into ~1,200 chars while body sections are
    thousands apart. Taking the first match yields 23 useless ~20-char
    sections and silently ingests nothing.
    """

    def test_sections_are_bodies_not_toc_entries(self, filing_path):
        sections = find_item_sections(extract_text(filing_path))
        start, end, _ = sections["1A"]
        assert end - start > MIN_CHUNK_CHARS

    def test_every_narrative_item_is_found(self, filing_path):
        sections = find_item_sections(extract_text(filing_path))
        for code in NARRATIVE_ITEMS:
            assert code in sections, f"Item {code} not detected"

    def test_part_iii_cluster_does_not_swallow_the_body(self, filing_path):
        """
        Regression guard for a real bug. Part III items are also tightly
        packed, so scanning for the LAST or densest cluster lands on those and
        discards the entire document body. Only the FIRST cluster is the TOC.
        """
        sections = find_item_sections(extract_text(filing_path))
        assert len(sections) > 3
        start, end, _ = sections["7"]
        assert end - start > MIN_CHUNK_CHARS

    def test_document_without_a_toc_still_parses(self, tmp_path):
        path = tmp_path / "10q.htm"
        path.write_text(_make_html(with_toc=False, with_part_iii_cluster=False))
        sections = find_item_sections(extract_text(path))
        assert "1A" in sections
        assert sections["1A"][1] - sections["1A"][0] > MIN_CHUNK_CHARS

    def test_no_items_yields_empty(self, tmp_path):
        path = tmp_path / "f.htm"
        path.write_text("<html><body><p>No item headings anywhere in this document.</p></body></html>")
        assert find_item_sections(extract_text(path)) == {}

    def test_inline_cross_references_are_not_treated_as_headings(self, tmp_path):
        # "see Item 1A" mid-sentence must not start a section.
        path = tmp_path / "f.htm"
        path.write_text(
            "<html><body><p>Item 1. Business</p>"
            f"<p>Please see Item 1A for more detail. {_body('1')}</p></body></html>"
        )
        sections = find_item_sections(extract_text(path))
        assert "1A" not in sections


class TestChunkFiling:
    def test_produces_chunks(self, filing_path):
        assert chunk_filing(filing_path, FILING)

    def test_item_8_is_excluded_by_design(self, filing_path):
        """
        Numbers come from XBRL, narrative from RAG. Financial statements become
        soup under HTML extraction and that soup is where hallucinated figures
        originate, so Item 8 is never chunked.
        """
        chunks = chunk_filing(filing_path, FILING)
        assert "8" not in {c["item_section"] for c in chunks}

    def test_only_narrative_items_are_chunked(self, filing_path):
        chunks = chunk_filing(filing_path, FILING)
        assert {c["item_section"] for c in chunks} <= set(NARRATIVE_ITEMS)

    def test_chunk_indexes_are_unique_and_sequential(self, filing_path):
        chunks = chunk_filing(filing_path, FILING)
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_metadata_is_propagated_from_the_filing(self, filing_path):
        chunk = chunk_filing(filing_path, FILING)[0]
        assert chunk["accession_no"] == FILING["accession_no"]
        assert chunk["ticker"] == "TEST"
        assert chunk["form_type"] == "10-K"
        assert chunk["source_url"] == FILING["url"]

    def test_fiscal_year_comes_from_the_report_period(self, filing_path):
        assert chunk_filing(filing_path, FILING)[0]["fiscal_year"] == 2025

    def test_short_chunks_are_dropped(self, filing_path):
        chunks = chunk_filing(filing_path, FILING)
        assert all(len(c["text"]) >= MIN_CHUNK_CHARS for c in chunks)

    def test_char_offsets_are_ordered(self, filing_path):
        for chunk in chunk_filing(filing_path, FILING):
            assert chunk["char_start"] < chunk["char_end"]

    def test_unparseable_document_returns_empty_rather_than_raising(self, tmp_path):
        # A filing that fails to parse must be skipped, never ingested as garbage.
        path = tmp_path / "junk.htm"
        path.write_text("<html><body><p>nothing structured here</p></body></html>")
        assert chunk_filing(path, FILING) == []

    def test_explicit_item_selection_is_honoured(self, filing_path):
        chunks = chunk_filing(filing_path, FILING, items={"1A": "Risk Factors"})
        assert {c["item_section"] for c in chunks} == {"1A"}


class TestFormSpecificItemNumbering:
    """
    REGRESSION GUARD for a real bug.

    "Item N" means different things per form. In a 10-Q, Item 1 is FINANCIAL
    STATEMENTS, not Business. Applying the 10-K map to a 10-Q ingested 589
    chunks of raw table soup — exactly what this design exists to prevent.
    """

    def _quarterly_html(self) -> str:
        return "\n".join(
            [
                "<html><body>",
                "<p>Item 1. Financial Statements</p>",
                f"<p>$ ( 13 ) Investments Balance beginning of period $ ( 1,051 ) {_body('fin', 500)}</p>",
                "<p>Item 2. Management's Discussion and Analysis</p>",
                f"<p>{_body('mdna')}</p>",
                "<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk</p>",
                f"<p>{_body('risk')}</p>",
                "<p>Item 1A. Risk Factors</p>",
                f"<p>{_body('rf')}</p>",
                "</body></html>",
            ]
        )

    @pytest.fixture
    def quarterly_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "test-10q.htm"
        path.write_text(self._quarterly_html())
        return path

    def test_10q_item_1_financial_statements_is_excluded(self, quarterly_path):
        chunks = chunk_filing(quarterly_path, _quarterly())
        assert "1" not in {c["item_section"] for c in chunks}, "10-Q Item 1 is Financial Statements — never chunk it"

    def test_10q_captures_mdna_as_item_2(self, quarterly_path):
        chunks = chunk_filing(quarterly_path, _quarterly())
        assert "2" in {c["item_section"] for c in chunks}, "10-Q MD&A is Item 2, not Item 7"

    def test_10q_captures_risk_factors(self, quarterly_path):
        chunks = chunk_filing(quarterly_path, _quarterly())
        assert "1A" in {c["item_section"] for c in chunks}

    def test_10q_never_ingests_the_table_soup(self, quarterly_path):
        chunks = chunk_filing(quarterly_path, _quarterly())
        assert not any("Investments Balance beginning of period" in c["text"] for c in chunks)

    def test_10k_still_captures_item_1_as_business(self, filing_path):
        chunks = chunk_filing(filing_path, FILING)
        assert "1" in {c["item_section"] for c in chunks}, "10-K Item 1 IS Business and should be chunked"

    def test_item_maps_differ_between_forms(self):
        from src.vectorstore.config import items_for_form

        assert items_for_form("10-K") != items_for_form("10-Q")

    def test_neither_map_includes_its_financial_statements_item(self):
        from src.vectorstore.config import items_for_form

        assert "8" not in items_for_form("10-K")
        assert "1" not in items_for_form("10-Q")

    def test_unknown_form_falls_back_to_the_10k_map(self):
        from src.vectorstore.config import NARRATIVE_ITEMS_10K, items_for_form

        assert items_for_form("S-1") == NARRATIVE_ITEMS_10K

    def test_form_lookup_is_case_insensitive(self):
        from src.vectorstore.config import items_for_form

        assert items_for_form("10-q") == items_for_form("10-Q")


class TestContextualHeader:
    """Applied at embed time only — never stored, so citations quote the real text."""

    def test_carries_entity_form_year_and_section(self, filing_path):
        chunk = chunk_filing(filing_path, FILING)[0]
        header = contextual_header(chunk)
        assert "TEST" in header
        assert "10-K" in header
        assert "FY2025" in header
        assert f"Item {chunk['item_section']}" in header

    def test_is_not_part_of_the_stored_text(self, filing_path):
        chunk = chunk_filing(filing_path, FILING)[0]
        assert not chunk["text"].startswith(contextual_header(chunk))


@pytest.mark.integration
class TestRealFiling:
    """Against a real Apple 10-K downloaded from EDGAR."""

    def test_chunks_a_real_10k(self):
        from src.data.edgar import download_filing_document, get_filing_index

        filing = get_filing_index("AAPL", forms=["10-K"], limit=1)[0]
        chunks = chunk_filing(download_filing_document(filing), filing)

        assert len(chunks) > 50
        sections = {c["item_section"] for c in chunks}
        assert "1A" in sections, "Risk Factors is the highest-value narrative section"
        assert "7" in sections, "MD&A must be captured"
        assert "8" not in sections, "financial statements must never be chunked"

    def test_risk_factors_is_the_largest_section(self):
        from src.data.edgar import download_filing_document, get_filing_index

        filing = get_filing_index("AAPL", forms=["10-K"], limit=1)[0]
        chunks = chunk_filing(download_filing_document(filing), filing)

        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk["item_section"]] = counts.get(chunk["item_section"], 0) + 1
        assert max(counts, key=lambda k: counts[k]) == "1A"
