# ═══════════════════════════════════════════════════════
# FinSight — SEC Filing Chunking
# ═══════════════════════════════════════════════════════
#
# Purpose : Turn a 10-K/10-Q HTML document into embeddable narrative chunks
#           that know which Item section they came from.
#
# Public API:
#   FilingChunk
#   extract_text(path)                  HTML -> clean text
#   find_item_sections(text)            locate Item boundaries, skipping TOC
#   chunk_filing(path, filing)          the full pipeline
#   contextual_header(chunk)            what gets prepended before embedding
#
# ══ NARRATIVE ONLY ══
#   Financial statements are deliberately NOT chunked. Their numbers come from
#   XBRL companyfacts, which is exact and self-citing. Financial tables become
#   unreadable soup under HTML text extraction, and that soup is exactly where
#   hallucinated figures originate. The LLM is never asked to read a number
#   out of one.
#
#   WHICH item that is depends on the FORM:
#       10-K  Item 8  Financial Statements
#       10-Q  Item 1  Financial Statements   (Item 1 is "Business" in a 10-K)
#   So the item map is selected by form type via items_for_form(). Assuming
#   10-K numbering for a 10-Q ingests precisely the table soup this design
#   exists to avoid.
#
# ══ THE TABLE-OF-CONTENTS TRAP ══
#   A 10-K contains every "Item N" heading TWICE: once in the table of
#   contents, once as the real section. Measured on Apple's FY2025 10-K, the
#   TOC packs 23 item markers into ~1,200 characters while the body sections
#   are thousands of characters apart. Naively taking the first match yields
#   23 sections of ~20 characters each and silently ingests nothing.
#
#   find_item_sections resolves it by picking, per item code, the occurrence
#   followed by the MOST text. See that function for why cluster-detection
#   heuristics were tried and abandoned.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

from src.data.schemas import FilingRef
from src.vectorstore.config import (
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
    MIN_CHUNK_CHARS,
    NARRATIVE_ITEMS,
    items_for_form,
)

logger = logging.getLogger(__name__)


class FilingChunk(TypedDict):
    """
    One embeddable slice of a filing's narrative.

    ``text`` is the RAW chunk as it appears in the document — that is what
    gets stored and quoted in citations. The contextual header is applied at
    embed time only, so a citation never quotes text the filing does not
    contain.
    """

    ticker: str
    cik: str
    accession_no: str
    form_type: str
    filing_date: str
    fiscal_year: int
    period_of_report: str | None
    item_section: str
    section_title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    source_url: str


# Matches "Item 1A." / "Item 7 -" / "ITEM 1A:" at the start of a line.
# Anchored to line start so mid-sentence cross-references ("see Item 1A")
# are not mistaken for headings.
ITEM_PATTERN = re.compile(
    r"^[ \t]*Item[ \t]+(\d{1,2}[A-C]?)[ \t]*[.\-–—:]?[ \t]*(.{0,80})", re.MULTILINE | re.IGNORECASE
)


def extract_text(path: Path) -> str:
    """
    Extract readable text from a filing's HTML, preserving paragraph breaks.

    Parameters
    ----------
    path : Path
        Local path to the filing document.

    Returns
    -------
    str
        Normalised text: scripts and styles removed, runs of whitespace
        collapsed, blank lines preserved as paragraph separators.
    """
    import warnings

    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        # Newer EDGAR filings are XHTML. The HTML parser handles them fine and
        # is more tolerant of the malformed markup older filings contain, so
        # the "this looks like XML" advisory is noise here.
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(path.read_text(errors="ignore"), "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text("\n")
    # Collapse horizontal whitespace (including the non-breaking spaces EDGAR
    # documents are full of) but keep line structure for the Item regex.
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()


def find_item_sections(text: str) -> dict[str, tuple[int, int, str]]:
    """
    Locate each Item section's span in the document.

    ── How the table-of-contents trap is avoided ──
    Every "Item N" heading appears at least twice: once in the TOC, once as
    the real section. The rule here is deliberately simple and has no magic
    window constants:

        For each item code, choose the occurrence followed by the MOST text
        before the next heading.

    That directly encodes the thing that actually distinguishes them — a TOC
    entry is followed by the next TOC entry a few characters later, while a
    real section is followed by thousands of characters of content.

    Earlier attempts detected the TOC as a "dense cluster of headings" and cut
    everything before it. Two things break that: Part III items (10-14) are
    also tightly packed, being one-liners incorporated by reference from the
    proxy statement, and the gap between the end of the TOC and the start of
    the body varies enough that a fixed window either overshoots into Item 1
    or stops short. Picking the longest span sidesteps both.

    Parameters
    ----------
    text : str
        Extracted filing text.

    Returns
    -------
    dict
        ``{item_code: (start, end, title)}``. Item codes are uppercased,
        e.g. ``"1A"``.
    """
    matches = list(ITEM_PATTERN.finditer(text))
    if not matches:
        return {}

    # Span from each heading to the next heading of a DIFFERENT item. Using a
    # different item matters: a heading repeated verbatim (headers/footers on
    # each page) would otherwise measure a span of zero against itself.
    spans: list[tuple[str, re.Match[str], int]] = []
    for i, match in enumerate(matches):
        code = match.group(1).upper()
        next_start = len(text)
        for later in matches[i + 1 :]:
            if later.group(1).upper() != code:
                next_start = later.start()
                break
        spans.append((code, match, next_start - match.start()))

    # Winner per item code = the occurrence with the most content after it.
    best: dict[str, tuple[re.Match[str], int]] = {}
    for code, match, length in spans:
        if code not in best or length > best[code][1]:
            best[code] = (match, length)

    ordered = sorted(best.items(), key=lambda kv: kv[1][0].start())
    sections: dict[str, tuple[int, int, str]] = {}

    for position, (code, (match, _)) in enumerate(ordered):
        start = match.start()
        end = ordered[position + 1][1][0].start() if position + 1 < len(ordered) else len(text)
        title = match.group(2).strip(" .:-–—\n") or NARRATIVE_ITEMS.get(code, "")
        sections[code] = (start, end, title)

    logger.debug("Item sections found: %s", sorted(sections))
    return sections


def contextual_header(chunk: FilingChunk) -> str:
    """
    Build the header prepended to a chunk BEFORE embedding.

    A bare chunk loses both the entity and the section it came from — "we face
    intense competition" is nearly meaningless without knowing it is Apple's
    Risk Factors. Prepending this is cheap and a large retrieval-quality win.

    The header is NOT stored in the payload, so citations quote the real
    document text rather than a synthetic preamble.
    """
    fiscal = f"FY{chunk['fiscal_year']}" if chunk["fiscal_year"] else chunk["filing_date"][:4]
    return (
        f"{chunk['ticker']} | {chunk['form_type']} | {fiscal} | "
        f"Item {chunk['item_section']}. {chunk['section_title']}\n\n"
    )


def _split_section(text: str) -> list[tuple[str, int, int]]:
    """Split one section into overlapping chunks, returning (text, start, end)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        add_start_index=True,
    )
    documents = splitter.create_documents([text])

    pieces: list[tuple[str, int, int]] = []
    for document in documents:
        content = document.page_content.strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue  # page furniture, stray headings
        offset = int(document.metadata.get("start_index", 0))
        pieces.append((content, offset, offset + len(document.page_content)))
    return pieces


def chunk_filing(path: Path, filing: FilingRef, *, items: dict[str, str] | None = None) -> list[FilingChunk]:
    """
    Chunk a filing's narrative sections.

    Parameters
    ----------
    path : Path
        Local path to the filing document.
    filing : FilingRef
        Metadata for the filing, providing accession number and dates.
    items : dict, optional
        ``{item_code: section_name}`` to extract. Defaults to the map for the
        filing's FORM TYPE, which matters: Item 1 is "Business" in a 10-K but
        "Financial Statements" in a 10-Q, and financial statements must never
        be chunked.

    Returns
    -------
    list of FilingChunk
        Empty if the document could not be parsed into recognisable sections,
        which is logged rather than raised: a filing that fails to parse
        should be skipped, never ingested as garbage.
    """
    wanted = items if items is not None else items_for_form(filing["form_type"])

    text = extract_text(path)
    if not text:
        logger.warning("%s: no text extracted from %s", filing["accession_no"], path.name)
        return []

    sections = find_item_sections(text)
    if not sections:
        logger.warning("%s: no Item sections found in %s", filing["accession_no"], path.name)
        return []

    fiscal_year = 0
    if filing.get("period_of_report"):
        fiscal_year = int(str(filing["period_of_report"])[:4])
    elif filing.get("filing_date"):
        fiscal_year = int(filing["filing_date"][:4])

    chunks: list[FilingChunk] = []
    index = 0

    for code, default_title in wanted.items():
        span = sections.get(code.upper())
        if span is None:
            continue

        start, end, detected_title = span
        section_text = text[start:end]

        if len(section_text) < MIN_CHUNK_CHARS:
            logger.debug("%s: Item %s is only %d chars, skipping", filing["accession_no"], code, len(section_text))
            continue

        for content, rel_start, rel_end in _split_section(section_text):
            chunks.append(
                FilingChunk(
                    ticker=filing["ticker"],
                    cik=filing["cik"],
                    accession_no=filing["accession_no"],
                    form_type=filing["form_type"],
                    filing_date=filing["filing_date"],
                    fiscal_year=fiscal_year,
                    period_of_report=filing.get("period_of_report"),
                    item_section=code.upper(),
                    section_title=detected_title or default_title,
                    chunk_index=index,
                    text=content,
                    char_start=start + rel_start,
                    char_end=start + rel_end,
                    source_url=filing["url"],
                )
            )
            index += 1

    by_item: dict[str, int] = {}
    for chunk in chunks:
        by_item[chunk["item_section"]] = by_item.get(chunk["item_section"], 0) + 1
    logger.info("%s %s: %d chunks %s", filing["ticker"], filing["form_type"], len(chunks), by_item or "(none)")

    return chunks
