# ═══════════════════════════════════════════════════════
# FinSight — Filings RAG Specialist
# ═══════════════════════════════════════════════════════
#
# Purpose : Narrative disclosure from 10-K/10-Q text — what management
#           actually said about risks, strategy, competition, and results.
#
# Public API:
#   filings_rag_node(payload)
#   select_sections(sub_question)
#
# Complements the fundamentals agent rather than overlapping it: numbers come
# from XBRL, narrative comes from here. Each hit carries the accession number
# of the filing it was retrieved from, plus an excerpt of the real document
# text — so a qualitative claim is quotable, not just assertable.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import time

from src.research.agents._common import finding, specialist, tool_record
from src.research.config import FILINGS_TOP_K
from src.vectorstore.config import COLLECTION_FILINGS

logger = logging.getLogger(__name__)

AGENT = "filings_rag"

# Which index this specialist reads. A module-level name rather than the
# search default, so the contextual-header ablation can point it at the twin
# collection for one experiment without editing any call site.
SEARCH_COLLECTION: str = COLLECTION_FILINGS

# Keyword -> Item sections. Narrowing the search to the right section is a
# large precision win: "what risks?" should not surface the business
# description, however similar it looks in vector space.
#
# Both 10-K and 10-Q codes are included because the collection holds both, and
# their numbering differs (MD&A is Item 7 in a 10-K, Item 2 in a 10-Q).
_SECTION_KEYWORDS: list[tuple[tuple[str, ...], list[str]]] = [
    (("risk", "threat", "headwind", "uncertaint", "exposure", "vulnerab"), ["1A"]),
    (("legal", "lawsuit", "litigation", "regulatory action", "proceeding"), ["3"]),
    (("md&a", "management discussion", "outlook", "guidance", "performance", "results"), ["7", "2"]),
    (("business", "strategy", "product", "segment", "compet", "market share"), ["1"]),
    (("market risk", "interest rate risk", "currency", "hedg"), ["7A", "3"]),
]


def select_sections(sub_question: str) -> list[str] | None:
    """
    Choose which Item sections to search.

    Parameters
    ----------
    sub_question : str
        The focused question routed to this specialist.

    Returns
    -------
    list of str or None
        Item codes, or None to search every ingested section when the
        question does not clearly map to one.
    """
    lowered = sub_question.lower()
    selected: list[str] = []

    for keywords, sections in _SECTION_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            selected.extend(s for s in sections if s not in selected)

    return selected or None


@specialist(AGENT)
def filings_rag_node(payload: dict, ticker: str) -> tuple[list, list, list]:
    """
    Retrieve narrative filing text relevant to the sub-question.

    Returns
    -------
    tuple
        ``(findings, citations, tool_calls)``.
    """
    from src.vectorstore.search import search_filings, to_citation

    sub_question = payload.get("sub_question") or payload.get("query", "")
    sections = select_sections(sub_question)

    started = time.monotonic()
    hits = search_filings(
        sub_question,
        ticker=ticker,
        item_sections=sections,
        limit=FILINGS_TOP_K,
        collection=SEARCH_COLLECTION,
    )
    elapsed = int((time.monotonic() - started) * 1000)

    calls = [
        tool_record(
            AGENT,
            "search_filings",
            args={"ticker": ticker, "sections": sections, "k": FILINGS_TOP_K, "collection": SEARCH_COLLECTION},
            provider="qdrant",
            latency_ms=elapsed,
            ok=bool(hits),
        )
    ]

    if not hits:
        logger.info("%s(%s): no hits — has this ticker been ingested?", AGENT, ticker)
        return [], [], calls

    findings = []
    citations = []

    for hit in hits:
        citation = to_citation(hit)
        citations.append(citation)

        excerpt = " ".join(hit["text"].split())
        findings.append(
            finding(
                AGENT,
                f"{ticker} {hit['form_type']} FY{hit['fiscal_year']} "
                f"Item {hit['item_section']} ({hit['section_title']}): {excerpt}",
                ticker=ticker,
                metric=None,
                # Narrative, not numeric — value stays None so the citation
                # verifier's numeric matching does not try to ground it.
                value=None,
                citations=[citation],
                # Retrieval score as confidence lets the synthesizer weigh a
                # marginal hit less than a strong one.
                confidence=round(float(hit["score"]), 3),
            )
        )

    return findings, citations, calls
