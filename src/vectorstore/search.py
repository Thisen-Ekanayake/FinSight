# ═══════════════════════════════════════════════════════
# FinSight — Filing Search
# ═══════════════════════════════════════════════════════
#
# Purpose : Semantic search over ingested filings, with hard payload filters.
#
# Public API:
#   FilingHit
#   search_filings(query, ticker=..., form_types=..., item_sections=..., ...)
#   build_filter(...)
#   to_citation(hit)
#
# ══ FILTERS ARE HARD CONSTRAINTS, NOT HINTS ══
#   Asking "what supply chain risks does Apple face?" must never return an
#   Exxon chunk, however semantically similar. Ticker scoping is expressed as
#   a Qdrant payload filter, not left to embedding similarity — and those
#   fields are indexed (see collections.py), so filtering stays cheap.
#
#   Every hit carries its accession number, so a citation is available without
#   the LLM being asked to remember one.
#
# Usage:
#   python -m src.vectorstore.search --ticker AAPL --section 1A \
#       "supply chain concentration risk"
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from typing import Any, TypedDict

from src.core.schemas import Citation
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.config import COLLECTION_FILINGS, DEFAULT_SEARCH_LIMIT
from src.vectorstore.embeddings import get_embedder

logger = logging.getLogger(__name__)


class FilingHit(TypedDict):
    """One retrieved filing chunk, with everything needed to cite it."""

    score: float
    text: str
    ticker: str
    accession_no: str
    form_type: str
    filing_date: str
    fiscal_year: int
    item_section: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int
    source_url: str


def build_filter(
    *,
    ticker: str | list[str] | None = None,
    form_types: list[str] | None = None,
    item_sections: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    since: date | None = None,
    until: date | None = None,
) -> Any | None:
    """
    Build a Qdrant filter from search constraints.

    Every field used here is payload-indexed, so these conditions are cheap.
    Filtering on an unindexed field would silently degrade to a full scan.

    Parameters
    ----------
    ticker : str or list of str, optional
        Restrict to one or more symbols. The most important filter — nearly
        every research query is scoped to a company.
    form_types : list of str, optional
        e.g. ``["10-K"]`` for annual reports only.
    item_sections : list of str, optional
        e.g. ``["1A"]`` to search only Risk Factors.
    fiscal_years : list of int, optional
        Restrict to specific fiscal years.
    since, until : date, optional
        Filing-date bounds, applied as a Range over the integer timestamp.

    Returns
    -------
    Filter or None
        None when no constraints were supplied.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

    conditions: list[Any] = []

    if ticker:
        if isinstance(ticker, str):
            conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker.upper())))
        else:
            conditions.append(FieldCondition(key="ticker", match=MatchAny(any=[t.upper() for t in ticker])))

    if form_types:
        conditions.append(FieldCondition(key="form_type", match=MatchAny(any=[f.upper() for f in form_types])))

    if item_sections:
        conditions.append(FieldCondition(key="item_section", match=MatchAny(any=[s.upper() for s in item_sections])))

    if fiscal_years:
        conditions.append(FieldCondition(key="fiscal_year", match=MatchAny(any=list(fiscal_years))))

    if since or until:
        conditions.append(
            FieldCondition(
                key="filing_date",
                range=Range(
                    gte=time.mktime(since.timetuple()) if since else None,
                    lte=time.mktime(until.timetuple()) if until else None,
                ),
            )
        )

    return Filter(must=conditions) if conditions else None


def _to_hit(scored: Any) -> FilingHit:
    """Convert a Qdrant scored point into a FilingHit."""
    payload = scored.payload or {}
    return FilingHit(
        score=float(scored.score),
        text=payload.get("text", ""),
        ticker=payload.get("ticker", ""),
        accession_no=payload.get("accession_no", ""),
        form_type=payload.get("form_type", ""),
        filing_date=payload.get("filing_date_iso", ""),
        fiscal_year=int(payload.get("fiscal_year") or 0),
        item_section=payload.get("item_section", ""),
        section_title=payload.get("section_title", ""),
        chunk_index=int(payload.get("chunk_index") or 0),
        char_start=int(payload.get("char_start") or 0),
        char_end=int(payload.get("char_end") or 0),
        source_url=payload.get("source_url", ""),
    )


def search_filings(
    query: str,
    *,
    ticker: str | list[str] | None = None,
    form_types: list[str] | None = None,
    item_sections: list[str] | None = None,
    fiscal_years: list[int] | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    score_threshold: float | None = None,
    collection: str = COLLECTION_FILINGS,
) -> list[FilingHit]:
    """
    Semantic search over ingested filing narrative.

    The query is encoded with ``embed_query`` — applying the model's
    instruction prefix — while stored chunks were encoded without it. That
    asymmetry is deliberate; see embeddings.py.

    Parameters
    ----------
    query : str
        Natural-language question.
    ticker, form_types, item_sections, fiscal_years, since, until
        Hard constraints. See build_filter.
    limit : int, default 8
        Maximum hits.
    score_threshold : float, optional
        Drop hits below this cosine score. Leave unset by default: bge
        similarity has an inflated floor, so a naive threshold discards
        genuinely relevant chunks.
    collection : str, default COLLECTION_FILINGS
        Which index to search. Payloads are identical across the ablation
        twin, so a hit is interchangeable whichever index produced it — only
        WHICH chunks come back differs.

    Returns
    -------
    list of FilingHit
        Highest-scoring first.
    """
    client = get_qdrant_client()
    vector = get_embedder().embed_query(query)

    response = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=build_filter(
            ticker=ticker,
            form_types=form_types,
            item_sections=item_sections,
            fiscal_years=fiscal_years,
            since=since,
            until=until,
        ),
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
    )

    hits = [_to_hit(point) for point in response.points]
    logger.info(
        "Search %r (ticker=%s, items=%s) -> %d hits, top score %.3f",
        query[:60],
        ticker or "any",
        item_sections or "any",
        len(hits),
        hits[0]["score"] if hits else 0.0,
    )
    return hits


def to_citation(hit: FilingHit, *, excerpt_chars: int = 300) -> Citation:
    """
    Convert a hit into a Citation for the research graph.

    The excerpt comes from the stored chunk text, which is the raw document
    content — never the embed-time contextual header — so a quoted citation
    matches what the filing actually says.
    """
    return Citation(
        source_type="EDGAR",
        source_id=hit["accession_no"],
        url=hit["source_url"],
        as_of=hit["filing_date"],
        excerpt=hit["text"][:excerpt_chars],
    )


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Search ingested filings from the shell."""
    parser = argparse.ArgumentParser(description="Search SEC filings in Qdrant")
    parser.add_argument("query", help="natural-language question")
    parser.add_argument("--ticker", action="append", help="restrict to symbol (repeatable)")
    parser.add_argument("--form", action="append", help="restrict to form type (repeatable)")
    parser.add_argument("--section", action="append", help="restrict to Item section, e.g. 1A (repeatable)")
    parser.add_argument("--year", action="append", type=int, help="restrict to fiscal year (repeatable)")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)

    from src.core.logging_setup import configure_logging

    configure_logging()

    hits = search_filings(
        args.query,
        ticker=args.ticker,
        form_types=args.form,
        item_sections=args.section,
        fiscal_years=args.year,
        limit=args.limit,
    )

    if not hits:
        print("\nNo hits. Has anything been ingested? Try: make ingest")
        return 1

    print(f"\n{len(hits)} hits for {args.query!r}\n")
    for i, hit in enumerate(hits, 1):
        print(f"  [{i}] {hit['score']:.4f}  {hit['ticker']} {hit['form_type']} FY{hit['fiscal_year']}")
        print(f"      Item {hit['item_section']}. {hit['section_title']}  (acc {hit['accession_no']})")
        text = " ".join(hit["text"].split())
        print(f"      {text[:220]}...")
        print(f"      {hit['source_url']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
