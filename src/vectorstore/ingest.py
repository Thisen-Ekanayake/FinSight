# ═══════════════════════════════════════════════════════
# FinSight — Filing Ingest
# ═══════════════════════════════════════════════════════
#
# Purpose : filing -> chunks -> embeddings -> Qdrant, idempotently.
#
# Public API:
#   point_id(accession_no, chunk_index)     deterministic UUIDv5
#   ingest_filing(filing, ...)              one filing
#   ingest_ticker(ticker, ...)              a ticker's recent filings
#   IngestReport
#
# ══ IDEMPOTENCY VIA DETERMINISTIC IDS ══
#   Point IDs are uuid5(NAMESPACE_URL, f"{accession_no}:{chunk_index}") rather
#   than random UUIDs. Re-ingesting the same filing therefore OVERWRITES the
#   same points instead of appending duplicates, so `run_ingest.sh` is safe to
#   run repeatedly and the collection never silently doubles.
#
#   This works because a filing is immutable once accepted — an accession
#   number will never point at different bytes — so the same input genuinely
#   should produce the same point.
#
# Usage:
#   python -m src.vectorstore.ingest --ticker AAPL --form 10-K --limit 2
#   python -m src.vectorstore.ingest --watchlist          # the default set
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import time
import uuid
from datetime import date
from typing import TypedDict

from src.core.errors import DataSourceError
from src.data.edgar import download_filing_document, get_filing_index
from src.data.schemas import FilingRef
from src.vectorstore.chunking import FilingChunk, chunk_filing, contextual_header
from src.vectorstore.client import get_qdrant_client
from src.vectorstore.collections import ensure_collections
from src.vectorstore.config import COLLECTION_FILINGS, COLLECTION_FILINGS_ABLATION
from src.vectorstore.embeddings import get_embedder

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 256

# Fixed namespace so IDs stay stable across runs, machines, and re-clones.
POINT_NAMESPACE = uuid.NAMESPACE_URL


class IngestReport(TypedDict):
    """Outcome of ingesting one filing."""

    accession_no: str
    ticker: str
    form_type: str
    chunks: int
    upserted: int
    skipped: bool
    reason: str | None
    elapsed_ms: int


def point_id(accession_no: str, chunk_index: int) -> str:
    """
    Deterministic point ID for a filing chunk.

    Same filing + same chunk index always yields the same UUID, which is what
    makes re-ingest an overwrite rather than a duplicate insert.

    Parameters
    ----------
    accession_no : str
        The filing's accession number — globally unique and immutable.
    chunk_index : int
        Position of the chunk within the filing.

    Returns
    -------
    str
        A UUIDv5 string.
    """
    return str(uuid.uuid5(POINT_NAMESPACE, f"{accession_no}:{chunk_index}"))


def _to_payload(chunk: FilingChunk) -> dict:
    """
    Build a chunk's Qdrant payload.

    Stores the RAW chunk text, not the embed-time contextual header, so a
    citation quotes what the filing actually says.
    """
    return {
        "ticker": chunk["ticker"],
        "cik": chunk["cik"],
        "accession_no": chunk["accession_no"],
        "form_type": chunk["form_type"],
        # Unix timestamp so Qdrant Range filters work on it.
        "filing_date": int(time.mktime(date.fromisoformat(chunk["filing_date"]).timetuple())),
        "filing_date_iso": chunk["filing_date"],
        "fiscal_year": chunk["fiscal_year"],
        "period_of_report": chunk["period_of_report"],
        "item_section": chunk["item_section"],
        "section_title": chunk["section_title"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"],
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "source_url": chunk["source_url"],
    }


def ingest_filing(
    filing: FilingRef,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_existing: bool = False,
    collection: str | None = None,
    contextual: bool = True,
) -> IngestReport:
    """
    Chunk, embed, and upsert one filing.

    Parameters
    ----------
    filing : FilingRef
        The filing to ingest.
    batch_size : int, default 256
        Chunks embedded and upserted per batch.
    skip_existing : bool, default False
        Skip the filing entirely if any of its points are already present.
        Deterministic IDs make re-ingest harmless, so this is purely a time
        saver for large backfills.
    collection : str, optional
        Target collection. Defaults to COLLECTION_FILINGS. Only the ablation
        twin should ever differ.

        ══ RESOLVED AT CALL TIME, NOT AS A DEFAULT ARGUMENT ══
        `collection: str = COLLECTION_FILINGS` would bind the value when this
        module is IMPORTED, so rebinding ingest.COLLECTION_FILINGS afterwards
        would change the module attribute and nothing else. That is the same
        late-binding trap documented in evals/variants.py, except a function
        default is out of reach of module-attribute patching entirely.

        It mattered: the integration tests point this at a throwaway
        collection that way, and with a default argument every write went to
        the production index instead. Nothing errored, because point IDs are
        deterministic — the writes landed on the real AAPL points and
        overwrote them with identical content. A filing NOT already present
        would simply have been added.
    contextual : bool, default True
        Prepend the contextual header before embedding. False builds the
        ablation index — see COLLECTION_FILINGS_ABLATION.

        These two travel together on purpose: writing header-less vectors into
        the production collection would silently corrupt it, because the
        payloads are byte-identical and the point IDs collide, so nothing would
        look wrong until retrieval quality quietly dropped.

    Returns
    -------
    IngestReport
        ``skipped`` is True when nothing was written, with ``reason`` set.
        A filing that fails to parse is skipped rather than raising — never
        ingest garbage.
    """
    collection = collection or COLLECTION_FILINGS

    if not contextual and collection == COLLECTION_FILINGS:
        raise ValueError(
            "Refusing to write header-less vectors into the production collection: point IDs are "
            f"deterministic, so they would overwrite the real vectors in place. Target {COLLECTION_FILINGS_ABLATION}."
        )

    started = time.monotonic()
    client = get_qdrant_client()

    def report(chunks: int, upserted: int, *, skipped: bool = False, reason: str | None = None) -> IngestReport:
        return IngestReport(
            accession_no=filing["accession_no"],
            ticker=filing["ticker"],
            form_type=filing["form_type"],
            chunks=chunks,
            upserted=upserted,
            skipped=skipped,
            reason=reason,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    if skip_existing and _already_ingested(filing["accession_no"], collection=collection):
        logger.info("%s %s: already ingested, skipping", filing["ticker"], filing["accession_no"])
        return report(0, 0, skipped=True, reason="already ingested")

    try:
        path = download_filing_document(filing)
    except DataSourceError as exc:
        logger.warning("%s: download failed — %s", filing["accession_no"], exc)
        return report(0, 0, skipped=True, reason=f"download failed: {exc}")

    chunks = chunk_filing(path, filing)
    if not chunks:
        return report(0, 0, skipped=True, reason="no narrative sections parsed")

    embedder = get_embedder()
    upserted = 0

    from qdrant_client.models import PointStruct

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        # Contextual header applied HERE, at embed time only. The payload keeps
        # the raw text so citations quote the real document.
        prefix = contextual_header if contextual else (lambda _c: "")
        vectors = embedder.embed_documents([prefix(c) + c["text"] for c in batch])

        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id(c["accession_no"], c["chunk_index"]),
                    vector=vector,
                    payload=_to_payload(c),
                )
                for c, vector in zip(batch, vectors)
            ],
            wait=False,  # let HNSW index in the background; we are bulk loading
        )
        upserted += len(batch)

    logger.info(
        "Ingested %s %s %s: %d chunks in %dms",
        filing["ticker"],
        filing["form_type"],
        filing["accession_no"],
        upserted,
        int((time.monotonic() - started) * 1000),
    )
    return report(len(chunks), upserted)


def _already_ingested(accession_no: str, *, collection: str | None = None) -> bool:
    """True if any point for this accession number already exists."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    collection = collection or COLLECTION_FILINGS

    client = get_qdrant_client()
    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[FieldCondition(key="accession_no", match=MatchValue(value=accession_no))]),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


def ingest_ticker(
    ticker: str,
    *,
    forms: list[str] | None = None,
    limit: int = 4,
    since: date | None = None,
    skip_existing: bool = False,
    collection: str | None = None,
    contextual: bool = True,
) -> list[IngestReport]:
    """
    Ingest a ticker's recent filings.

    Parameters
    ----------
    ticker : str
        US-listed symbol.
    forms : list of str, optional
        Form types. Defaults to 10-K and 10-Q.
    limit : int, default 4
        Maximum filings to ingest.
    since : date, optional
        Only filings on or after this date.
    skip_existing : bool, default False
        Skip filings already present in the collection.
    collection : str, default COLLECTION_FILINGS
        Target collection.
    contextual : bool, default True
        Prepend the contextual header before embedding.

    Returns
    -------
    list of IngestReport
        One per filing attempted.
    """
    filings = get_filing_index(ticker, forms=forms or ["10-K", "10-Q"], since=since, limit=limit)
    if not filings:
        logger.warning("%s: no matching filings found", ticker)
        return []

    return [
        ingest_filing(f, skip_existing=skip_existing, collection=collection, contextual=contextual) for f in filings
    ]


# ── CLI ─────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    """Ingest filings into Qdrant."""
    from src.core.config import PROJECT_ROOT  # noqa: F401  (ensures .env is loaded)
    from src.core.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Ingest SEC filings into Qdrant")
    parser.add_argument("--ticker", action="append", help="symbol (repeatable)")
    parser.add_argument("--watchlist", action="store_true", help="ingest the configured watchlist")
    parser.add_argument("--form", action="append", help="form type (repeatable); default 10-K and 10-Q")
    parser.add_argument("--limit", type=int, default=4, help="max filings per ticker")
    parser.add_argument("--since", help="only filings on/after this ISO date")
    parser.add_argument("--skip-existing", action="store_true", help="skip filings already ingested")
    parser.add_argument(
        "--no-headers",
        action="store_true",
        help=(
            f"ablation: embed WITHOUT the contextual header into {COLLECTION_FILINGS_ABLATION}. "
            "Implies --collection; the production index is never touched."
        ),
    )
    args = parser.parse_args(argv)

    configure_logging()

    contextual = not args.no_headers
    collection = COLLECTION_FILINGS if contextual else COLLECTION_FILINGS_ABLATION

    if args.watchlist:
        import os

        tickers = [t.strip() for t in os.getenv("MONITOR_WATCHLIST", "AAPL,MSFT,NVDA,JPM,XOM").split(",") if t.strip()]
    elif args.ticker:
        tickers = args.ticker
    else:
        parser.error("pass --ticker or --watchlist")

    ensure_collections()
    if not contextual:
        from src.vectorstore.collections import ensure_collection
        from src.vectorstore.config import FILINGS_PAYLOAD_INDEXES

        ensure_collection(collection, FILINGS_PAYLOAD_INDEXES)
        print(f"\nABLATION: embedding without contextual headers into {collection}")

    all_reports: list[IngestReport] = []
    for ticker in tickers:
        all_reports.extend(
            ingest_ticker(
                ticker,
                forms=args.form,
                limit=args.limit,
                since=date.fromisoformat(args.since) if args.since else None,
                skip_existing=args.skip_existing,
                collection=collection,
                contextual=contextual,
            )
        )

    print(f"\n{'ticker':8s} {'form':8s} {'accession':22s} {'chunks':>7s}  status")
    print("─" * 72)
    for r in all_reports:
        status = f"skipped: {r['reason']}" if r["skipped"] else f"ok ({r['elapsed_ms']}ms)"
        print(f"{r['ticker']:8s} {r['form_type']:8s} {r['accession_no']:22s} {r['chunks']:>7,}  {status}")

    total = sum(r["upserted"] for r in all_reports)
    print(f"\n{total:,} chunks upserted across {len(all_reports)} filings")

    from src.vectorstore.collections import collection_stats

    stats = collection_stats(collection)
    print(f"{collection}: {stats['points']:,} points total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
