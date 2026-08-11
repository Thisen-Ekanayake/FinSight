# ═══════════════════════════════════════════════════════
# FinSight — Monitor Eval Runner (Suite B): the TAU_HIGH sweep
# ═══════════════════════════════════════════════════════
#
# Purpose : Measure suppression precision/recall against the REAL embedder
#           over a hand-labelled set of canonical-style text pairs, and
#           sweep TAU_HIGH to find the threshold docs/dedup_algorithm.md
#           promised: the lowest value where suppression precision >= 0.97.
#
# Public API:
#   load_golden(path)        jsonl -> list[GoldenText]
#   score_pairs(rows)        real embedder -> list[ScoredPair]
#   confusion_at(scored, threshold)
#   sweep(scored)             -> list[Confusion], one per threshold step
#   choose_threshold(rows)    the doc's selection rule
#   main()
#
# Usage:
#   ./shell_scripts/run_evals.sh alerts
#   .venv/bin/python -m evals.run_monitor_eval
#   .venv/bin/python -m evals.run_monitor_eval --min-precision 0.95
#
# ══ WHY THIS COSTS NOTHING ══
#   No LLM call anywhere in this file. The golden pairs ARE the canonical,
#   numeric-stripped qualitative text dedup.py embeds in production — hand-
#   writing them means Suite B measures the embedding + threshold layer in
#   isolation from canonicalization quality, which is exactly the layer
#   TAU_HIGH lives in. Embeddings are bge-small, local, free — the sweep can
#   be re-run as often as the thresholds are questioned.
#
# ══ WHY EVERY PAIR SHARES TICKER AND ALERT_TYPE ══
#   Those are hard payload filters in production (see search_similar in
#   src/monitor/alert_store.py) — a threshold is never asked to separate two
#   different tickers or two different alert types, only two candidates that
#   already passed both filters. A dataset that mixed them would spend its
#   statistical power measuring a discrimination the threshold never
#   actually has to make.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import NamedTuple, TypedDict

from evals.config import (
    DEDUP_GOLDEN_PATH,
    MIN_SUPPRESSION_PRECISION,
    SWEEP_HIGH,
    SWEEP_LOW,
    SWEEP_STEP,
)
from src.vectorstore.config import TAU_HIGH

logger = logging.getLogger(__name__)

RULE = "─" * 74


class GoldenText(TypedDict):
    """One hand-written canonical-style text, labelled with which real event it paraphrases."""

    id: str
    ticker: str
    alert_type: str
    cluster: str
    text: str


class ScoredPair(NamedTuple):
    """Two golden texts sharing (ticker, alert_type), and their real cosine similarity."""

    ticker: str
    alert_type: str
    a_id: str
    b_id: str
    same_event: bool
    score: float


class Confusion(NamedTuple):
    """The 2x2 outcome of treating one threshold as the SUPPRESS/duplicate boundary."""

    threshold: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float | None:
        """Of everything this threshold would suppress, what fraction were true duplicates. None if it suppresses nothing."""
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else None

    @property
    def recall(self) -> float | None:
        """Of every true duplicate, what fraction this threshold actually catches."""
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else None


def load_golden(path: Path = DEDUP_GOLDEN_PATH) -> list[GoldenText]:
    """Read the hand-labelled canonical-text set."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_pairs(rows: list[GoldenText]) -> list[ScoredPair]:
    """
    Embed every text once, then score every pair that shares (ticker, alert_type).

    bge-small vectors are already normalised (see FastEmbedBackend), so
    cosine similarity is a plain dot product — no extra normalisation step
    to get subtly wrong.
    """
    from src.vectorstore.embeddings import get_embedder

    embedder = get_embedder()
    vectors = {row["id"]: embedder.embed_symmetric(row["text"]) for row in rows}

    by_group: dict[tuple[str, str], list[GoldenText]] = {}
    for row in rows:
        by_group.setdefault((row["ticker"], row["alert_type"]), []).append(row)

    scored: list[ScoredPair] = []
    for (ticker, alert_type), group in by_group.items():
        for a, b in itertools.combinations(group, 2):
            score = sum(x * y for x, y in zip(vectors[a["id"]], vectors[b["id"]]))
            scored.append(
                ScoredPair(
                    ticker=ticker,
                    alert_type=alert_type,
                    a_id=a["id"],
                    b_id=b["id"],
                    same_event=a["cluster"] == b["cluster"],
                    score=score,
                )
            )
    return scored


def confusion_at(scored: list[ScoredPair], threshold: float) -> Confusion:
    """Classify every pair by ``score >= threshold`` and count the four outcomes."""
    tp = sum(1 for p in scored if p.same_event and p.score >= threshold)
    fp = sum(1 for p in scored if not p.same_event and p.score >= threshold)
    fn = sum(1 for p in scored if p.same_event and p.score < threshold)
    tn = sum(1 for p in scored if not p.same_event and p.score < threshold)
    return Confusion(threshold, tp, fp, fn, tn)


def sweep(
    scored: list[ScoredPair],
    *,
    low: float = SWEEP_LOW,
    high: float = SWEEP_HIGH,
    step: float = SWEEP_STEP,
) -> list[Confusion]:
    """Confusion counts at every threshold from low to high, ascending."""
    n_steps = round((high - low) / step)
    return [confusion_at(scored, round(low + i * step, 2)) for i in range(n_steps + 1)]


def choose_threshold(rows: list[Confusion], *, min_precision: float = MIN_SUPPRESSION_PRECISION) -> Confusion | None:
    """
    The lowest threshold clearing the precision floor.

    Not max F1 — F1 treats a false suppress and a false fire as equally bad,
    and dedup.py's whole design rests on them not being. ``rows`` is swept
    ascending, so the first one clearing the floor IS the lowest.

    Returns
    -------
    Confusion or None
        None if no threshold in the swept range reaches min_precision.
    """
    for row in rows:
        if row.precision is not None and row.precision >= min_precision:
            return row
    return None


def _print_report(scored: list[ScoredPair], rows: list[Confusion], *, min_precision: float) -> None:
    print(f"\n{RULE}\nSuite B — dedup threshold sweep")
    print(f"{RULE}\n{len(scored)} labelled pairs across {len({(p.ticker, p.alert_type) for p in scored})} groups\n")

    print(f"  {'threshold':>9}  {'precision':>9}  {'recall':>9}  {'tp':>4} {'fp':>4} {'fn':>4} {'tn':>4}")
    for row in rows:
        # Every 3rd row plus the current config and the chosen one — the full
        # grid at step 0.01 is 30 rows, more than a human needs to eyeball.
        marker = ""
        if abs(row.threshold - TAU_HIGH) < 1e-9:
            marker = "  <- current TAU_HIGH"
        precision = f"{row.precision:.3f}" if row.precision is not None else "   -  "
        recall = f"{row.recall:.3f}" if row.recall is not None else "   -  "
        print(
            f"  {row.threshold:>9.2f}  {precision:>9}  {recall:>9}  "
            f"{row.true_positive:>4} {row.false_positive:>4} {row.false_negative:>4} {row.true_negative:>4}{marker}"
        )

    chosen = choose_threshold(rows, min_precision=min_precision)
    print(f"\n{RULE}")
    if chosen is None:
        print(f"  No threshold in [{SWEEP_LOW}, {SWEEP_HIGH}] reaches precision >= {min_precision:.2f}.")
    else:
        print(
            f"  Recommended TAU_HIGH = {chosen.threshold:.2f}  "
            f"(precision {chosen.precision:.3f}, recall {chosen.recall:.3f})"
        )
        if abs(chosen.threshold - TAU_HIGH) > 1e-9:
            print(f"  Configured TAU_HIGH is {TAU_HIGH} — consider updating src/vectorstore/config.py.")
        else:
            print("  Matches the configured TAU_HIGH.")
    print(f"{RULE}\n")


def main(argv: list[str] | None = None) -> int:
    """Run the sweep from a shell."""
    parser = argparse.ArgumentParser(description="Suite B: dedup TAU_HIGH threshold sweep")
    parser.add_argument("--golden", type=Path, default=DEDUP_GOLDEN_PATH, help="path to the labelled dataset")
    parser.add_argument("--min-precision", type=float, default=MIN_SUPPRESSION_PRECISION)
    args = parser.parse_args(argv)

    rows = load_golden(args.golden)
    scored = score_pairs(rows)
    if not scored:
        print("No pairs to score — every (ticker, alert_type) group has only one text.")
        return 1

    confusions = sweep(scored)
    _print_report(scored, confusions, min_precision=args.min_precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
