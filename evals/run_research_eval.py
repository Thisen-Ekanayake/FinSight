# ═══════════════════════════════════════════════════════
# FinSight — Research Eval Runner (Suite A)
# ═══════════════════════════════════════════════════════
#
# Purpose : Push the golden dataset to LangSmith, run the research graph over
#           it under one named variant, and print the scores split by archetype.
#
# Public API:
#   sync_dataset()            create or update finsight-research-golden
#   research_target(inputs)   one graph run, shaped for the evaluators
#   run_experiment(...)       one named experiment
#   main()                    CLI
#
# Usage:
#   ./run_evals.sh research                        # baseline, all 40
#   ./run_evals.sh research --variant k12
#   ./run_evals.sh research --limit 5 --no-judges  # free smoke run
#
# ══ WHY EVERY EXPERIMENT RUNS THE WHOLE DATASET ══
#   --limit exists for checking the harness, not for producing results. Two
#   experiments over different subsets are not comparable, and the comparison
#   view will happily plot them side by side anyway.
#
# ══ WHY EVAL RUNS ARE NOT CHECKPOINTED OR RECORDED ══
#   No checkpointer: four concurrent graphs writing one SQLite file buys
#   contention in exchange for replay nobody wants — LangSmith already holds
#   the full trace of every eval run.
#
#   No research_runs row either. That table backs GET /research/runs, which is
#   a history of questions a person asked. Two hundred eval rows would bury
#   them, and the "runs" number in the API would stop meaning anything.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

from evals.build_dataset import load_jsonl
from evals.config import (
    ARCHETYPES,
    DATASET_RESEARCH,
    EVAL_PROJECT,
    GOLDEN_PATH,
    MAX_CONCURRENCY,
    estimate_run,
)
from evals.evaluators import all_evaluators
from evals.variants import VARIANTS, apply_variant, describe

logger = logging.getLogger(__name__)

RULE = "─" * 74

# Feedback keys in the order they belong in the summary table: the headline
# first, ground truth second, the derived checks after.
SUMMARY_KEYS: tuple[str, ...] = (
    "citation_coverage",
    "numeric_accuracy",
    "answer_correctness",
    "citation_faithfulness",
    "source_validity",
    "expected_source_recall",
    "answer_groundedness",
    "refusal_correctness",
)


# ── Dataset ─────────────────────────────────────────────
def sync_dataset(*, path: Any = None) -> Any:
    """
    Create or refresh the LangSmith dataset from the golden .jsonl.

    Idempotent by design. Examples are replaced wholesale rather than appended,
    because appending on a second run would silently double the dataset and
    every subsequent experiment would report a mean over two copies.

    Parameters
    ----------
    path : Path, optional
        Golden file. Defaults to ``GOLDEN_PATH``.

    Returns
    -------
    Dataset
        The LangSmith dataset object.
    """
    from langsmith import Client

    rows = load_jsonl(path or GOLDEN_PATH)
    client = Client()

    if client.has_dataset(dataset_name=DATASET_RESEARCH):
        dataset = client.read_dataset(dataset_name=DATASET_RESEARCH)
        existing = list(client.list_examples(dataset_id=dataset.id))
        if existing:
            client.delete_examples(example_ids=[e.id for e in existing])
            logger.info("Cleared %d stale examples from %s", len(existing), DATASET_RESEARCH)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_RESEARCH,
            description=(
                "FinSight research questions across five archetypes. Expected figures are "
                "resolved from SEC XBRL companyfacts by evals/build_dataset.py, not typed by hand."
            ),
        )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {
                    "question": row["question"],
                    "archetype": row["archetype"],
                    "tickers": row["tickers"],
                },
                "outputs": {
                    "expected_facts": row["expected_facts"],
                    "expected_sources": row["expected_sources"],
                    "reference_answer": row["reference_answer"],
                    "answerable": row["answerable"],
                },
                "metadata": {"archetype": row["archetype"], "notes": row["notes"]},
            }
            for row in rows
        ],
    )

    logger.info("Synced %d examples to LangSmith dataset %s", len(rows), DATASET_RESEARCH)
    return dataset


# ── Target ──────────────────────────────────────────────
def research_target(inputs: dict) -> dict:
    """
    Run one research query and shape the result for the evaluators.

    Both ``draft_answer`` and ``answer`` are returned because they measure
    different things: the draft is what the synthesizer produced, the answer is
    what survived finalize. Scoring only the answer would hide every claim the
    verifier had to strip.

    Parameters
    ----------
    inputs : dict
        A dataset example's inputs; ``question`` is required.

    Returns
    -------
    dict
        Never raises. A crashed run returns an empty answer with the error
        attached, so it scores zero and stays visible — an exception here would
        drop the example out of the experiment entirely, quietly raising the
        mean of everything that did run.
    """
    from src.research.config import RECURSION_LIMIT
    from src.research.graph import build_research_graph
    from src.research.state import new_state

    question: str = inputs["question"]
    started = time.monotonic()

    try:
        graph = build_research_graph()
        state: dict = graph.invoke(
            new_state(question),
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except Exception as exc:  # pragma: no cover - network dependent
        logger.exception("Graph failed on %r", question)
        return {
            "answer": "",
            "draft_answer": "",
            "findings": [],
            "citations": [],
            "coverage": 0.0,
            "repair_count": 0,
            "errors": [f"graph failed: {exc}"],
            "latency_ms": int((time.monotonic() - started) * 1000),
        }

    report: dict = state.get("verification") or {}

    return {
        "answer": state.get("final_answer") or state.get("draft_answer") or "",
        "draft_answer": state.get("draft_answer") or "",
        "findings": state.get("findings") or [],
        "citations": state.get("citations") or [],
        # The system's OWN stage-1 number, kept for cross-checking against what
        # the evaluator independently recomputes. A gap between them means the
        # verifier and the evaluator disagree, which is worth knowing.
        "coverage": report.get("citation_coverage", 0.0),
        "verification_passed": bool(report.get("passed", False)),
        "unsupported_count": len(report.get("unsupported_claims") or []),
        "repair_count": state.get("repair_count", 0),
        "agents_used": sorted({f.get("agent", "") for f in state.get("findings") or []}),
        "errors": state.get("errors") or [],
        "latency_ms": int((time.monotonic() - started) * 1000),
    }


# ── Summary ─────────────────────────────────────────────
def _collect_scores(results: Any) -> tuple[dict[str, list[float]], dict[str, dict[str, list[float]]]]:
    """
    Pull feedback scores out of an ExperimentResults, overall and per archetype.

    Returns
    -------
    tuple
        ``(overall, by_archetype)``, both keyed by feedback key.
    """
    overall: dict[str, list[float]] = {}
    by_archetype: dict[str, dict[str, list[float]]] = {}

    for row in results:
        example = row.get("example")
        archetype = ((getattr(example, "inputs", None) or {}) or {}).get("archetype", "unknown")

        for feedback in row.get("evaluation_results", {}).get("results", []):
            key = getattr(feedback, "key", None)
            score = getattr(feedback, "score", None)
            if key is None or score is None:
                continue
            overall.setdefault(key, []).append(float(score))
            by_archetype.setdefault(archetype, {}).setdefault(key, []).append(float(score))

    return overall, by_archetype


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def print_summary(name: str, results: Any, *, elapsed: float) -> dict[str, float]:
    """
    Print the scores, overall and split by archetype, and return the means.

    The split is the point. "Coverage is 0.78" is a number; "coverage is 0.78
    because cross-ticker sits at 0.51 while single-metric is at 1.00" is a lead.
    """
    overall, by_archetype = _collect_scores(results)

    print(f"\n{RULE}\nEXPERIMENT  {name}    ({elapsed / 60:.1f} min)\n{RULE}")

    present = [key for key in SUMMARY_KEYS if key in overall]
    for key in present:
        values = overall[key]
        print(f"  {key:26s} {_mean(values):.3f}   (n={len(values)})")

    print(f"\n  {'archetype':16s} " + " ".join(f"{k[:11]:>11s}" for k in present))
    for archetype in ARCHETYPES:
        scores = by_archetype.get(archetype)
        if not scores:
            continue
        cells = " ".join(
            f"{_mean(scores.get(key, [])):>11.3f}" if scores.get(key) else f"{'-':>11s}" for key in present
        )
        print(f"  {archetype:16s} {cells}")

    print()
    return {key: _mean(values) for key, values in overall.items()}


# ── Run ─────────────────────────────────────────────────
def run_experiment(
    *,
    variant: str = "baseline",
    limit: int | None = None,
    judges: bool = True,
    prefix: str = "p5",
) -> dict[str, float]:
    """
    Run one named experiment against the golden dataset.

    Parameters
    ----------
    variant : str, default "baseline"
        Key of ``evals.variants.VARIANTS``. Exactly one variable changes.
    limit : int, optional
        Cap on examples. For smoke-testing the harness only — results from a
        subset are not comparable with a full run.
    judges : bool, default True
        Include the two LLM judges.
    prefix : str, default "p5"
        Experiment name prefix; the variant is appended.

    Returns
    -------
    dict
        Mean score per feedback key.
    """
    from langsmith import Client, evaluate

    client = Client()
    examples: Any = DATASET_RESEARCH
    if limit is not None:
        dataset = client.read_dataset(dataset_name=DATASET_RESEARCH)
        examples = list(client.list_examples(dataset_id=dataset.id))[:limit]

    started = time.monotonic()

    with apply_variant(variant) as applied:
        results = evaluate(
            research_target,
            data=examples,
            evaluators=all_evaluators(judges=judges),
            experiment_prefix=f"{prefix}-{variant}",
            description=applied.description,
            metadata={
                "variant": variant,
                "judges": judges,
                "hypothesis": applied.hypothesis,
                "phase": "P5",
            },
            max_concurrency=MAX_CONCURRENCY,
            client=client,
        )

    return print_summary(f"{prefix}-{variant}", results, elapsed=time.monotonic() - started)


def regrade(experiment: str, *, judges: bool = True, prefix: str = "p5-regrade") -> dict[str, float]:
    """
    Re-score a completed experiment's stored runs with the current evaluators.

    LangSmith accepts an existing experiment as the target, in which case it
    replays the runs it already has instead of calling the system again. That
    makes this the correct tool for the specific case where THE EVALUATOR
    changed and the system did not: the target outputs are byte-identical, so
    any movement is attributable to the measurement and nothing else.

    It is also roughly a tenth of the cost — only the judge calls are paid for,
    not the graph underneath them.

    Parameters
    ----------
    experiment : str
        Name of a completed experiment, e.g. ``p5-bugfix-baseline-b45b642c``.
    judges : bool, default True
        Include the LLM judges. False makes this free.
    prefix : str, default "p5-regrade"
        Experiment name prefix for the new scores.

    Returns
    -------
    dict
        Mean score per feedback key.
    """
    from langsmith import Client, evaluate

    # No experiment_prefix, description, or metadata: LangSmith rejects all
    # three when the target is an existing experiment, because the new scores
    # attach to the SAME experiment's runs rather than creating a new one.
    # That is the behaviour we want — the runs are the constant.
    started = time.monotonic()
    results = evaluate(
        experiment,
        evaluators=all_evaluators(judges=judges),
        max_concurrency=MAX_CONCURRENCY,
        client=Client(),
    )

    return print_summary(f"{prefix} <- {experiment}", results, elapsed=time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    """Sync the dataset and run one experiment."""
    from src.core.logging_setup import configure_logging
    from src.core.tracing import configure_tracing

    parser = argparse.ArgumentParser(description="Run FinSight research eval suite A")
    parser.add_argument("--variant", default="baseline", choices=sorted(VARIANTS), help="single-variable change")
    parser.add_argument("--limit", type=int, default=None, help="cap examples (smoke runs only)")
    parser.add_argument("--no-judges", action="store_true", help="deterministic evaluators only — free")
    parser.add_argument("--prefix", default="p5", help="experiment name prefix")
    parser.add_argument("--sync-only", action="store_true", help="push the dataset and stop")
    parser.add_argument("--regrade", default="", help="re-score a completed experiment's stored runs")
    parser.add_argument("--yes", action="store_true", help="skip the spend confirmation")
    args = parser.parse_args(argv)

    configure_logging(level="WARNING")
    # Experiments go to their own project so eval traffic never mixes into the
    # interactive history in finsight-dev.
    if not configure_tracing(project=EVAL_PROJECT, force=True):
        print("LangSmith tracing is OFF — experiments cannot be recorded.", file=sys.stderr)
        print("Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env.", file=sys.stderr)
        return 1

    sync_dataset()
    if args.sync_only:
        print(f"Dataset {DATASET_RESEARCH} synced.")
        return 0

    if args.regrade:
        # No spend gate: replaying stored runs costs only the judge calls, and
        # skipping the graph is roughly a tenth of a full experiment.
        regrade(args.regrade, judges=not args.no_judges, prefix=args.prefix)
        return 0

    total = len(load_jsonl(GOLDEN_PATH))
    count = min(args.limit, total) if args.limit else total
    judges = not args.no_judges
    estimate = estimate_run(count, judges=judges)

    print(f"\n{RULE}")
    print(f"  {describe(args.variant)}")
    print(
        f"  examples {estimate['examples']}   graph calls ~{estimate['llm_calls']}   judge calls ~{estimate['judge_calls']}"
    )
    print(f"  estimated ~${estimate['usd']:.2f} and ~{estimate['minutes']:.0f} min at concurrency {MAX_CONCURRENCY}")
    print(f"{RULE}\n")

    if not args.yes:
        if input("  proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("  aborted — nothing spent.")
            return 0

    run_experiment(variant=args.variant, limit=args.limit, judges=judges, prefix=args.prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
