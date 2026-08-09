# ═══════════════════════════════════════════════════════
# FinSight — Tests: Evaluation Suite A
# ═══════════════════════════════════════════════════════
#
# Every test here runs offline at zero quota. The judges are exercised through
# their _mock_response hook, and the deterministic evaluators need no hook
# because they never call anything.
#
# ══ WHAT THESE TESTS ARE FOR ══
#   An evaluator is measurement equipment. A miscalibrated one does not fail
#   loudly — it reports a plausible number, and every experiment run against it
#   inherits the error. So the cases below are mostly about the SHAPE of the
#   scores: that an unanswerable question scores its refusal correctly, that a
#   narrative answer with no numbers is not punished for having none, and that
#   an example with nothing to check emits no feedback rather than a free 1.0.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from evals import build_dataset
from evals.config import ARCHETYPES, GOLDEN_PATH, estimate_run
from evals.evaluators import all_evaluators
from evals.evaluators.citation import answer_groundedness, citation_coverage, source_validity
from evals.evaluators.judge import answer_correctness, citation_faithfulness
from evals.evaluators.numeric import numeric_accuracy, refusal_correctness
from evals.variants import VARIANTS, Patch, apply_variant
from src.core.schemas import AgentFinding, Citation

ACCESSION = "0000320193-25-000079"


def _raise() -> None:
    raise RuntimeError("precondition failed")


def _citation(source_id: str = ACCESSION) -> Citation:
    return Citation(
        source_type="EDGAR",
        source_id=source_id,
        url="https://example.com",
        as_of="2025-09-27",
        excerpt=None,
    )


def _finding(claim: str, *, value: float | None = None, metric: str = "revenue@2025 FY") -> AgentFinding:
    return AgentFinding(
        agent="fundamentals",
        ticker="AAPL",
        claim=claim,
        metric=metric,
        period=None,
        value=value,
        unit="USD",
        citations=[_citation()],
        confidence=1.0,
        error=None,
    )


def _outputs(answer: str, *, findings: list[AgentFinding] | None = None, draft: str | None = None) -> dict:
    return {
        "answer": answer,
        "draft_answer": draft if draft is not None else answer,
        "findings": findings if findings is not None else [],
        "citations": [_citation()],
    }


# ── citation_coverage ───────────────────────────────────
def test_a_fully_grounded_answer_scores_one():
    findings = [_finding("AAPL revenue for 2025 FY was 416,161,000,000 USD", value=416_161_000_000.0)]
    result = citation_coverage(_outputs("Apple reported $416.161B in revenue.", findings=findings))

    assert result["score"] == 1.0
    assert "1/1" in result["comment"]


def test_an_invented_number_drags_coverage_down():
    findings = [_finding("AAPL revenue for 2025 FY was 416,161,000,000 USD", value=416_161_000_000.0)]
    answer = "Apple reported $416.161B in revenue and $500.0B in bookings."

    assert citation_coverage(_outputs(answer, findings=findings))["score"] == 0.5


def test_an_answer_with_no_numbers_is_not_punished_for_having_none():
    # The narrative archetype. Scoring this 0 would make every correct
    # qualitative answer look like a failure.
    result = citation_coverage(_outputs("Management flagged supplier concentration as a risk."))

    assert result["score"] == 1.0
    assert "0/0" in result["comment"]


def test_coverage_reads_the_draft_not_the_shipped_answer():
    # This is the whole reason the target returns both. finalize() strips the
    # ungrounded claim, so an evaluator reading only `answer` would report 1.0
    # and hide that the synthesizer invented something.
    findings = [_finding("AAPL revenue for 2025 FY was 416,161,000,000 USD", value=416_161_000_000.0)]
    outputs = _outputs(
        "Apple reported $416.161B in revenue.",
        findings=findings,
        draft="Apple reported $416.161B in revenue and $500.0B in bookings.",
    )

    assert citation_coverage(outputs)["score"] == 0.5
    assert answer_groundedness(outputs)["score"] == 1.0


def test_groundedness_below_one_names_finalize_as_the_culprit():
    findings = [_finding("AAPL revenue for 2025 FY was 416,161,000,000 USD", value=416_161_000_000.0)]
    result = answer_groundedness(_outputs("Apple booked $500.0B.", findings=findings))

    assert result["score"] == 0.0
    assert "survived finalize" in result["comment"]


# ── source_validity ─────────────────────────────────────
def test_a_marker_resolving_to_a_retrieved_source_passes():
    outputs = _outputs(f"Apple reported $416.161B [SRC:EDGAR:{ACCESSION}].")
    results = source_validity(outputs, {})["results"]

    assert results[0]["score"] == 1.0


def test_a_fabricated_accession_number_fails_even_though_it_looks_right():
    outputs = _outputs("Apple reported $416.161B [SRC:EDGAR:0000320193-99-000001].")
    results = source_validity(outputs, {})["results"]

    assert results[0]["score"] == 0.0
    assert "no such source" in results[0]["comment"]


def test_expected_source_recall_catches_the_right_number_from_the_wrong_filing():
    outputs = _outputs(f"Apple reported $416.161B [SRC:EDGAR:{ACCESSION}].")
    results = source_validity(outputs, {"expected_sources": [ACCESSION, "0000320193-24-000123"]})["results"]

    assert results[1]["key"] == "expected_source_recall"
    assert results[1]["score"] == 0.5


def test_no_expected_sources_means_no_recall_score():
    outputs = _outputs(f"Apple reported $416.161B [SRC:EDGAR:{ACCESSION}].")

    assert len(source_validity(outputs, {})["results"]) == 1


# ── numeric_accuracy ────────────────────────────────────
def _expected(value: float, label: str = "AAPL FY2025 revenue") -> dict:
    return {"label": label, "value": value, "unit": "USD", "source_type": "EDGAR", "source_id": ACCESSION}


def test_a_correctly_stated_filed_figure_scores_one():
    result = numeric_accuracy(
        {"archetype": "single_metric"},
        _outputs("Apple reported revenue of $416.161 billion."),
        {"expected_facts": [_expected(416_161_000_000.0)]},
    )

    assert result["score"] == 1.0


def test_the_right_citation_on_the_wrong_number_still_fails():
    # The failure this evaluator exists for: coverage and source validity both
    # pass, because the figure matches a finding and the marker resolves. Only
    # ground truth from XBRL can see that the finding itself was stale.
    result = numeric_accuracy(
        {"archetype": "single_metric"},
        _outputs(f"NVIDIA reported revenue of $26.914 billion [SRC:EDGAR:{ACCESSION}]."),
        {"expected_facts": [_expected(215_938_000_000.0, "NVDA FY2026 revenue")]},
    )

    assert result["score"] == 0.0
    assert "NVDA FY2026 revenue" in result["comment"]


def test_rounding_inside_tolerance_is_accepted():
    result = numeric_accuracy(
        {"archetype": "single_metric"},
        _outputs("Apple reported roughly $416.2 billion."),
        {"expected_facts": [_expected(416_161_000_000.0)]},
    )

    assert result["score"] == 1.0


def test_an_example_with_no_pinned_figures_emits_no_feedback():
    # Not a score of 1.0 — a macro or narrative question was never examined,
    # and counting it as a pass would inflate the mean with nothing.
    result = numeric_accuracy({"archetype": "narrative"}, _outputs("Risks include supply concentration."), {})

    assert result == {"results": []}


# ── refusal_correctness ─────────────────────────────────
def test_declining_without_a_figure_is_the_correct_answer():
    result = refusal_correctness(
        {"archetype": "unanswerable"},
        _outputs("JPMorgan does not report a gross profit line, so the figure is not available."),
    )

    assert result["score"] == 1.0


def test_answering_an_unanswerable_question_scores_zero():
    result = refusal_correctness(
        {"archetype": "unanswerable"},
        _outputs("JPMorgan reported gross profit of $92.4 billion in 2025."),
    )

    assert result["score"] == 0.0
    assert "no available data" in result["comment"]


def test_hedging_while_still_producing_a_number_scores_half():
    result = refusal_correctness(
        {"archetype": "unanswerable"},
        _outputs("The figure is not available, though revenue was $182.4 billion."),
    )

    assert result["score"] == 0.5


def test_refusal_is_not_scored_on_answerable_archetypes():
    result = refusal_correctness({"archetype": "single_metric"}, _outputs("Apple reported $416.161B."))

    assert result == {"results": []}


def test_a_refusal_may_mention_a_fiscal_year_without_penalty():
    # extract_numeric_claims already masks years, so "fiscal 2030" is not a
    # fabricated figure. Without that, every refusal that named the period it
    # was refusing about would score 0.5.
    result = refusal_correctness(
        {"archetype": "unanswerable"},
        _outputs("Fiscal 2030 has not occurred, so no filed revenue figure is available."),
    )

    assert result["score"] == 1.0


# ── judges (mocked — zero quota) ────────────────────────
def test_the_faithfulness_judge_reports_its_grade():
    result = citation_faithfulness(
        {"question": "What was Apple's revenue?"},
        _outputs(f"Apple reported $416.161B [SRC:EDGAR:{ACCESSION}]."),
        _mock_response=(0.5, "the marker follows the wrong clause"),
    )

    assert result["score"] == 0.5
    assert result["comment"] == "the marker follows the wrong clause"


def test_an_answer_with_no_markers_is_not_sent_to_the_faithfulness_judge():
    # It cannot cite wrongly if it does not cite. source_validity already
    # scores the absence; paying for a judge to say so would be waste.
    result = citation_faithfulness({"question": "q"}, _outputs("Apple reported revenue."))

    assert result == {"results": []}


def test_the_correctness_judge_grades_against_the_reference():
    result = answer_correctness(
        {"question": "What was Apple's revenue?"},
        _outputs("Apple reported $416.161B."),
        {"reference_answer": "Apple reported total net sales of $416.161 billion for fiscal 2025."},
        _mock_response=(1.0, "same figure and period"),
    )

    assert result["score"] == 1.0


def test_the_judge_sees_a_whole_filing_chunk_without_truncation():
    # The instrument defect. The cap was 400 characters while every narrative
    # finding is a retrieved chunk of up to CHUNK_SIZE — measured, 16 of 16
    # truncated on a real risk-factors query. The judge was shown a third of
    # each chunk and asked whether it supported a claim whose supporting
    # sentence was usually in the discarded remainder, so it said UNSUPPORTED,
    # correctly, about evidence it could not see.
    #
    # Re-grading the same stored runs with the cap raised took narrative
    # citation_faithfulness from 0.375 to 1.000. The system had been right all
    # along; the evaluator was blindfolding its own judge.
    from evals.evaluators.judge import MAX_JUDGE_CLAIM_CHARS, _render_findings
    from src.vectorstore.config import CHUNK_SIZE

    assert MAX_JUDGE_CLAIM_CHARS > CHUNK_SIZE

    chunk = "AAPL 10-K FY2025 Item 1A (Risk Factors): " + ("supplier concentration. " * 50)
    rendered = _render_findings([_finding(chunk)])

    assert len(chunk) > 400, "fixture must be longer than the old cap or it proves nothing"
    assert chunk in rendered


def test_the_judge_prompt_still_bounds_a_runaway_finding():
    # Unbounded is not the fix either — one malformed finding should not push
    # an entire filing into every graded call.
    from evals.evaluators.judge import MAX_JUDGE_CLAIM_CHARS, _render_findings

    rendered = _render_findings([_finding("x" * 50_000)])

    assert len(rendered) < MAX_JUDGE_CLAIM_CHARS + 500


def test_a_failed_judge_call_leaves_the_example_ungraded_rather_than_failing_it():
    # -1.0 is the sentinel _grade() returns when the call raises. Turning that
    # into a 0.0 would blame the system for a network problem.
    from evals.evaluators import judge

    result = judge._feedback("citation_faithfulness", -1.0, "judge unavailable")

    assert result == {"results": []}


# ── variants ────────────────────────────────────────────
def test_a_variant_patches_the_module_that_uses_the_constant():
    from src.research.agents import filings_rag

    baseline = filings_rag.FILINGS_TOP_K
    with apply_variant("k12"):
        assert filings_rag.FILINGS_TOP_K == 12
    assert filings_rag.FILINGS_TOP_K == baseline


def test_the_baseline_variant_patches_nothing():
    with apply_variant("baseline") as variant:
        assert variant.patches == ()


def test_patching_a_value_that_is_already_set_raises_rather_than_measuring_nothing():
    from src.research.agents import filings_rag

    filings_rag.FILINGS_TOP_K = 12
    try:
        with pytest.raises(ValueError, match="already 12"):
            with apply_variant("k12"):
                pass
    finally:
        filings_rag.FILINGS_TOP_K = 8


def test_the_header_ablation_refuses_to_run_without_its_index():
    # The dangerous shape: a missing collection does not error, retrieval just
    # returns nothing, the narrative archetype collapses — and that reads
    # exactly like a dramatic confirmation that contextual headers matter.
    # A broken setup and a real effect produce the same numbers.
    from evals import variants

    missing = {"name": "x", "exists": False, "points": 0, "vectors": 0, "indexed_fields": [], "status": "missing"}
    with patch("src.vectorstore.collections.collection_stats", return_value=missing):
        with pytest.raises(RuntimeError, match="does not exist"):
            variants.require_ablation_corpus()


def test_the_header_ablation_refuses_to_run_on_a_different_corpus():
    from evals import variants

    def stats(name: str) -> dict:
        points = 2022 if name.endswith("filings") else 1500
        return {"name": name, "exists": True, "points": points, "vectors": points, "indexed_fields": [], "status": "ok"}

    with patch("src.vectorstore.collections.collection_stats", side_effect=stats):
        with pytest.raises(RuntimeError, match="Corpus mismatch"):
            variants.require_ablation_corpus()


def test_matching_corpora_pass_the_ablation_precondition():
    from evals import variants

    def stats(name: str) -> dict:
        return {"name": name, "exists": True, "points": 2022, "vectors": 2022, "indexed_fields": [], "status": "ok"}

    with patch("src.vectorstore.collections.collection_stats", side_effect=stats):
        variants.require_ablation_corpus()


def test_a_precondition_runs_before_any_patch_is_applied():
    # If it ran after, a failing precondition would leave the process patched.
    from evals import variants
    from src.research.agents import filings_rag

    baseline = filings_rag.SEARCH_COLLECTION
    boom = variants.Variant("d", "h", (Patch("src.research.agents.filings_rag", "SEARCH_COLLECTION", "other"),), _raise)

    with patch.dict(variants.VARIANTS, {"boom": boom}):
        with pytest.raises(RuntimeError, match="precondition failed"):
            with variants.apply_variant("boom"):
                pass

    assert filings_rag.SEARCH_COLLECTION == baseline


def test_every_variant_targets_an_attribute_that_actually_exists():
    # The silent failure this guards against: patching the config module
    # instead of its consumer runs the experiment, reports a number, and the
    # number is the baseline's.
    import importlib

    for name, variant in VARIANTS.items():
        for target in variant.patches:
            module = importlib.import_module(target.module)
            assert hasattr(module, target.attribute), f"{name}: {target.module} has no {target.attribute}"


def test_every_variant_changes_exactly_one_thing():
    for name, variant in VARIANTS.items():
        assert len(variant.patches) <= 1, f"{name} changes {len(variant.patches)} variables at once"


# ── dataset integrity ───────────────────────────────────
def test_the_committed_dataset_is_well_formed():
    rows = [json.loads(line) for line in GOLDEN_PATH.open(encoding="utf-8") if line.strip()]

    assert len(rows) == 40
    for row in rows:
        assert row["archetype"] in ARCHETYPES
        assert row["question"].strip()
        assert row["reference_answer"].strip()
        assert row["answerable"] == (row["archetype"] != "unanswerable")


def test_the_archetypes_are_evenly_represented():
    rows = [json.loads(line) for line in GOLDEN_PATH.open(encoding="utf-8") if line.strip()]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["archetype"]] = counts.get(row["archetype"], 0) + 1

    assert set(counts) == set(ARCHETYPES)
    assert all(count == 8 for count in counts.values()), counts


def test_every_pinned_figure_carries_a_real_accession_number():
    import re

    pattern = re.compile(r"^\d{10}-\d{2}-\d{6}$")
    rows = [json.loads(line) for line in GOLDEN_PATH.open(encoding="utf-8") if line.strip()]

    for row in rows:
        for fact in row["expected_facts"]:
            assert pattern.match(fact["source_id"]), fact
            assert fact["value"] != 0


def test_unanswerable_examples_pin_no_figures():
    rows = [json.loads(line) for line in GOLDEN_PATH.open(encoding="utf-8") if line.strip()]

    for row in rows:
        if row["archetype"] == "unanswerable":
            assert row["expected_facts"] == [], row["question"]


def test_the_annual_period_filter_rejects_a_quarter_tagged_as_a_year():
    # A 10-K carries quarterly durations tagged FY. Passing one off as an
    # annual figure is a 4x error in the ground truth itself.
    quarter = {
        "form_type": "10-K",
        "period_start": "2025-06-29",
        "period_end": "2025-09-27",
        "value": 102_466_000_000.0,
    }
    year = {
        "form_type": "10-K",
        "period_start": "2024-09-29",
        "period_end": "2025-09-27",
        "value": 416_161_000_000.0,
    }

    assert not build_dataset._is_annual(quarter)  # type: ignore[arg-type]
    assert build_dataset._is_annual(year)  # type: ignore[arg-type]


def test_a_balance_sheet_instant_counts_as_annual():
    instant = {"form_type": "10-K", "period_start": None, "period_end": "2025-09-27", "value": 359_241_000_000.0}

    assert build_dataset._is_annual(instant)  # type: ignore[arg-type]


def test_a_ten_q_fact_is_never_annual():
    fact = {"form_type": "10-Q", "period_start": "2024-09-29", "period_end": "2025-09-27", "value": 1.0}

    assert not build_dataset._is_annual(fact)  # type: ignore[arg-type]


# ── run economics ───────────────────────────────────────
def test_the_estimate_scales_with_the_dataset():
    small, large = estimate_run(10), estimate_run(40)

    assert large["llm_calls"] == 4 * small["llm_calls"]
    assert large["usd"] > small["usd"]


def test_dropping_the_judges_removes_their_calls_and_lowers_the_cost():
    with_judges, without = estimate_run(40), estimate_run(40, judges=False)

    assert without["judge_calls"] == 0
    assert without["usd"] < with_judges["usd"]


def test_the_full_evaluator_set_is_assembled_in_the_right_order():
    # Deterministic first: they cost nothing and their failures explain the
    # judges' failures more often than the reverse.
    names = [getattr(e, "__name__", "") for e in all_evaluators()]

    assert names.index("citation_coverage") < names.index("citation_faithfulness")
    assert len(all_evaluators(judges=False)) == len(names) - 2
