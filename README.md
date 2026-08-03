# FinSight

**Multi-agent financial intelligence platform** — grounded research on demand, plus autonomous portfolio surveillance.

Built with **LangGraph** (orchestration), **LangChain** (tooling), **LangSmith** (tracing + evaluation), **Qdrant** (vector search), and **Google Gemini** (reasoning). All financial data comes from free, official sources.

> Status: **Phase 6 complete** — both subsystems are live. Research is routed, fanned out, synthesised, **verified**, and **measured** (40 golden questions, 7 evaluators, six named LangSmith experiments). Monitoring watches a ticker list, scores what it finds by rule, and **deduplicates** it semantically. See [the phase plan](#phases).

## Measured, not asserted

Every number in an answer is supposed to trace back to a primary source. Phase 5
turned that claim into a metric — and the metric found four bugs that 485 unit
tests had not.

| | `baseline` | **`bugfix`** ✅ | `strict-src` | `k12` | `pro-router` | `no-headers` |
|---|---|---|---|---|---|---|
| citation coverage ★ | 0.973 | **0.997** | 0.994 | 0.992 | 0.996 | 0.995 |
| **numeric accuracy** (vs SEC XBRL) | **0.725** | **0.957** | 0.935 | 0.935 | 0.935 | 0.935 |
| answer correctness | 0.700 | 0.800 | 0.825 | **0.875** | 0.825 | 0.800 |
| citation faithfulness | 0.971 | **0.986** | 0.986 | 0.973 | 0.986 | 0.973 |
| refusal correctness | 0.750 | **0.812** | 0.812 | 0.562 | 0.750 | 0.688 |

Coverage read **0.973** at baseline — a system that looks like it works.
Ground-truth accuracy read **0.725**, because the answers were citing real
filings for figures that were four years stale:

```
Q: What was NVIDIA's revenue in fiscal year 2026?
A: NVIDIA reported revenue of $26.914 billion [SRC:EDGAR:0001045810-...]
                                  ↑ that is FY2022. FY2026 is $215.938B.
```

The citation is genuine, so no citation check can catch it. Only ground truth
can.

Four variants were then tested and **none was adopted**: one bought nothing, one
traded better answers for worse refusals, one bought nothing for +31% latency,
and the contextual-header ablation changed 22% of retrieved chunks while
changing the answer not at all.

A fifth finding was a defect in the *evaluator* — a 400-character cap was
truncating every filing chunk before the LLM judge saw it, so an archetype that
scores **1.000** had been reading 0.375 for three experiments.

Full write-up: [`docs/eval_results.md`](./docs/eval_results.md).

## The dedup engine

Subsystem 2's centrepiece, and the part that measurement changed most.

Embedding the display text is wrong in **both** directions:

```
"AAPL fell 5.2%"  vs  "AAPL fell 5.4%"     SAME event, different strings
"AAPL fell 5.2%"  vs  "MSFT fell 5.2%"     DIFFERENT events, near-identical
```

So the embedded text is qualitative, and ticker/type are hard payload filters
rather than soft semantic signals. Run against the real embedder, one story
covered by three outlets plus two unrelated events:

```
 score  decision           headline
 -----  -----------------  -----------------------------------------------
   -    FIRE               Apple hit with DOJ antitrust probe over App Store
 0.913  SUPPRESS_SEMANTIC  Justice Department opens App Store inquiry
 0.898  SUPPRESS_SEMANTIC  Apple faces federal scrutiny of App Store rules
 0.811  MERGE              Apple sued by shareholders over disclosure
   -    FIRE               Apple recalls MacBook adapters over fire risk
   -    FIRE               MSFT hit with DOJ antitrust probe    (filtered out)
```

That run **refuted a design decision**. The plan said "never suppress a HIGH
alert below 0.96 similarity" — but genuine paraphrases land at 0.90, so a 0.96
floor does not prevent a missed event, it guarantees one page per outlet for
every serious story. The rule is now about information rather than similarity:
*a HIGH candidate fires unless the matched parent is already a reported HIGH.*

Full write-up: [`docs/dedup_algorithm.md`](./docs/dedup_algorithm.md).

---

## What it does

Two subsystems, 15 graph nodes, two `StateGraph`s. (Phase 7 adds two more.)

### 1. Interactive Research — request/response

Ask a natural-language question; get an answer where **every number traces back to a primary source**.

```
START → router ──(Send: dynamic fan-out over agent × ticker)──┐
                                                              ▼
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ fundamentals │ filings_rag  │    macro     │  technical   │
        └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘
               └──────────────┴──────────────┴──────────────┘
                                    ▼
                          aggregator  (source-trust conflict resolution)
                                    ▼
                       citation_verifier  (deterministic → LLM)
                          │                        │
             unsupported? ├── Send(targeted requery, max 1) ──┐
                          ▼                                   │
                       finalize → END  ◄─────────────────────┘
```

### 2. Autonomous Monitoring — scheduled, long-running

Watches a ticker watchlist and deduplicates alerts by semantic similarity. Phase 7 adds the human-approval gate for high-severity findings.

```
START → load_watchlist ──(Send: batched price/macro, per-ticker filings/news)──┐
                                                                               ▼
        ┌───────────────┬────────────────┬───────────────┬────────────────┐
        │ price_monitor │ filing_monitor │ news_monitor  │ macro_monitor  │
        └───────┬───────┴────────┬───────┴───────┬───────┴────────┬───────┘
                └────────────────┴───────────────┴────────────────┘
                                        ▼
                        alert_synthesizer  (severity rules → Qdrant dedup)
                                        ▼
                                 persist_cycle → END

           ── Phase 7 inserts here ──────────────────────────────
             any HIGH? → human_approval (interrupt) → dispatcher
```

Five tickers is **12 branches, not 20**: price and macro batch into one `Send`
each because their data sources take a list; filings and news are per-ticker
because EDGAR and Finnhub are per-symbol. That asymmetry is the rate-limit
strategy written as graph topology.

---

## Design decisions worth knowing

- **Numbers come from XBRL; narrative comes from RAG.** A 10-K's financial tables become soup under HTML text extraction — which is exactly where hallucinated figures originate. The LLM is never asked to read a number out of a table. `data.sec.gov` XBRL facts give exact values *and* an accession number for free.
- **Citation verification is deterministic first, LLM second.** A regex extracts every number from the draft answer and matches it against what the tools actually returned, within 0.5% tolerance. The LLM judge only rules on qualitative claims. An LLM checking its own arithmetic is circular.
- **Grounding and accuracy are different metrics, and the gap between them is where the bugs live.** "Does this number match what a tool returned?" is a question the system answers with its own output — a stale tool passes it perfectly. So the eval suite scores answers against SEC XBRL as well, and that is the metric that moved: 0.725 → 0.957.
- **Contextual chunk headers are kept, but they are not the win they were assumed to be.** Ablated against a twin index in Phase 5: they change 22% of retrieved chunks and change the answer not at all. Unfiltered they buy +0.02 on retrieving the right ticker, and in production they cannot matter for the entity at all, because `ticker` is a hard payload filter. Documented because a plausible, well-argued, unverified design decision is exactly what an eval suite is for.
- **Alert dedup strips volatile numerics before embedding.** `"AAPL fell 5.2%"` and `"AAPL fell 5.4%"` are the same event; `"AAPL fell 5.2%"` and `"MSFT fell 5.2%"` are unrelated. So the embedded text is qualitative, and ticker/type are hard payload filters rather than soft semantic signals. The LLM writes the qualitative summary and its compliance is then *verified* with a regex — a summary carrying a magnitude is discarded rather than repaired.
- **The dedup thresholds are not the tutorial's 0.7.** Two *unrelated* financial sentences score 0.65–0.78 with bge-small, and the payload filter has already constrained candidates to the same ticker and type — so the hard negatives are two different events sharing both, which still score ~0.73. A 0.7 threshold would suppress everything.
- **Severity is deterministic rules, not LLM judgement.** An 8-K carrying Item 4.02 (non-reliance on previously issued financials) is automatically HIGH. This keeps severity testable, keeps it stable across months, and stops the model inflating urgency to seem useful — an alert stream whose severity drifts upward is one nobody reads, and a system nobody reads has a false-negative rate of 100%.
- **Conflicting sources are surfaced, not silently resolved.** Beyond a 1% tolerance the answer says which sources disagreed and which was used.

---

## Quick start

```bash
make venv && make install     # uv venv (Python 3.12) + deps
cp .env.example .env          # then fill in — see docs/api_keys.md
gcloud auth application-default login   # if using GEMINI_BACKEND=vertex
make qdrant                   # FinSight's own Qdrant on :6335
make qdrant-check             # verify isolation
make smoke                    # one traced Gemini call → LangSmith run URL
make test                     # unit tests (no network, no LLM spend)
```

### Ask it something

```bash
./run_ingest.sh                                         # filings → Qdrant (once)
./run_research.sh "How did Apple's gross margin trend over the last three fiscal years?"
./run_api.sh                                            # then open localhost:8000/docs
```

### Watch something

```bash
./run_monitor.sh --once --warmup    # FIRST RUN: index candidates, report nothing
./run_monitor.sh --once             # a real cycle
./run_monitor.sh --decisions        # every dedup decision, with its score
```

A cold dedup index makes every candidate look new, so cycle 1 without
`--warmup` would report every open filing, article, and price move at once.

```
  Cycle 20260803T081205Z-66e0b9bd
  5 candidates -> 0 fired, 5 suppressed (5 exact-key, 0 semantic)
```

The exact-key path costs no embedding and no LLM call — verified on a live
cycle, which is how the batching bug that *was* spending one was found.

### Measure it

```bash
make evals                    # baseline over all 40 golden questions
make evals V=k12              # one named single-variable experiment
make evals-check              # is the committed golden dataset current?

# Changed an evaluator rather than the system? Re-score stored runs instead of
# paying for the graph again — the target outputs are identical, so any
# movement is attributable to the measurement.
./run_evals.sh research --regrade p5-bugfix-baseline-b45b642c
```

An eval run is the largest quota spike in this project — ~320 LLM calls in one
batch. `run_evals.sh` prints the estimated calls, minutes, and dollars, and
waits for confirmation before spending anything.

Both doors write to the same checkpoint database, so a question asked at the CLI is replayable at `GET /research/threads/{thread_id}`.

```
STEP  NODES THAT RAN                                    FINDINGS
   1  router                                                   0
   2  fundamentals, fundamentals, technical, technical        22   ← 4 branches, one superstep
   3  aggregator                                              22
   4  citation_verifier                                       22
   5  fundamentals                                            28   ← targeted repair: one agent, not four
   6  aggregator                                              28
   7  citation_verifier                                       28
   8  finalize                                                28
```

That trail is read back from the checkpointer, not written by a logger alongside the run.

### Gemini backends

`GEMINI_BACKEND` picks how Gemini is reached; `get_llm()` hides the difference from every caller.

| | `vertex` | `aistudio` |
|---|---|---|
| Auth | Application Default Credentials — **no key in `.env`** | `GOOGLE_API_KEY` |
| Cost | billed per token, no free tier | free |
| Limit | high quota | requests/minute, `429` on excess |
| `GEMINI_RPM_*` acts as | a **cost** guard | a **quota** guard |

Either way a shared per-tier rate limiter throttles *every* call, including the concurrent `Send()` branches that would otherwise all fire at once.

### ⚠️ Qdrant isolation

This machine already runs a Qdrant on **:6333** owned by another project. FinSight runs its **own** pinned instance on **:6335** with a separate volume, and `src/vectorstore/client.py` refuses to connect if it detects the other project's collections. Never point `QDRANT_URL` at 6333.

---

## Phases

| # | Deliverable | Teaches |
|---|---|---|
| 0 | ✅ Skeleton, isolated Qdrant, traced Gemini call | LangSmith setup |
| 1 | ✅ Data layer: EDGAR, FRED, prices, news — cached, rate-limited, fallback-chained | — (plumbing) |
| 2 | ✅ Qdrant filings RAG: chunking, payload indexes, idempotent ingest | **Qdrant** |
| 3 | ✅ Research graph: router + parallel specialists + aggregator | **LangGraph core** |
| 4 | ✅ Citation verifier, repair loop, checkpointing, REST API | LangGraph checkpoints |
| 5 | ✅ Eval suite A: 40 golden questions, 7 evaluators, measured iteration | **LangSmith evals** |
| 6 | ✅ Monitoring subsystem + the dedup engine | **LangGraph + Qdrant as a data structure** |
| 7 | HITL `interrupt` gate, dispatcher, scheduler, eval suite B | Durable execution |
| 8 | Streamlit dashboard, Docker, public deploy | Consolidation |
| 9 | *(stretch)* Hybrid search — sparse + RRF fusion | Advanced Qdrant |

**Phase gate rule:** do not start phase N+1 until phase N is demoable, tested, and committed.

## Not doing

Deliberately out of scope, to keep this finishable: text-to-SQL, WebSocket streaming, trade execution, options/crypto, non-US equities, a React frontend, multi-user auth, Kubernetes.

---

## Data sources

All free. See [`free_financial_data_sources.md`](./free_financial_data_sources.md).

| Category | Source |
|---|---|
| Filings & XBRL | SEC EDGAR (official, no key, 10 req/s, User-Agent mandatory) |
| Macro | FRED |
| Prices & technicals | yfinance → Finnhub → FMP (fallback chain) |
| News | Finnhub → RSS |

## Architecture background

[`agentic_ai_financial_systems_detailed.md`](./agentic_ai_financial_systems_detailed.md) — the production systems this design is modeled on (Kensho/S&P grounding router, Uber Finch, Captide).
