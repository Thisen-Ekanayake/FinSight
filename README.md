# FinSight

**Multi-agent financial intelligence platform** — grounded research on demand, plus autonomous portfolio surveillance.

Built with **LangGraph** (orchestration), **LangChain** (tooling), **LangSmith** (tracing + evaluation), **Qdrant** (vector search), and **Google Gemini** (reasoning). All financial data comes from free, official sources.

> Status: **Phase 0 — setup gate.** See [the phase plan](#phases).

---

## What it does

Two subsystems, 14 graph nodes, two `StateGraph`s.

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

Watches a ticker watchlist, deduplicates alerts by semantic similarity, and pauses for human approval on high-severity findings.

```
START → load_watchlist ──(Send: batched price/macro, per-ticker filings/news)──┐
                                                                               ▼
        ┌───────────────┬────────────────┬───────────────┬────────────────┐
        │ price_monitor │ filing_monitor │ news_monitor  │ macro_monitor  │
        └───────┬───────┴────────┬───────┴───────┬───────┴────────┬───────┘
                └────────────────┴───────────────┴────────────────┘
                                        ▼
                        alert_synthesizer  (severity rules → Qdrant dedup)
                                        │
                     any HIGH? ─────────┼───────── no ─────────┐
                                        ▼                      │
                            human_approval (interrupt)         │
                                        └──────────┬───────────┘
                                                   ▼
                                             dispatcher → persist_cycle → END
```

---

## Design decisions worth knowing

- **Numbers come from XBRL; narrative comes from RAG.** A 10-K's financial tables become soup under HTML text extraction — which is exactly where hallucinated figures originate. The LLM is never asked to read a number out of a table. `data.sec.gov` XBRL facts give exact values *and* an accession number for free.
- **Citation verification is deterministic first, LLM second.** A regex extracts every number from the draft answer and matches it against what the tools actually returned, within 0.5% tolerance. The LLM judge only rules on qualitative claims. An LLM checking its own arithmetic is circular.
- **Alert dedup strips volatile numerics before embedding.** `"AAPL fell 5.2%"` and `"AAPL fell 5.4%"` are the same event; `"AAPL fell 5.2%"` and `"MSFT fell 5.2%"` are unrelated. So the embedded text is qualitative, and ticker/type are hard payload filters rather than soft semantic signals.
- **Severity is deterministic rules, not LLM judgement.** An 8-K carrying Item 4.02 (non-reliance on previously issued financials) is automatically HIGH. This keeps severity testable and stops the model inflating urgency to seem useful.
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
| 0 | Skeleton, isolated Qdrant, traced Gemini call | LangSmith setup |
| 1 | Data layer: EDGAR, FRED, prices, news — cached, rate-limited, fallback-chained | — (plumbing) |
| 2 | Qdrant filings RAG: chunking, payload indexes, idempotent ingest | **Qdrant** |
| 3 | Research graph: router + parallel specialists + aggregator | **LangGraph core** |
| 4 | Citation verifier, repair loop, checkpointing, REST API | LangGraph checkpoints |
| 5 | Eval suite A: citation faithfulness, measured iteration | **LangSmith evals** |
| 6 | Monitoring subsystem + the dedup engine | LangGraph + Qdrant as a data structure |
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
