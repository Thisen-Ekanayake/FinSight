# FinSight

[![CI](https://github.com/Thisen-Ekanayake/FinSight/actions/workflows/ci.yml/badge.svg)](https://github.com/Thisen-Ekanayake/FinSight/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker Compose](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)](./docker-compose.yml)

**Multi-agent financial intelligence platform** — grounded, source-cited research on demand, plus autonomous portfolio surveillance.

Built with **LangGraph** (orchestration), **LangChain** (tooling), **LangSmith** (tracing + evaluation), **Qdrant** (vector search), **Google Gemini** (reasoning), and **React + three.js** (dashboard). Fully containerized; every financial data source is free.

---

## Overview

FinSight is two `StateGraph`s built around one principle: **every claim traces back to a primary source.**

| Subsystem | What it does |
|---|---|
| **Interactive Research** | Ask a natural-language question, get an answer where every number is verified against the tool output that produced it |
| **Autonomous Monitoring** | Watches a ticker watchlist, deduplicates findings semantically, and pauses every HIGH-severity alert behind a durable human-approval gate before dispatch |

### Research pipeline

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

### Monitoring pipeline

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
                       human_approval  (interrupt() iff a HIGH is pending)
                                        ▼
                         dispatcher  (console / file / email)
                                        ▼
                                 persist_cycle → END
```

A HIGH alert checkpoints the run to disk and returns control immediately;
a later `Command(resume=decisions)` call resumes the exact same run:

```bash
./run_monitor.sh --once                                    # may pause: exit code 2
./run_monitor.sh --pending                                 # what's waiting, and on what
./run_monitor.sh --resume <cycle_id> --approve <id> --reject <id>
```

---

## Results

A dedicated eval suite (LangSmith, 40 golden questions, 7 evaluators) caught 4 correctness
bugs that 872 unit tests had not — including citations that were genuine but four years stale.

| | `baseline` | **`bugfix`** ✅ |
|---|---|---|
| citation coverage | 0.973 | **0.997** |
| numeric accuracy (vs SEC XBRL) | 0.725 | **0.957** |
| answer correctness | 0.700 | 0.800 |
| citation faithfulness | 0.971 | **0.986** |

Full write-up: [`docs/eval_results.md`](./docs/eval_results.md).

A second suite (140 hand-labelled pairs, no LLM calls) tuned the alert-dedup
similarity thresholds and found the safe operating band is precision-safe but
recall-fragile — real paraphrases sit close enough to the boundary that most
recall comes from the exact-key path, not the semantic threshold alone. Full
write-up: [`docs/dedup_algorithm.md`](./docs/dedup_algorithm.md).

---

## Design highlights

- **Numbers come from XBRL; narrative comes from RAG.** The LLM is never asked to read a figure out of an HTML table — `data.sec.gov` XBRL facts give exact values and an accession number for free.
- **Citation verification is deterministic first, LLM second.** A regex checks every number in a draft answer against tool output within 0.5% tolerance; the LLM judge only rules on qualitative claims.
- **Grounding and accuracy are measured separately.** "Does this match what a tool returned?" doesn't catch a stale tool; a ground-truth eval against SEC XBRL does.
- **Alert dedup embeds meaning, not text.** Volatile numerics are stripped before embedding so `"AAPL fell 5.2%"` and `"AAPL fell 5.4%"` collapse to one event, while ticker/type stay as hard payload filters rather than soft semantic signals.
- **Severity is deterministic rules, not LLM judgement** — keeps it testable, stable, and immune to urgency inflation.
- **A HIGH alert cannot reach a reader without a human in the loop.** The approval gate is a checkpointed LangGraph `interrupt()`, so it survives a process restart.
- **Conflicting sources are surfaced, not silently resolved.** Beyond a 1% tolerance, the answer states which sources disagreed and which was used.

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

**Ask it something:**

```bash
./run_ingest.sh
./run_research.sh "How did Apple's gross margin trend over the last three fiscal years?"
./run_api.sh    # then open localhost:8000/docs
```

**Look at it:**

```bash
./run_api.sh    # backend first
./run_web.sh    # then localhost:5173
```

Or run the whole stack — Qdrant, API, dashboard — in one command:

```bash
docker compose up --build    # dashboard on localhost:3000
```

See [`docs/deploy.md`](./docs/deploy.md) for putting the stack on a host with a public URL.

**Watch something:**

```bash
./run_monitor.sh --once --warmup    # first run: index candidates, report nothing
./run_monitor.sh --once             # a real cycle
./run_monitor.sh --pending          # cycles paused awaiting a decision
```

**Measure it:**

```bash
make evals           # baseline over all 40 golden questions (~$2 of Vertex spend, prompts before running)
make evals-alerts    # dedup threshold sweep — local embedder only, no cost
```

**Check it:**

```bash
make lint          # flake8 + black --check + isort --check
make type-check     # mypy on src/
make test           # unit tests, no network, no quota
```

---

## Dashboard

A React + three.js frontend in [`frontend/`](frontend/) talks to the API over
plain HTTP and holds no state of its own.

| View | Answers |
|---|---|
| **Desk** | Is there anything that needs a decision — and lets you make it, in place |
| **Ask** | A question, streamed node by node, with every number traceable |
| **Findings** | What was reported, and what was filtered out underneath it |
| **Watchlist** | What's being watched, and what's still cold |
| **System** | What's configured, and what quota remains |

The original Streamlit dashboard remains in [`src/ui/`](src/ui/) as a reference implementation (`./run_ui.sh`, or `docker compose --profile legacy up -d ui`).

---

## Data sources

All free — see [`free_financial_data_sources.md`](./free_financial_data_sources.md).

| Category | Source |
|---|---|
| Filings & XBRL | SEC EDGAR (official, no key, 10 req/s) |
| Macro | FRED |
| Prices & technicals | yfinance → Finnhub → FMP (fallback chain) |
| News | Finnhub → RSS |

---

## Roadmap

| Phase | Deliverable | Focus |
|---|---|---|
| 0 | ✅ Skeleton, isolated Qdrant, traced Gemini call | LangSmith setup |
| 1 | ✅ Data layer: EDGAR, FRED, prices, news | Caching, rate limits, fallback chains |
| 2 | ✅ Qdrant filings RAG | Chunking, payload indexes, idempotent ingest |
| 3 | ✅ Research graph | Router + parallel specialists + aggregator |
| 4 | ✅ Citation verifier, repair loop, REST API | LangGraph checkpoints |
| 5 | ✅ Eval suite A | LangSmith evals, measured iteration |
| 6 | ✅ Monitoring subsystem + dedup engine | LangGraph + Qdrant as a data structure |
| 7 | ✅ HITL interrupt gate, dispatcher, scheduler | Durable execution |
| 8 | ✅ Dashboard, Docker, deploy guide | Consolidation |
| 9 | Hybrid search — sparse + RRF fusion | Advanced Qdrant |

---

## Not doing

Scope decisions, listed because "not built yet" and "deliberately absent" look
identical from outside and only one of them is worth a pull request.

**Access control stops at an allowlist.** `AUTH_ENABLED=true` puts every route
but `/health` behind a Google sign-in, checked against a list of addresses in
`.env` — see [`docs/deploy.md`](./docs/deploy.md) §6. That is the whole model.
There are no roles, no per-user data isolation, and no ownership column on any
table: every allowlisted account can do everything, including approving another
operator's paused alerts. This is a single-operator tool with a real front
door, not a multi-tenant service.

**No per-caller rate limiting.** The limiters in `src/data/rate_limit.py` pace
outbound calls to protect free tiers; they do not throttle inbound ones. An
allowlisted caller can spend Vertex quota freely, and `DAILY_BUDGETS` sets no
Gemini ceiling.

**No session layer.** The Google ID token *is* the credential, so a session
lasts about an hour and expiry means signing in again. No refresh tokens, no
server-side sessions, no cookie or CSRF surface.

**No Kubernetes, no managed platform.** `docker compose up` on one host is the
deployment story, and `docs/deploy.md` is a guide someone follows rather than
anything this repo executes.

---

## Contributing

Contributions are welcome. Please:

1. Follow the [commit convention](./COMMIT_CONVENTION.md) (Conventional Commits).
2. Run `make lint && make type-check && make test` before opening a PR.
3. Read the [Code of Conduct](./CODE_OF_CONDUCT.md) — participation in this project is governed by it.

## License

Licensed under the [MIT License](./LICENSE).
