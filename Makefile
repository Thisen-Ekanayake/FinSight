# ═══════════════════════════════════════════════
# FinSight — Developer Makefile
# ═══════════════════════════════════════════════
#
# Usage:
#   make help          Show this help
#   make smoke         One traced Gemini call -> LangSmith run URL
#   make qdrant        Start FinSight's ISOLATED Qdrant (port 6335)
#   make test          Run unit tests (no network, no LLM quota)
#   make lint          Run all linters
# ───────────────────────────────────────────────

.PHONY: help venv install smoke qdrant qdrant-check dev build \
        test test-all test-cov lint type-check format \
        api ingest research monitor monitor-warmup watchlist decisions \
        evals evals-build evals-check logs clean clean-images

# ── Defaults ────────────────────────────────────
PYTHON  ?= .venv/bin/python
COMPOSE := docker compose

# ── Help ────────────────────────────────────────
help:
	@echo ""
	@echo "FinSight — available targets"
	@echo "────────────────────────────────────────────"
	@echo "  venv             Create .venv with Python 3.12 (uv)"
	@echo "  install          Install requirements into .venv"
	@echo "  smoke            One traced Gemini call -> prints LangSmith run URL"
	@echo ""
	@echo "  api              Start the FastAPI server -> /docs"
	@echo "  qdrant           Start FinSight's isolated Qdrant on :6335"
	@echo "  qdrant-check     Verify FinSight :6335 is up AND Athena :6333 is untouched"
	@echo "  dev              Start the full stack (qdrant + api + ui)"
	@echo "  build            Build the FinSight Docker image"
	@echo ""
	@echo "  test             Unit tests only (no network, no LLM quota)"
	@echo "  test-all         All tests including integration + llm"
	@echo "  test-cov         Unit tests with coverage report"
	@echo "  lint             flake8 + black --check + isort --check"
	@echo "  type-check       mypy on src/"
	@echo "  format           Auto-format (black + isort)"
	@echo ""
	@echo "  ingest           EDGAR filings -> Qdrant"
	@echo "  monitor-warmup   FIRST RUN: index candidates, report nothing"
	@echo "  monitor          One monitoring cycle"
	@echo "  watchlist        What is watched, and when it was last checked"
	@echo "  decisions        Every dedup decision, with its similarity score"
	@echo "  research         One-shot research query"
	@echo "  evals            Eval suite A  (make evals V=strict-src for a variant)"
	@echo "  evals-check      Verify the committed golden dataset is current"
	@echo ""
	@echo "  logs             Follow container logs"
	@echo "  clean            Stop containers and remove volumes (GUARDED)"
	@echo ""

# ── Environment ─────────────────────────────────
venv:
	uv venv --python 3.12
	@echo "Created .venv — now run: make install"

install:
	uv pip install --python .venv/bin/python -r requirements.txt

# Ensure .env exists before anything that needs keys
.env:
	@if [ ! -f .env ]; then \
		echo "Creating .env from .env.example..."; \
		cp .env.example .env; \
		echo "  Edit .env and add GOOGLE_API_KEY + LANGSMITH_API_KEY before running"; \
	fi

smoke: .env
	$(PYTHON) -m src.core.smoke

# ── Qdrant (ISOLATED from Athena's instance on :6333) ──
qdrant:
	$(COMPOSE) up -d qdrant

qdrant-check:
	@echo "── FinSight Qdrant (:6335) ──"
	@curl -s http://localhost:6335/collections || echo "  NOT RUNNING"
	@echo ""
	@echo "── Athena Qdrant (:6333) — must be untouched ──"
	@curl -s http://localhost:6333/collections || echo "  not running (fine)"
	@echo ""

# ── Full stack ──────────────────────────────────
dev: .env
	$(COMPOSE) up --build

build:
	docker build -t finsight:local .

# ── Testing ─────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ \
		-m "not slow and not integration and not llm" \
		-v --tb=short

test-all:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-cov:
	$(PYTHON) -m pytest tests/ \
		-m "not slow and not integration and not llm" \
		--cov=src --cov-report=term-missing --cov-report=html \
		-v

# ── Linting / formatting ────────────────────────
lint:
	$(PYTHON) -m flake8 . --config .flake8
	$(PYTHON) -m black --check --diff .
	$(PYTHON) -m isort --check-only --diff .

type-check:
	$(PYTHON) -m mypy src/ --ignore-missing-imports --no-error-summary

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

# ── Entrypoints ─────────────────────────────────
api: .env
	./run_api.sh

ingest:
	./run_ingest.sh

# Pass a question:  make research Q="How did Apple's gross margin trend?"
research:
	./run_research.sh "$(Q)"

# Run --warmup ONCE before the first real cycle. A cold dedup index has nothing
# to match against, so cycle 1 would otherwise report every open filing, every
# recent article, and every price move in one burst.
monitor:
	./run_monitor.sh --once

monitor-warmup:
	./run_monitor.sh --once --warmup

# The watchlist, with each monitor's last-checked watermark.
watchlist:
	./run_monitor.sh --watchlist

# Every dedup decision and the similarity score behind it. This is the table
# Phase 7's threshold sweep is calibrated against — eyeballing it is how you
# notice the bands have drifted before the sweep tells you so.
decisions:
	./run_monitor.sh --decisions

# An eval run is the largest quota spike in this project; run_evals.sh prints
# an estimate and waits for confirmation before spending anything.
#   make evals                    baseline over all 40 examples
#   make evals V=strict-src       one named single-variable experiment
evals:
	./run_evals.sh research $(if $(V),--variant $(V))

# Rebuild the golden dataset from XBRL. Only needed when the spec changes —
# filed figures are immutable, so the committed values never go stale.
evals-build:
	./run_evals.sh build

evals-check:
	./run_evals.sh check

# ── Container utilities ─────────────────────────
logs:
	$(COMPOSE) logs -f

# ── Cleanup ─────────────────────────────────────
# GUARD: `down -v` destroys volumes. Athena's Qdrant/Postgres/Redis live in a
# different compose project on this machine — running this from the wrong
# directory would take their data with it. Refuse unless we are in FinSight.
clean:
	@if [ "$(notdir $(CURDIR))" != "FinSight" ]; then \
		echo "REFUSING: 'make clean' must run from the FinSight directory."; \
		echo "  cwd = $(CURDIR)"; \
		exit 1; \
	fi
	$(COMPOSE) down -v --remove-orphans

clean-images:
	docker rmi -f finsight:local 2>/dev/null || true
