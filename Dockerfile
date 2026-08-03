# ═══════════════════════════════════════════════════════
# FinSight — Container Image
# ═══════════════════════════════════════════════════════
#
# One image, two roles. docker-compose.yml's `api` and `ui` services both
# build from this file and differ only in their `command:` — the API serves
# on 8000, the dashboard on 8501, and both need the same src/ and the same
# dependencies, so there is nothing a second Dockerfile would buy beyond a
# second thing to keep in sync.
#
# Usage:
#   docker build -t finsight:local .
#   docker compose up --build          # api + ui + qdrant together
# ═══════════════════════════════════════════════════════

FROM python:3.12-slim

# fastembed's onnxruntime wheel is precompiled for manylinux, but it dlopen's
# libgomp at import time — slim's base image does not carry it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies before source: this layer only invalidates when
# requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
COPY .streamlit/ .streamlit/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Neither service needs root once its dependencies are installed.
RUN useradd --create-home --uid 1000 finsight \
    && mkdir -p /app/data \
    && chown -R finsight:finsight /app
USER finsight

EXPOSE 8000 8501

# Overridden by docker-compose.yml's `ui` service. --no-reload: autoreload
# watches the filesystem for edits, which is a dev-machine feature a
# container restart on every deploy makes pointless.
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
