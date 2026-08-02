# ═══════════════════════════════════════════════════════
# FinSight — Embedding Backends
# ═══════════════════════════════════════════════════════
#
# Purpose : Turn text into vectors, with the query/document asymmetry made
#           explicit in the API so it cannot be got wrong by accident.
#
# Public API:
#   EmbeddingBackend (Protocol)
#   FastEmbedBackend        local ONNX, no torch, no GPU
#   get_embedder()          cached singleton chosen by EMBEDDING_BACKEND
#   reset_embedder_cache()
#
# ══ THREE ENCODE METHODS, NOT ONE ══
#   bge models are ASYMMETRIC. They were trained with an instruction prefix on
#   the QUERY side only:
#
#     embed_documents()  no prefix   — stored filing chunks
#     embed_query()      prefix      — a user's search question
#     embed_symmetric()  no prefix   — alert-vs-alert dedup comparison
#
#   Getting this backwards does not raise; it silently degrades recall, which
#   is far worse. Hence three named methods rather than one flag.
#
#   Dedup is symmetric — comparing two alerts of the same kind — so it uses
#   NO prefix on either side. That distinction is why embed_symmetric exists
#   separately from embed_documents even though they currently do the same
#   thing: the intent is different, and a future model may treat them so.
#
# ══ WHY LOCAL ══
#   Keeps bulk ingest off the metered Gemini quota, needs no GPU (the 4060's
#   8GB stays free), and — decisively — FROZEN WEIGHTS MEAN FROZEN VECTORS.
#   The Phase 7 dedup thresholds are calibrated empirically against this exact
#   model; if a hosted provider silently updated theirs, every tuned threshold
#   would quietly become invalid.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from src.core.config import EMBEDDING_BACKEND, EMBEDDING_DIM, EMBEDDING_MODEL
from src.core.errors import ConfigurationError
from src.vectorstore.config import BGE_QUERY_PREFIX

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface every embedding backend implements."""

    dimension: int
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode texts for STORAGE. No query prefix."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Encode a SEARCH QUERY. Applies the model's query prefix."""
        ...

    def embed_symmetric(self, text: str) -> list[float]:
        """Encode for like-vs-like comparison. No prefix on either side."""
        ...


class FastEmbedBackend:
    """
    Local ONNX embeddings via fastembed.

    Runs on CPU with no torch dependency (~50MB of ONNX weights rather than
    ~2.5GB of framework). On a 14900HX this encodes roughly 800 chunks/second,
    so ingesting 20 filings takes minutes and costs nothing.

    Parameters
    ----------
    model_name : str, optional
        fastembed model id. Defaults to EMBEDDING_MODEL.
    threads : int, optional
        ONNX thread count. Defaults to fastembed's own choice.
    """

    def __init__(self, model_name: str | None = None, *, threads: int | None = None) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name or EMBEDDING_MODEL
        self._model = TextEmbedding(model_name=self.model_name, threads=threads)

        # Trust the model's own reported dimension over the configured one,
        # then fail loudly if they disagree — a silent mismatch would produce
        # vectors Qdrant rejects at upsert time, far from the real cause.
        reported = self._resolve_dimension()
        if reported != EMBEDDING_DIM:
            raise ConfigurationError(
                f"EMBEDDING_DIM is {EMBEDDING_DIM} but {self.model_name} produces "
                f"{reported}-dimensional vectors.\n"
                f"  Fix EMBEDDING_DIM in .env, and note that changing the embedding "
                f"model invalidates every tuned dedup threshold — re-run the sweep."
            )
        self.dimension = reported
        logger.info("Embeddings: %s (%d-d, local ONNX/CPU)", self.model_name, self.dimension)

    def _resolve_dimension(self) -> int:
        """Read the model's output dimension from fastembed's registry."""
        from fastembed import TextEmbedding

        for entry in TextEmbedding.list_supported_models():
            if entry["model"] == self.model_name:
                return int(entry["dim"])
        # Unlisted model: fall back to measuring one embedding.
        return len(list(self._model.embed(["dimension probe"]))[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode texts for storage, in batch.

        Parameters
        ----------
        texts : list of str
            Chunk texts, already carrying their contextual headers.

        Returns
        -------
        list of list of float
            Normalised vectors, aligned with the input order.
        """
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """
        Encode a search query, applying the bge instruction prefix.

        The prefix belongs on queries ONLY. Applying it to stored documents
        (or omitting it here) silently costs recall rather than erroring.
        """
        prefixed = BGE_QUERY_PREFIX + text if self._is_bge() else text
        vector: list[float] = list(self._model.embed([prefixed]))[0].tolist()
        return vector

    def embed_symmetric(self, text: str) -> list[float]:
        """
        Encode for like-vs-like comparison — no prefix on either side.

        Used by the alert dedup engine, which compares one candidate alert
        against previously fired alerts. Both sides are the same kind of text,
        so the asymmetric query prefix would only add noise.
        """
        vector: list[float] = list(self._model.embed([text]))[0].tolist()
        return vector

    def _is_bge(self) -> bool:
        """True when the loaded model uses the bge query-instruction convention."""
        return "bge" in self.model_name.lower()


_EMBEDDER: EmbeddingBackend | None = None


def get_embedder() -> EmbeddingBackend:
    """
    Return the process-wide embedding backend, constructing it once.

    Loading the ONNX model costs a second or two and a first-run download, so
    it is shared rather than rebuilt per call.

    Returns
    -------
    EmbeddingBackend

    Raises
    ------
    ConfigurationError
        If EMBEDDING_BACKEND names an unknown backend.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        if EMBEDDING_BACKEND != "fastembed":
            raise ConfigurationError(
                f"EMBEDDING_BACKEND={EMBEDDING_BACKEND!r} is not supported. "
                f"Only 'fastembed' is implemented; see src/vectorstore/embeddings.py."
            )
        _EMBEDDER = FastEmbedBackend()
    return _EMBEDDER


def reset_embedder_cache() -> None:
    """Drop the cached backend. For tests."""
    global _EMBEDDER
    _EMBEDDER = None
