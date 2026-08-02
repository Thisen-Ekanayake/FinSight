# ═══════════════════════════════════════════════════════
# FinSight — Tests: Embedding Backends
# ═══════════════════════════════════════════════════════
#
# Marked `slow`: the first run downloads ~50MB of ONNX weights. No network
# after that, and no API quota is ever consumed — embeddings are local.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import math

import pytest

from src.core.config import EMBEDDING_DIM, EMBEDDING_MODEL
from src.core.errors import ConfigurationError
from src.vectorstore import embeddings as emb
from src.vectorstore.config import BGE_QUERY_PREFIX


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@pytest.fixture(scope="module")
def backend():
    emb.reset_embedder_cache()
    return emb.get_embedder()


class TestBackendSelection:
    def teardown_method(self):
        emb.reset_embedder_cache()

    def test_unknown_backend_is_rejected(self):
        emb.reset_embedder_cache()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(emb, "EMBEDDING_BACKEND", "openai")
            with pytest.raises(ConfigurationError, match="not supported"):
                emb.get_embedder()

    @pytest.mark.slow
    def test_embedder_is_cached(self, backend):
        assert emb.get_embedder() is backend

    @pytest.mark.slow
    def test_satisfies_the_protocol(self, backend):
        assert isinstance(backend, emb.EmbeddingBackend)


@pytest.mark.slow
class TestDimensions:
    """A dimension mismatch must fail loudly, not at Qdrant upsert time."""

    def test_matches_configuration(self, backend):
        assert backend.dimension == EMBEDDING_DIM

    def test_bge_small_is_384(self, backend):
        if EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5":
            assert backend.dimension == 384

    def test_document_vectors_have_the_right_width(self, backend):
        vectors = backend.embed_documents(["hello world", "second chunk"])
        assert len(vectors) == 2
        assert all(len(v) == backend.dimension for v in vectors)

    def test_query_vector_has_the_right_width(self, backend):
        assert len(backend.embed_query("what are the risk factors?")) == backend.dimension

    def test_symmetric_vector_has_the_right_width(self, backend):
        assert len(backend.embed_symmetric("AAPL | PRICE_MOVE | sharp decline")) == backend.dimension


@pytest.mark.slow
class TestAsymmetry:
    """
    The query prefix belongs on queries ONLY. Getting it backwards does not
    raise — it silently costs recall — so pin the behaviour here.
    """

    def test_query_and_document_encodings_differ(self, backend):
        text = "supply chain concentration risk"
        assert backend.embed_query(text) != backend.embed_documents([text])[0]

    def test_symmetric_and_document_encodings_match(self, backend):
        # Neither applies the prefix; they must agree.
        text = "AAPL | NEW_FILING | departure of principal financial officer"
        assert backend.embed_symmetric(text) == pytest.approx(backend.embed_documents([text])[0])

    def test_query_encoding_equals_manually_prefixed_document(self, backend):
        text = "what did management flag as a headwind?"
        manual = backend.embed_documents([BGE_QUERY_PREFIX + text])[0]
        assert backend.embed_query(text) == pytest.approx(manual)


@pytest.mark.slow
class TestSemantics:
    """Sanity checks that the vectors carry meaning, not just shape."""

    def test_identical_text_is_identical(self, backend):
        a, b = backend.embed_documents(["the company faces supply chain risk"] * 2)
        assert _cosine(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_related_text_scores_above_unrelated(self, backend):
        anchor = backend.embed_symmetric("Apple reported record iPhone revenue this quarter")
        related = backend.embed_symmetric("Apple's iPhone sales reached an all-time high")
        unrelated = backend.embed_symmetric("The Federal Reserve raised interest rates")
        assert _cosine(anchor, related) > _cosine(anchor, unrelated)

    def test_hard_negatives_sit_on_an_inflated_floor(self, backend):
        """
        THE THRESHOLD TRAP, pinned as a test.

        The negatives that matter for dedup are NOT random sentences — the
        Qdrant payload filter already constrains candidates to the same ticker
        AND alert type. So the hard case is two genuinely DIFFERENT events
        that share a ticker and type, and those still score ~0.73 cosine.

        A 0.7 threshold — the number every tutorial reaches for — would
        therefore suppress unrelated real events. Thresholds are a property of
        the model and the canonical text format, never a universal constant.

        Measured on bge-small-en-v1.5 (see src/vectorstore/config.py):
            DUPLICATE  0.895 - 0.929
            RELATED    0.801 - 0.887
            DISTINCT   0.728 - 0.742   <- this test
        """
        a = backend.embed_symmetric(
            "AAPL Apple Inc. | PRICE_MOVE | sharp single-day decline breaking below "
            "the 20-day moving average on elevated volume"
        )
        b = backend.embed_symmetric(
            "AAPL Apple Inc. | PRICE_MOVE | strong rally to a new 52-week high on above-average volume"
        )
        score = _cosine(a, b)
        # Opposite-direction moves on the same ticker — semantically distinct.
        assert score > 0.65, f"hard negatives should sit on an inflated floor, got {score:.3f}"
        assert score < 0.80, f"distinct events must stay clear of the merge band, got {score:.3f}"

    def test_measured_thresholds_separate_the_bands(self, backend):
        """The configured TAU values must actually sit between the bands."""
        from src.vectorstore.config import TAU_HIGH, TAU_LOW

        duplicate = _cosine(
            backend.embed_symmetric(
                "AAPL Apple Inc. | NEW_FILING | 8-K Item 5.02 departure of principal financial officer"
            ),
            backend.embed_symmetric(
                "AAPL Apple Inc. | NEW_FILING | 8-K reporting the resignation of the chief financial officer"
            ),
        )
        distinct = _cosine(
            backend.embed_symmetric(
                "AAPL Apple Inc. | NEW_FILING | 8-K Item 5.02 departure of principal financial officer"
            ),
            backend.embed_symmetric(
                "AAPL Apple Inc. | NEW_FILING | 10-Q quarterly report for the third fiscal quarter"
            ),
        )
        assert duplicate >= TAU_HIGH, f"a true duplicate ({duplicate:.3f}) must reach TAU_HIGH ({TAU_HIGH})"
        assert distinct < TAU_LOW, f"a distinct event ({distinct:.3f}) must stay below TAU_LOW ({TAU_LOW})"

    def test_empty_batch_returns_empty(self, backend):
        assert backend.embed_documents([]) == []
