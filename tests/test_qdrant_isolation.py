# ═══════════════════════════════════════════════════════
# FinSight — Tests: Qdrant Cross-Project Isolation Guard
# ═══════════════════════════════════════════════════════
#
# This machine runs two Qdrant instances. Another project owns :6333
# (collections athena_content, image_embeddings); FinSight owns :6335.
# A stale QDRANT_URL would have FinSight writing into their database, and
# `make clean` would then destroy their volume.
#
# These tests prove the guard fires. They are the cheapest insurance in the
# project, so they run offline on every commit.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.errors import InfrastructureError, QdrantIsolationError
from src.vectorstore import client as vs_client
from src.vectorstore.config import FORBIDDEN_COLLECTIONS


def _fake_client(collection_names: list[str]) -> MagicMock:
    """A QdrantClient stub whose get_collections returns the given names."""
    fake = MagicMock()
    fake.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name=n) for n in collection_names])
    return fake


class TestForbiddenCollections:
    """The guard list must name the neighbouring project's actual collections."""

    def test_athena_collections_are_listed(self):
        assert "athena_content" in FORBIDDEN_COLLECTIONS
        assert "image_embeddings" in FORBIDDEN_COLLECTIONS

    def test_finsight_collections_are_not_listed(self):
        from src.vectorstore.config import COLLECTION_ALERTS, COLLECTION_FILINGS

        assert COLLECTION_FILINGS not in FORBIDDEN_COLLECTIONS
        assert COLLECTION_ALERTS not in FORBIDDEN_COLLECTIONS


class TestAssertNotForeignInstance:
    """assert_not_foreign_instance is the whole guard — test it hard."""

    def test_passes_on_empty_instance(self):
        vs_client.assert_not_foreign_instance(_fake_client([]))

    def test_passes_on_finsight_own_collections(self):
        vs_client.assert_not_foreign_instance(_fake_client(["finsight_filings", "finsight_alerts"]))

    def test_raises_on_athena_content(self):
        with pytest.raises(QdrantIsolationError, match="belongs to another project"):
            vs_client.assert_not_foreign_instance(_fake_client(["athena_content"]))

    def test_raises_on_image_embeddings(self):
        with pytest.raises(QdrantIsolationError):
            vs_client.assert_not_foreign_instance(_fake_client(["image_embeddings"]))

    def test_raises_even_when_mixed_with_our_own(self):
        # The dangerous case: someone already ingested into the wrong instance.
        with pytest.raises(QdrantIsolationError):
            vs_client.assert_not_foreign_instance(_fake_client(["finsight_filings", "athena_content"]))

    def test_error_names_the_offending_collections(self):
        with pytest.raises(QdrantIsolationError, match="athena_content"):
            vs_client.assert_not_foreign_instance(_fake_client(["athena_content"]))

    def test_error_tells_you_how_to_fix_it(self):
        with pytest.raises(QdrantIsolationError, match="6335"):
            vs_client.assert_not_foreign_instance(_fake_client(["athena_content"]))

    def test_unreachable_instance_raises_infrastructure_error(self):
        broken = MagicMock()
        broken.get_collections.side_effect = ConnectionError("connection refused")

        with pytest.raises(InfrastructureError, match="Cannot reach Qdrant"):
            vs_client.assert_not_foreign_instance(broken)


class TestGetQdrantClient:
    """The factory must verify by default and cache per URL."""

    def setup_method(self):
        vs_client._CLIENT_CACHE.clear()

    def teardown_method(self):
        vs_client._CLIENT_CACHE.clear()

    def test_verifies_by_default(self):
        with patch("qdrant_client.QdrantClient", return_value=_fake_client(["athena_content"])):
            with pytest.raises(QdrantIsolationError):
                vs_client.get_qdrant_client(url="http://localhost:6333")

    def test_verify_false_skips_the_guard(self):
        # Only for throwaway eval collections on a known-safe instance.
        with patch("qdrant_client.QdrantClient", return_value=_fake_client(["athena_content"])):
            assert vs_client.get_qdrant_client(url="http://localhost:6333", verify=False) is not None

    def test_caches_per_url(self):
        with patch("qdrant_client.QdrantClient", return_value=_fake_client([])) as ctor:
            first = vs_client.get_qdrant_client(url="http://localhost:6335")
            second = vs_client.get_qdrant_client(url="http://localhost:6335")

        assert first is second
        assert ctor.call_count == 1


@pytest.mark.integration
class TestLiveIsolation:
    """Against the real containers. Requires `make qdrant`."""

    def test_finsight_instance_is_reachable_and_clean(self):
        client = vs_client.get_qdrant_client()
        names = {c.name for c in client.get_collections().collections}
        assert not (names & FORBIDDEN_COLLECTIONS)

    def test_pointing_at_6333_is_refused(self):
        vs_client._CLIENT_CACHE.clear()
        with pytest.raises(QdrantIsolationError):
            vs_client.get_qdrant_client(url="http://localhost:6333")
        vs_client._CLIENT_CACHE.clear()
