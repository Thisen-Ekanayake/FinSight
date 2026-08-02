# ═══════════════════════════════════════════════════════
# FinSight — Tests: Core Configuration
# ═══════════════════════════════════════════════════════
# Offline only. No network, no Gemini quota.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core import config
from src.core.errors import MissingCredentialError


class TestPaths:
    """Path resolution should anchor to the project root, not the cwd."""

    def test_project_root_contains_pyproject(self):
        assert (config.PROJECT_ROOT / "pyproject.toml").is_file()

    def test_relative_paths_resolve_under_project_root(self):
        assert config.CHECKPOINT_DB.is_absolute()
        assert config.EDGAR_CACHE_DIR.is_absolute()
        assert config.PROJECT_ROOT in config.CHECKPOINT_DB.parents

    def test_absolute_configured_path_is_left_alone(self):
        assert config._resolve("/tmp/somewhere.sqlite") == Path("/tmp/somewhere.sqlite")


class TestModelTiers:
    """Both tiers must map to a real model name, and RPM must be positive."""

    def test_every_tier_has_a_model(self):
        assert set(config.MODEL_BY_TIER) == {"flash", "pro"}
        assert all(name for name in config.MODEL_BY_TIER.values())

    def test_every_tier_has_a_rate_limit(self):
        assert set(config.GEMINI_RPM) == set(config.MODEL_BY_TIER)
        assert all(rpm > 0 for rpm in config.GEMINI_RPM.values())

    def test_pro_tier_is_more_constrained_than_flash(self):
        # Pro quota is scarcer on the free tier; the defaults should reflect
        # that so a careless `get_llm("pro")` in a fan-out does not stall.
        assert config.GEMINI_RPM["pro"] <= config.GEMINI_RPM["flash"]


class TestRequireKey:
    """require_key defers validation to call time so tests can import freely."""

    def test_returns_value_when_set(self):
        with patch.dict("os.environ", {"FAKE_KEY": "abc123"}):
            assert config.require_key("FAKE_KEY") == "abc123"

    def test_raises_when_missing(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(MissingCredentialError, match="NOPE_KEY is not set"):
                config.require_key("NOPE_KEY")

    def test_raises_when_empty_string(self):
        with patch.dict("os.environ", {"EMPTY_KEY": ""}):
            with pytest.raises(MissingCredentialError):
                config.require_key("EMPTY_KEY")

    def test_error_points_at_the_provisioning_doc(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(MissingCredentialError, match="docs/api_keys.md"):
                config.require_key("SOME_KEY")


class TestQdrantUrlDefault:
    """The default must be FinSight's port, never the neighbouring project's."""

    def test_default_url_is_not_port_6333(self):
        assert "6333" not in config.QDRANT_URL, (
            "QDRANT_URL must not point at 6333 — that instance belongs to another "
            "project on this machine. FinSight uses 6335."
        )

    def test_default_url_is_port_6335(self):
        assert "6335" in config.QDRANT_URL


class TestEmbeddingConfig:
    """Embedding dimension must match the model; changing it invalidates thresholds."""

    def test_bge_small_is_384_dimensional(self):
        if config.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5":
            assert config.EMBEDDING_DIM == 384

    def test_backend_is_local_by_default(self):
        # Local embeddings keep bulk ingest off the rate-limited Gemini quota.
        assert config.EMBEDDING_BACKEND == "fastembed"
