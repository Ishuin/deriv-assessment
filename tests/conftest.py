"""Shared fixtures. Tests stay deterministic: in-memory DB, fixed base URL."""

import pytest

from shortener.config import Settings

TEST_BASE_URL = "http://short.test"
IN_MEMORY_DB = ":memory:"


@pytest.fixture
def settings() -> Settings:
    """Settings that never touch the filesystem or environment."""
    return Settings(db_path=IN_MEMORY_DB, base_url=TEST_BASE_URL)
