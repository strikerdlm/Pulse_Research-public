"""Shared pytest fixtures."""
from __future__ import annotations

import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture(scope="session")
def project_root() -> pathlib.Path:
    """Absolute path to the repo root."""
    return PROJECT_ROOT
