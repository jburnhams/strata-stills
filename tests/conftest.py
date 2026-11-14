"""Shared pytest configuration: every test is hermetic (its own home and projects folder, no network, no stray tuning variables), through strata-python's pytest plugin plus `app_env` here. Tests under tests/integration get the `integration` marker automatically."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))


def pytest_collection_modifyitems(items):
    for item in items:
        if os.sep + "integration" + os.sep in str(item.fspath):
            item.add_marker(
                pytest.mark.integration
            )  # strata-python's plugin does this from its next release (CHANGELOG, Unreleased): delete then


@pytest.fixture(autouse=True)
def app_env(isolated_env, monkeypatch):
    """On top of strata-python's `isolated_env` (HOME, STRATASTILLS_HOME and _DATA in a fresh folder, the tuning variables of `strata_tuning_vars` in pyproject.toml removed)
    and its `no_network` guard: the app's own variables. Moves to `strata_env` in pyproject.toml once strata-python releases it (its CHANGELOG, Unreleased).
    The projects folder sits outside the test's own tmp_path, which tests may browse as a source."""
    monkeypatch.setenv("STRATASTILLS_PROJECTS", str(isolated_env / "projects"))
    monkeypatch.setenv(
        "STRATASTILLS_NSFW", "0"
    )  # the nudity filter needs its model: off here; tests/unit/test_nsfw.py turns it on with a fake detector
    monkeypatch.setenv(
        "STRATASTILLS_LEASE", "0"
    )  # never a lease from a strata-models service running on the developer's machine (it can wait behind another app's for an hour); tests/unit/test_lease.py turns it on with a fake client
    monkeypatch.setenv(
        "STRATASTILLS_AI", "0"
    )  # `denoise: auto` brings the learned model in on a noisy picture: not in the tests (the weights may not be there); the tests of that turn it on
    return isolated_env
