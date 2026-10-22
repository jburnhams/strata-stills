"""Shared pytest configuration: every test is hermetic (its own home and projects folder, no network, no stray tuning variables). Tests under tests/integration get the `integration` marker automatically."""
import os, socket, sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
TUNING = ('STRATASTILLS_GPU', 'STRATASTILLS_DEVICE', 'STRATASTILLS_GPU_SLOTS', 'STRATASTILLS_MAX_PIXELS', 'STRATASTILLS_TILE')             # what a developer sets on their own machine to try things: never leaks into a test


def pytest_collection_modifyitems(items):
    for item in items:
        if os.sep + 'integration' + os.sep in str(item.fspath): item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path_factory, monkeypatch):
    """`~/.stratastills` and the projects folder point into a tmp dir (outside the test's own tmp_path, which tests may browse as a source), so nothing reads or writes the developer's real settings or projects; tuning variables are removed."""
    base = tmp_path_factory.mktemp('env')
    monkeypatch.setenv('HOME', str(base / 'home')); monkeypatch.setenv('USERPROFILE', str(base / 'home'))
    monkeypatch.setenv('STRATASTILLS_HOME', str(base / 'stratastills-home')); monkeypatch.setenv('STRATASTILLS_PROJECTS', str(base / 'projects'))
    for k in TUNING: monkeypatch.delenv(k, raising=False)
    return base


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A test that opens an outbound connection fails at once (loopback and unix sockets are fine)."""
    real = socket.socket.connect

    def guarded(self, address, *a, **kw):
        host = address[0] if isinstance(address, tuple) else None
        if self.family in (socket.AF_INET, socket.AF_INET6) and host not in ('127.0.0.1', '::1', 'localhost'): raise AssertionError(f'tests must not use the network (tried {address!r})')
        return real(self, address, *a, **kw)
    monkeypatch.setattr(socket.socket, 'connect', guarded)
