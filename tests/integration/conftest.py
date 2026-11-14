"""Integration-suite fixtures: the CLI (in-process and as a real process), a synthetic drone shoot on disk, a project over it, and the HTTP API over real project files. Everything is synthetic (tests/utils/photos.py): no photos, models or network."""

import os

import pytest
from fastapi.testclient import TestClient
from photos import make_shoot
from strata_python.testing import run_cli, run_module

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


@pytest.fixture
def cli():
    """`cli('scan', folder, '--json')` -> CliResult(stdout, stderr, code, .out), in this process (fast, and counted by coverage). Raises AssertionError with the output on a non-zero exit unless `check=False`."""
    from stratastills import cli as C

    return lambda *args, check=True: run_cli(C.main, *args, check=check)


@pytest.fixture
def cli_subprocess(isolated_env):
    """The same through a real `python -m stratastills` process: for the entry-point smoke test and anything that must not share this process's state. Slower."""
    return lambda *args, check=True: run_module("stratastills", *args, src=SRC, check=check)


@pytest.fixture
def shoot(tmp_path):
    """A synthetic drone shoot folder: two JPEG shots (one straight down), a linear-DNG panorama named after the first, a video stub."""
    return make_shoot(str(tmp_path / "shoot"), 2)


@pytest.fixture
def project(cli, shoot):
    """(project id, source id) of a project made over `shoot` through the CLI."""
    from stratastills import projects as P

    cli("new", "Demo", "--source", shoot)
    pid = P.list_projects()[0]["id"]
    return pid, P.source_summaries(pid)[0]["id"]


@pytest.fixture
def client(project):
    """The HTTP API over the real project files of `project`."""
    from stratastills.server.app import create_app

    return TestClient(create_app())
