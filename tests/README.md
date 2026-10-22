# Python tests

pytest 8, pytest-cov, **hypothesis**, **pytest-randomly**, **pytest-timeout**, pytest-xdist, diff-cover (`pip install -r requirements-test.txt`). The web tests are in `web/tests` (see `web/tests/README.md`).

This file holds only what **both** Python suites share, and it is rarely edited. Each suite keeps its own notes so that parallel work on them does not collide:

| Suite | Notes and fixtures |
| --- | --- |
| Unit (`tests/unit/`) | [`tests/unit/README.md`](unit/README.md): conventions, the CPU/GPU equivalence tests, known issues |
| Integration (`tests/integration/`) | [`tests/integration/README.md`](integration/README.md): fixtures, conventions, known issues |
| Web (`web/tests/`) | [`web/tests/README.md`](../web/tests/README.md) |

```
pytest tests/unit --cov                    # unit suite + coverage (what CI gates on: fail_under in pyproject.toml, only ratchets up)
pytest tests/unit -n auto                  # in parallel
pytest tests/integration -n auto           # the CLI, pipelines and HTTP API on synthetic photos
pytest -p randomly -p "randomly_seed=N"    # reproduce a failure from a CI run (the seed is in the header of every run)
pytest tests/unit/test_x.py -k name        # one test
ruff check src tests
diff-cover coverage.xml --compare-branch origin/main    # after `--cov-report=xml`: coverage of just the lines your change touches
```

CI (`.github/workflows/`): `unit.yml` (lint, unit tests with coverage on Linux for Python 3.11-3.13 plus macOS and Windows; web typecheck, coverage gate and build) and `integration.yml` (integration tests on Linux, macOS and Windows; the web component tests; also weekly and on demand).

## Layout

- `unit/`: fast, isolated, synthetic data; no real photos, no models.
- `integration/`: the CLI (in process and as a real process), pipelines end to end, the HTTP API over real project files, on synthetic photos written by `utils/photos.py`. Not part of the coverage gate.
- Both suites are hermetic (autouse in `conftest.py`): `~`, `STRATASTILLS_HOME` and `STRATASTILLS_PROJECTS` point into a tmp dir outside `tmp_path`, the tuning variables (`STRATASTILLS_GPU`, `_DEVICE`, `_GPU_SLOTS`, `_MAX_PIXELS`, `_TILE`) are removed, and any outbound (non-loopback) connection fails the test.
- `utils/`: helpers importable by name in any test (`from photos import make_shoot`; `tests/utils` is on `pythonpath`). `photos.py` makes the synthetic photos: textured JPEGs with a DJI drone's XMP, a Lightroom-style linear DNG, a whole shoot folder.
- One test file per module (`develop.py` -> `unit/test_develop.py`; add to the existing file if there is one).
- Real photos (`shoots/`, the drone folders on `\\Server\Drone`) are never required by tests and never committed.

## Who owns which files (so parallel branches don't conflict)

| Area | Unit work | Integration work |
| --- | --- | --- |
| Tests | `tests/unit/**` | `tests/integration/**` |
| Fixtures | `tests/conftest.py`, `tests/unit/conftest.py` | `tests/integration/conftest.py` only |
| Helpers | `tests/utils/<topic>.py` | `tests/utils/integration_<topic>.py` |
| Notes | `tests/unit/README.md` | `tests/integration/README.md` |
| Coverage gate | `fail_under` in `pyproject.toml` (the only writer) | none: integration coverage is informational |

Don't edit `tests/README.md` (this file), `AGENTS.md`, `CLAUDE.md`, `requirements-test.txt` or `.github/workflows/*` unless the task cannot be done without it; say why in the PR.

## Shared conventions

- pytest style: `def test_...`, plain `assert`, `pytest.raises(..., match=...)`, `@pytest.mark.parametrize`. No `sys.path` hacks, no `if __name__ == '__main__'` runners.
- Test behaviour through the public function, CLI or HTTP endpoint, with specific assertions (exact values, not `is not None`).
- A test must pass in any order, alone and in parallel (pytest-randomly shuffles every run) and must not touch the real home folder or the network. Use `monkeypatch` and `tmp_path`, not module globals and `tempfile`.
- No `time.sleep` in tests (poll with a bound, or inject the clock).
- Never "fix" the code to make a test pass in a test-only change. A bug found while testing: write the test for the correct behaviour with `@pytest.mark.xfail(strict=True, reason='...')` and say so in the PR.
- **GPU code**: every function in `gpu.py` has a CPU original in `develop.py` / `detail.py`, and `unit/test_gpu.py` holds them to the same answer on every device (torch-CPU always, CUDA and MPS where present; the MPS tests skip elsewhere). Change the recipe in the CPU code first and mirror it. Likewise `gpu_tiled.py` and `cpu_tiled.py` are held to the whole-picture result (`test_gpu.py`, and `integration/test_pipeline.py` on a real file).
