# Integration tests

Shared commands, layout, hermetic setup and file-ownership rules are in [`tests/README.md`](../README.md); read that first. This file is owned by integration-test work.

These run the program's boundaries on synthetic photos: the command line, whole pipelines, the HTTP API over real project files. No photos, models or network; `pytest tests/integration -n auto` takes about 20 s.

## Fixtures (`conftest.py`) and helpers (`tests/utils/photos.py`)

| Fixture | What it gives you |
| --- | --- |
| `cli` | `cli('scan', folder, '--json')` -> `Result(stdout, stderr, code, out)`, **in process** (fast, and counted by coverage). Raises with the output on a non-zero exit unless `check=False`. |
| `cli_subprocess` | The same through a real `python -m stratastills` process: the entry-point smoke test, and anything that must not share this process's state. |
| `shoot` | A synthetic drone folder: two JPEG shots (the first straight down, `DJI_..._0001_D.JPG`), a Lightroom-style linear-DNG panorama named after it, a video stub. |
| `project` | `(project id, source id)` of a project made over `shoot` through the CLI. |
| `client` | A FastAPI `TestClient` over the real project files of `project`. |

`photos.py`: `make_jpeg` (texture, grid lines turned by `tilt` degrees, the XMP block a DJI drone writes), `make_linear_dng` (a preview in the main IFD, 16 bit RGB in a sub-IFD marked LinearRaw, DJI make and model, colour matrices, as-shot neutral), `make_shoot`.

## Conventions

- **Decide whether it belongs here.** A test that needs a real file layout, several modules together, or the CLI / HTTP boundary is an integration test. Pure logic on arrays belongs in `unit/`.
- Use `cli` in process by default; assert on the files a command wrote and on the JSON it returned, not only on printed text.
- Fake only what needs a model or a network; everything else is real, including the files.
- Keep each test under ~10 s and parallel-safe (no fixed ports, no shared paths).
- The panorama is a stand-in for Lightroom's merges: it exercises the linear-DNG reader, the device profile, the levels matching to the neighbouring frames, and (with `STRATASTILLS_TILE` / `STRATASTILLS_MAX_PIXELS`) the tiled developers.

## Coverage

Not gated. `pytest tests/integration -n auto --cov --cov-fail-under=0 --cov-report=term-missing`.

| Area | Tests |
| --- | --- |
| `cli.py` | `test_cli.py`: doctor, scan (shots, kinds, drone metadata), project and source commands, settings, a real process |
| `pipeline.py`, `develop.py`, `lineardng.py`, `inputs.py`, `match.py` | `test_pipeline_end_to_end.py`: a JPEG shot, a levelled oblique shot, a linear DNG panorama, rules, tiled against whole on both code paths |
| `server/app.py` | `test_server_api.py`: browse, generate, download, a rule |

## Known issues

- The real Mini 4 Pro files carry JPEG XL tiles, opcode lists and Adobe profiles the synthetic DNG does not (uncompressed, no opcodes); those paths are covered by unit tests of the parsing functions and by the manual runs recorded in `docs/progress.md`.
