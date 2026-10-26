# strata-stills

Takes raw(ish) still photos (DJI drone DNG + JPEG today; other cameras to follow) and makes them look their best automatically: picks the source, develops the raw, levels and straightens, finds what is in the picture, frames and crops it, labels it. Sister project of [strata-360](../strata-360) (video): same structure, libraries and build.

## Quick start

```bash
py -3.13 -m venv .venv && .venv\Scripts\pip install -r requirements.txt   # once (Windows); Python 3.14 also works for the light dependencies
./stratastills doctor                                                        # checks the environment
./stratastills scan "\\Server\Drone\20250402 - Hazelmere" --info            # the shots in a folder: raw/JPEG pairs, panoramas, videos, gimbal pitch
```

## The visual report in the browser

`./stratastills serve --host 0.0.0.0` (the default 127.0.0.1 is this machine only) serves the app on port 8361 and, once `scripts/build_report.py` has written `out/report/`, the report at `/report/` (a "Visual report" link shows in the app's header; `STRATASTILLS_REPORT` or `report_dir` in the settings file point it elsewhere). From another machine open `http://<this machine>:8361/report/`. The server has no login: use it on a trusted network only.

## Speed

A 48 MP raw is developed, levelled, resized, sharpened and written in about 15 s on an NVIDIA RTX 3060 (about 95 s on the CPU). The pixel work runs on the GPU when there is one: NVIDIA CUDA or Apple silicon (MPS); with neither it runs on the CPU, with the same result (the CPU code is the reference the GPU code is tested against). Install PyTorch for your machine (`pip install torch`; on Windows or Linux with an NVIDIA card use `--index-url https://download.pytorch.org/whl/cu128`). `./stratastills doctor` shows which is in use.

## Documentation

| Document | What it is |
|---|---|
| [docs/overview.md](docs/overview.md) | The design brief: goals, what the first test shoot looks like, the pipeline. |
| [docs/implementation-plan.md](docs/implementation-plan.md) | What to build next, in order. |
| [docs/progress.md](docs/progress.md) | The log of what was tried, measured and decided. |
| [AGENTS.md](AGENTS.md) | Repository conventions and commands (for people and coding agents). |

## Tests

```bash
pip install -r requirements-test.txt
pytest tests/unit --cov                 # unit tests + coverage (CI gates on it)
pytest tests/integration -n auto        # CLI, pipelines and HTTP API on synthetic photos
ruff check src tests
cd web && npm run test:coverage:gate    # web tests against the coverage floors
```

See [tests/README.md](tests/README.md) for the layout and conventions. `scripts/setup_env.ps1` (Windows) and `scripts/setup_env.sh` (macOS, Linux) build the environment from scratch. CI runs on GitHub Actions (`.github/workflows/`).
