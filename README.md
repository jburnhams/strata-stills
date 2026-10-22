# strata-stills

Takes raw(ish) still photos (DJI drone DNG + JPEG today; other cameras to follow) and makes them look their best automatically: picks the source, develops the raw, levels and straightens, finds what is in the picture, frames and crops it, labels it. Sister project of [strata-360](../strata-360) (video): same structure, libraries and build.

## Quick start

```bash
py -3.13 -m venv .venv && .venv\Scripts\pip install -r requirements.txt   # once (Windows); Python 3.14 also works for the light dependencies
./stratastills doctor                                                        # checks the environment
./stratastills scan "\\Server\Drone\20250402 - Hazelmere" --info            # the shots in a folder: raw/JPEG pairs, panoramas, videos, gimbal pitch
```

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
pytest tests/unit --cov
ruff check src tests
```
