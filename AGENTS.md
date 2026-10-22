# Repository Guidelines

Sister project of `../strata-360` (video): same structure, libraries, build and conventions, applied to still images. When in doubt, do what strata-360 does; port code from it rather than rewriting it, and note ports in `NOTICE`/`docs/progress.md`.

- Python application code lives in `src/stratastills/`; the React web app lives in `web/src/` and build into `src/stratastills/server/static/`, as in strata-360.
- Python tests live under `tests/`: `tests/unit/` (fast, isolated, synthetic data; no models, no real photos), `tests/integration/` (CLI/server on synthetic photos), `tests/utils/` (shared helpers). Real photos (`shoots/`, the drone folders on `\\Server\Drone`) are never required by tests and never committed.
- Commands (from the repo root, with `.venv` active or via `./stratastills` / `stratastills.cmd`):
  - `pip install -r requirements-test.txt` is enough for the test suites.
  - `pytest tests/unit --cov` (coverage floor `fail_under` in `pyproject.toml`; it only ratchets up), `ruff check src tests`.
  - `./stratastills doctor` checks the environment; `./stratastills scan <folder> [--info]` lists the shots in a folder.
- Cross-platform, Windows (NVIDIA GPU) first. OS differences go in one module (`oslib.py`, to be ported from strata-360 when needed); GPU/encoder choice in `hw.py`.
- Originals are read-only: nothing here ever writes into, renames or deletes files in a source folder. Output goes to a separate project/output folder.
- Style: dense, comment-the-why, long lines are fine (ruff line-length 400), pytest style tests.
- Docs live in `docs/`: `overview.md` (design brief), `implementation-plan.md` (what to build next, in order), `progress.md` (log of findings and decisions).
- Web (from `web/`): `npm install`, `npm run dev` (proxies /api to `./stratastills serve` on 8361), `npm run build` (writes into `src/stratastills/server/static/`, which `serve` serves), `npm run typecheck`, `npm run test:unit` (node), `npm run test:integration` (jsdom + msw: every endpoint the app calls needs a handler in `web/tests/utils/server.ts`).
- App shape: projects -> source folders -> contents (items) -> item (variants: pick the primary; transforms: original, levelled ...). Transforms live in `src/stratastills/transforms.py` (each has a VERSION in its cache key) and the frontend lists them in `web/src/api.ts` TRANSFORMS: add to both.
- Pipelines: `src/stratastills/pipeline.py` (stages, presets, generate, rules). A new stage needs an entry in `ORDER`, `PARAMS`, `version()` and a runner; its settings are part of the key of what it makes, so bump its version when its result changes. Generated pictures are variants of an item (`projects.generated_variants`); the web Generate panel shows the stage settings of the pipeline that will run from the server's `generate/check`, so the UI never duplicates the preset logic.
