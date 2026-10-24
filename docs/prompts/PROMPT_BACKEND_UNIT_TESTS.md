# Task: add thorough backend unit tests to strata-stills (one PR, then stop)

You are a coding agent new to this repo. Find a coherent slice of **under-tested Python code**, cover it with thorough, best-practice unit tests, and extend and refactor the shared test helpers as you go. Open **one PR**, drive CI to green, then stop. This prompt is run many times: each run continues from the state the last one left, so choose work from the *current* coverage ledger and leave it up to date.

## Unattended operation (read first)

You run without a human: nobody will answer questions, approve a plan or review before you open the PR. Never ask for clarification or confirmation, never stop to wait for input, and never end your turn with a question. When something is ambiguous, pick the most reasonable option, note the decision in the PR description and carry on. Specifically:

- If the branch name, the slice, a convention or a tool is unclear, decide using this prompt, `AGENTS.md`, the READMEs and the sister project `../strata-360`; record the choice under "Decisions" in the PR.
- If a command fails or a tool is missing (no GPU, no node, no network access to a registry), work around it or rely on CI as the runner; do not stop. If you truly cannot make progress on part of the slice, drop that part, list it under "Not done" in the PR, and finish the rest.
- If you find a bug or something out of scope, record it under "Findings" in the PR (as an xfail or `it.fails` where this prompt says so); don't ask whether to fix it.
- Always finish by opening the PR (or, if one already exists for your branch, pushing to it) and driving CI to green as described below; a PR with a clear "Not done" list beats no PR. Don't merge it, and don't wait for a reviewer.

## 0. Orient yourself (read before writing anything)

1. Read `AGENTS.md` and `CLAUDE.md` at the repo root. They are authoritative. Web code (`web/`) is out of scope for this task. This project is the sister of `../strata-360`: if a convention is unclear here, strata-360 shows how it was solved.
2. Read **`tests/README.md`** (shared commands, hermetic setup, file-ownership rules) and **`tests/unit/README.md`** (unit conventions and the "Where things are tested" table, which is the ledger) in full. **This task runs in parallel with the integration-test and web-test tasks on other branches, so obey the ownership table in `tests/README.md`**: you may edit `tests/unit/**`, `tests/conftest.py`, `tests/unit/conftest.py`, `tests/utils/<topic>.py` (not `integration_*.py`), `tests/unit/README.md` and the `fail_under` line in `pyproject.toml`. Don't touch `tests/integration/**`, `tests/README.md`, `web/`, `requirements-test.txt` (unless a new test dependency is unavoidable, and say so in the PR) or the workflows. Then read `tests/conftest.py`, `tests/utils/photos.py` and the worked examples: `tests/unit/test_gpu.py` (CPU/GPU equivalence, `synthetic_dng`), `test_pipeline.py`, `test_server.py`. Copy their style.
3. Layout: application code is in `src/stratastills/` (`develop.py` and `detail.py` are the NumPy reference, `gpu.py`/`gpu_tiled.py`/`cpu_tiled.py` the fast paths, `lineardng.py`, `inputs.py`, `match.py`, `tonematch.py`, `lenscorr.py`, `lensfundb.py`, `denoise_ai.py`, `jpeg_quality.py`, `pipeline.py`, `transforms.py`, `projects.py`, `ingest.py`, `meta.py`, `preview.py`, `export.py`, `fullres.py`, `images.py`, `settings.py`, `sizeplan.py`, `resources.py`, `cli.py`, `server/app.py`). Unit tests are in `tests/unit/`, fast and hermetic (no models, no real photos, no network, no real home folder: enforced by the autouse fixtures in `tests/conftest.py`). Integration tests (`tests/integration/`) are out of scope except to avoid breaking them.
4. Setup and commands, from the repo root: `pip install -r requirements-test.txt`, then
   - `pytest tests/unit --cov --cov-report=term-missing --cov-report=xml` (coverage gate: `fail_under` in `pyproject.toml`)
   - `pytest tests/unit -n auto` (parallel), `pytest tests/unit -p randomly --randomly-seed=N` (reproduce an order)
   - `ruff check src tests`
   - `diff-cover coverage.xml --compare-branch origin/main` (coverage of just your changed lines)
5. **GitHub Actions is the source of truth.** `unit.yml` runs the unit suite with coverage and ruff on Python 3.11-3.13 on Linux plus macOS and Windows (all blocking), and the web checks. The PR is done only when CI is green on the head commit. Code must be cross-platform (Windows first): build paths with `os.path`/`pathlib`, don't assume symlinks or POSIX permissions (skip with a reason), and put OS or GPU differences only where AGENTS.md says (`oslib.py`, `hw.py`, `gpu.py`).

## 1. Choose the slice (data-driven, so reruns don't collide)

1. Open the ledger in `tests/unit/README.md` and cross-check it against a fresh `pytest tests/unit --cov --cov-report=term-missing` (or the latest CI log). The report wins on conflict. If the ledger lists modules, not percentages, add the percentages as you go (see section 4).
2. Check what's in flight: `git log --oneline -20`, `git branch -a`, and the open PRs (GitHub MCP tools; the integration task has its own ledger in `tests/integration/README.md`). Don't pick modules another branch or PR already covers. If an earlier run's PR for the same slice is still open, continue on it.
3. Pick **one cohesive slice**, about 2-5 related modules or 300-600 statements of missing coverage, ranked by:
   - logic density testable without heavy dependencies (colour and tone maths, EXIF/XMP/DNG parsing, lens-model and lensfun lookup, noise tables and ISO interpolation, rules and stage keys, sizing and tile planning, project files, path handling, HTTP endpoints through `TestClient`);
   - pure or near-pure code at 0-50%;
   - bug-prone areas: cache keys and versions (a stage whose settings change must change its key), the "originals are read-only" guarantee (nothing may write into, rename or delete in a source folder), path security in the server, concurrency (GPU slots, workers), `auto` rules that choose a treatment from picture data, boundaries (zero-size, 1-pixel, odd sizes, NaN/Inf, 16-bit extremes, missing metadata).
   Test the pure parts of modules that need torch, a GPU, a model or LibRaw, and fake the rest at the boundary (`monkeypatch` on the single function that loads a model or reads a raw; `pytest.importorskip('torch')`). A change to a CPU reference in `develop.py`/`detail.py` and its mirror in `gpu.py` is held to the equivalence tests: new `gpu.py` coverage goes in `test_gpu.py` as an equivalence test against the CPU original, with mean and 99.9th-percentile bounds as tight as they pass.
4. State the chosen slice and why at the top of the PR description.
5. **Stop condition:** if every module is `done` (>= 90%, or a justified gap in the ledger), change nothing in the code, open a PR whose only content is a one-line ledger/notes update saying so (or, if there is truly nothing to change, report it in your final message), and don't invent work.

## 2. Reuse and extend the shared helpers (don't bypass them)

- Use `tmp_path` and `monkeypatch`, not `tempfile.mkdtemp()` or assignment to module globals. Build pictures with `tests/utils/photos.py` (`make_jpeg`, `make_linear_dng`, `make_shoot`) or with synthetic arrays; never real photos (`shoots/`, `candidates/`, the drone folders).
- If a helper is missing, add it: a fake for a model loader or for `rawpy`/LibRaw in `tests/conftest.py` (and a test for the fake), a builder next to `photos.py` in `tests/utils/<topic>.py` (a gradient/chart generator, a profile bundle builder, a fake EXIF dict). Anything two test files share moves into `tests/utils/`.
- Add a new test dependency only for a clear reason, put it in `requirements-test.txt`, and say why in the PR.
- When you touch an older test file, bring it to the house style: pytest functions/classes with plain `assert`, `monkeypatch`/`tmp_path`, no `sys.path` hacks, no `__main__` runner, no module-global assignments. Preserve every assertion's meaning. Don't convert files outside your slice.
- Use `pytest-randomly` as a bug-finder: run your slice and the whole suite under several seeds (`--randomly-seed=1..5`) and with `-n 4`. An order-dependent failure is a leaked global (the tuning variables `STRATASTILLS_*` are removed per test; `gpu._slots` must be reset by the test that sets it): fix the cause with `monkeypatch` and mention it in the PR.

## 3. How to write the tests

- Test behaviour through the public function (or the HTTP endpoint via `TestClient`), asserting exact values or tight numeric bounds. Cover for each unit: the happy path, empty/missing input, malformed input (truncated file, bad XMP, NaNs), boundaries (zero, one, many; exact limits), each error branch, and idempotence/repeat calls. Use `--cov-report=term-missing` to find unreached lines, but don't write a test purely to touch a line without asserting something meaningful.
- `@pytest.mark.parametrize` for tables of cases; classes to group related tests; names that read as a sentence (`test_a_stage_setting_change_changes_its_cache_key`).
- **hypothesis** for numeric code where an invariant is easy to state (output in [0, 1], monotonic tone curve, round-trip of a colour transform, tile split then merge equals whole, never raises on any input). Keep strategies bounded and fast.
- Assert side effects on the filesystem via `tmp_path`; assert that a source folder is byte-for-byte untouched after an operation. Tests must pass alone, in any order and in parallel, and never touch the real home folder, repo files, the network, or `time.sleep`.
- Don't change `src/` behaviour in a test-only PR. Allowed: tiny testability changes (a parameter with a default, extracting a pure function), called out in the PR. For a bug you find, write the test for the *correct* behaviour with `@pytest.mark.xfail(strict=True, reason='...')`, and list it under "Findings". Never skip, delete, or weaken an existing test, and never lower `fail_under`.

## 4. Update the ledger and the ratchet

- In `tests/unit/README.md`, update the "Where things are tested" rows for every module you touched (add the coverage percent from CI or the report, `done`/`partial`/`todo`, notes on what remains), and keep its conventions and known issues current. Document a new shared fixture or helper there, under a short "Added by unit work" heading, rather than editing `tests/README.md`, which stays untouched to avoid merge conflicts.
- You are the only task that edits `fail_under`. After CI is green and the total coverage is known, raise `fail_under` in `pyproject.toml` to the floor of the new total, rounded down (it only ratchets up). Do this in your last commit. Mind that coverage is measured on CI's runners, where CUDA/MPS tests skip: take the number from the CI log, not from a machine with a GPU.
- Aim for >= 90% (line + branch) on the modules in your slice. Justify any gap in the ledger.

## 5. Verify locally

`pip install -r requirements-test.txt`, `ruff check src tests`, `pytest tests/unit --cov --cov-report=term-missing --cov-report=xml`, the suite under three random seeds and with `-n 4`, `diff-cover coverage.xml --compare-branch origin/main`, and `pytest tests/integration -x` if you touched shared fixtures or `tests/conftest.py`. Mutation spot-check: break the source (flip a condition) for 2-3 representative tests, confirm they fail, then revert. Re-read your diff adversarially for tests that would still pass if the code were broken.

## 6. Branch, PR, and drive CI to green

- Use the branch name the environment gives you (otherwise `test/backend-coverage-<slice>` off the latest default branch). Make small logical commits (helpers -> tests per module -> ledger/ratchet). Push with `git push -u origin <branch>`.
- Open **one PR** (follow its PR template if the repo has one) as soon as the first meaningful commit is pushed; keep it a draft until CI is green if you can.
- Loop until CI is green on the latest head commit:
  1. Read results with the GitHub MCP tools (`actions_list`, `actions_get`, `get_job_logs`).
  2. If red, reproduce, root-cause, push a minimal fix, repeat. macOS- or Windows-only failures are usually path, line-ending, symlink/permission or float32 tolerance assumptions: fix them in the test.
  3. If a failure is clearly not yours (the web jobs, or an infrastructure error), say so in one PR comment with the evidence and re-run at most once. Never mask a flake, skip tests, or push an empty commit to retrigger CI.
- The PR description must include: the slice and why (and what's left); coverage before -> after per module and overall (from CI); new dependencies/fixtures/helpers and which older tests were converted; "Findings" (bugs as xfails, order-dependence fixes, risky or untestable areas, any `src/` change).
- Finish by stating the CI status on the head commit, the PR link, and what remains. Don't merge. Don't start a second slice. If review comments or CI events arrive on your PR, handle them as part of this same PR.
