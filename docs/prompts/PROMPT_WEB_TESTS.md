# Task: add thorough front-end tests to strata-stills (one PR, then stop)

You are a coding agent new to this repo. Find a coherent slice of **untested front-end code**, cover it with thorough, best-practice tests, and extend and refactor the shared test helpers as you go. Open **one PR**, drive CI to green, then stop. This prompt is run many times: each run continues from the state the last one left, so choose work from the *current* coverage report and leave the notes up to date.

## Unattended operation (read first)

You run without a human: nobody will answer questions, approve a plan or review before you open the PR. Never ask for clarification or confirmation, never stop to wait for input, and never end your turn with a question. When something is ambiguous, pick the most reasonable option, note the decision in the PR description and carry on. Specifically:

- If the branch name, the slice, a convention or a tool is unclear, decide using this prompt, `AGENTS.md`, the READMEs and the sister project `../strata-360`; record the choice under "Decisions" in the PR.
- If a command fails or a tool is missing (no GPU, no node, no network access to a registry), work around it or rely on CI as the runner; do not stop. If you truly cannot make progress on part of the slice, drop that part, list it under "Not done" in the PR, and finish the rest.
- If you find a bug or something out of scope, record it under "Findings" in the PR (as an xfail or `it.fails` where this prompt says so); don't ask whether to fix it.
- Always finish by opening the PR (or, if one already exists for your branch, pushing to it) and driving CI to green as described below; a PR with a clear "Not done" list beats no PR. Don't merge it, and don't wait for a reviewer.

## 0. Orient yourself (read before writing anything)

1. Read `AGENTS.md` and `CLAUDE.md` at the repo root. They are authoritative. Python code is out of scope: don't touch `src/stratastills/`, the Python `tests/`, `pyproject.toml` or the workflows. **This task runs in parallel with the backend unit-test and integration-test tasks on other branches**, which never touch `web/`; keep your changes inside `web/` (plus `package.json`/`package-lock.json` only if a new dev dependency is unavoidable) so nothing collides. Your notes live only in `web/tests/README.md`; don't edit `tests/README.md` or the other READMEs. This project is the sister of `../strata-360`, whose `web/tests/` has a richer set of helpers (`utils/api.ts`, `handlers.ts`, `factories.ts`, `media.ts`); port the ones you need rather than rewriting them, and say so in the PR.
2. Read **`web/tests/README.md`** in full, then everything in `web/tests/utils/` (`server.ts`: the one msw server and its default happy-path handlers; `setup.ts`: the jsdom setup) and the existing tests: `unit/logic.test.ts` (pure functions), `integration/App.test.tsx`, `integration/Generate.test.tsx`, `integration/Loupe.test.tsx` (components against the mocked backend). Copy their style.
3. Stack: React, TypeScript, Vite, vitest, Testing Library, user-event, msw. Source is in `web/src/` (`api.ts` is the typed client, with the `TRANSFORMS` list; the app is projects -> source folders -> contents -> item with variants and transforms, plus the Generate panel, whose stage settings come from the server's `generate/check`, so tests must not duplicate preset logic). The app calls the FastAPI backend through same-origin relative `/api/...` URLs.
4. Test projects: `unit` runs in node (`tests/unit/**`); `integration` runs in jsdom (`tests/integration/**`, setup `utils/setup.ts`). Commands, from `web/`: `npm ci` (or `npm install`), `npm run test:unit`, `npm run test:integration`, `npm run test:coverage:gate` (unit + integration together, what CI gates on; floors in `web/coverage-floor.json`), `npm run test:coverage:all` (no gate), `npm run test:coverage` (unit only, informational), `npm run typecheck` (tests are typechecked too), `npm run build`.
5. **GitHub Actions is the source of truth.** `unit.yml` runs the web typecheck, unit tests, the coverage gate and the build; `integration.yml` runs `npm run test:integration`. Run what you can locally, but the PR is done only when CI is green on the head commit.

## 1. Choose the slice (data-driven, so reruns don't collide)

1. There is no per-file ledger yet: **create the "Coverage status" table in `web/tests/README.md`** on your first run (file, line %, done/partial/todo, notes) from a fresh `npm run test:coverage:all`, and maintain it afterwards. The report wins over the table on conflict.
2. Check what's in flight: `git log --oneline -20`, `git branch -a`, and the open PRs (GitHub MCP tools). Don't pick files another branch or PR already covers. If an earlier run's PR for the same slice is still open, continue on it.
3. Pick **one cohesive slice**, about 3-6 related source files or 300-600 lines of source, from the files marked `todo`/`partial`, ranked by:
   - logic density (parsing, size and position math, zoom and pan in the Loupe, state machines, polling of generate jobs, error handling) over presentational markup;
   - interaction with the backend through `api.ts` (URL builders, query strings, JSON bodies, uploads, deletes);
   - bug-prone flows (choosing the primary variant, switching transforms, the Generate panel and its stage settings, rules, dialogs).
   Test your logic, not the browser's image decoding or canvas internals.
4. State the chosen slice and why at the top of the PR description.
5. **Stop condition:** if every file is `done` (>= 90% lines, or a justified gap noted in the table), change nothing in the code, open a PR whose only content is a one-line ledger/notes update saying so (or, if there is truly nothing to change, report it in your final message), and don't invent work.

## 2. Reuse and extend the shared helpers (don't bypass them)

- Every endpoint the app calls needs a default happy-path handler in `utils/server.ts`; a request without a handler fails the test loudly. When your slice hits a missing endpoint, add it there. Per-test variations (an error, a delay, a different payload) use `server.use(...)` inside the test; as the helper set grows, factor out in `utils/` what two test files share (typed factories with `Partial<T>` overrides for the interfaces in `src/api.ts`; `mockGet`/`mockError`/`recordRequests`-style helpers, ported from strata-360's `web/tests/utils/`). Never stub `fetch` by hand. When you touch an older test, move it onto the helpers.
- A new dev dependency needs a clear reason and a line in the PR (candidates: `vitest-axe` for accessibility checks and `fast-check` for pure math, both used in strata-360). Property tests suit the pure functions (clamping, zoom/pan limits, size formatting: invariants are easy to state).
- Gotchas: msw v3's option is `onUnhandledFrame`, not `onUnhandledRequest`; relative `/api` URLs work in jsdom, and need the base URL set in node; use `vi.useFakeTimers({ shouldAdvanceTime: true })` with user-event for debounces and polling; jsdom lacks `ResizeObserver`, `scrollIntoView`, canvas and image decoding, so stub them in `utils/` (one shared stub, not one per test).

## 3. How to write the tests

- **Behaviour, not implementation.** Query by role, label or text (`getByRole`, `findByText`). No class-name queries, no big snapshots. Add a `data-testid` only if there's no alternative, and mention it in the PR.
- One concept per `it`, named as a sentence. Arrange, act, assert.
- Cover, for each unit: happy path, loading state, empty state, server error (4xx/5xx with `{ detail }`), dropped connection, edge values (null/undefined optionals, empty lists, a picture without metadata, boundaries), and keyboard/mouse interaction.
- Assert the **outgoing requests** (method, URL, query, JSON body) by recording them in the handler.
- Accessibility: one axe check per component where it's cheap. If it flags a violation in existing app code, don't fix the app here: add an `it.todo`/`it.fails` and report it under "Findings".
- Deterministic and independent: no sleeps (`findBy*`, `vi.waitFor`, fake timers), no shared mutable state, no real network.
- Placement: pure/no-DOM -> `tests/unit/<name>.test.ts`; renders React or needs jsdom -> `tests/integration/<Name>.test.tsx`. One test file per source file where practical.
- **Don't change `web/src` behaviour.** Allowed: a tiny testability change (export a pure function, add an accessible label), called out in the PR. When you find a real bug, document it with `it.todo`/`it.fails` plus a comment, list it under "Findings", and don't fix it in this PR. Never skip, `.only` or delete a test, and never lower a threshold to get green.

## 4. Update the notes (this is what makes reruns work)

- In `web/tests/README.md`, update the "Coverage status" row of every file you touched (line %, `done`/`partial`/`todo`, notes) and add rows for new files. Take the numbers from `npm run test:coverage:all` or the CI log, not an estimate.
- Keep the README's layout, helper list and conventions current. If you added a helper or learned a gotcha, write it down.
- Aim for >= 90% lines and >= 85% branches on files in your slice. Justify any gap in the table.
- **Ratchet:** you are the only task that edits `web/coverage-floor.json`. After CI is green, raise each floor (lines, statements, functions, branches) to the integer floor of the new totals from the gate step's report, in your last commit. Floors only go up: never lower one to get green, and never add a threshold anywhere else.

## 5. Verify locally (if node is available)

From `web/`: `npm ci`, `npm run typecheck`, `npm run test:unit`, `npm run test:integration`, `npm run test:coverage:all`, `npm run build`. Run the integration suite 3 times to catch flakiness. Mutation spot-check: break the source (flip a condition) for 2-3 representative tests, confirm they fail, then revert. Re-read your diff adversarially for tests that would still pass if the component were broken. If you can't run node locally, push early and use CI as your runner.

## 6. Branch, PR, and drive CI to green

- Use the branch name the environment gives you (otherwise `test/frontend-coverage-<slice>` off the latest default branch). Make small logical commits (helpers -> tests per file -> notes). Push with `git push -u origin <branch>`.
- Open **one PR** (follow its PR template if the repo has one) as soon as the first meaningful commit is pushed. The PR triggers `unit.yml` and `integration.yml`. Keep it a draft until CI is green if you can.
- Loop until the **web** jobs of "Unit tests" and "Integration tests" are green on the latest head commit:
  1. Read the results with the GitHub MCP tools (`actions_list`, `actions_get`, `get_job_logs`).
  2. If red, reproduce, root-cause, push a minimal fix, repeat.
  3. If a failure is clearly not yours (a Python job, or an infrastructure error naming something your diff doesn't touch), say so in one PR comment with the evidence and re-run at most once. Never mask a flake. Never skip tests or push an empty commit to retrigger CI.
  4. CI-only failures to anticipate: `package-lock.json` out of sync (commit a lockfile generated with the Node version CI uses), timing-sensitive tests on slower runners (use `findBy*`/fake timers), and type errors in test files (tests are typechecked).
- The PR description must include:
  - the slice and why, and what's left for next runs;
  - coverage before -> after for each target file, taken from CI;
  - new dev dependencies and why, and the helpers/handlers/factories added or ported, plus which older tests were refactored onto them;
  - **Findings**: bugs, a11y violations, untestable areas, and any `web/src` changes (should be minimal).
- Finish by stating plainly the CI status on the head commit, the PR link, and what remains. Don't merge the PR. Don't start a second slice. If review comments or CI events arrive on your PR, handle them as part of this same PR.
