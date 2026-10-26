# Web tests

vitest 4, Testing Library, user-event, msw (backend mocks). Run from `web/`: `npm run test:unit`, `npm run test:integration`, `npm run test:coverage:gate` (unit + integration together, the number CI gates on: floors in `web/coverage-floor.json`, which only ratchet up), `npm run test:coverage:all` (the same run without the gate), `npm run test:coverage` (unit only, informational), `npm run typecheck`, `npm run build`.

## Layout

- `unit/`: node environment, no DOM. Pure functions (`logic.test.ts`).
- `integration/`: jsdom. Components rendered with Testing Library against a mocked backend (`App`, `Generate`, `Loupe`).
- `utils/`: `server.ts` is the one msw server and its default happy-path handlers: **every endpoint the app calls needs a handler there** (an unmocked request fails the test); `setup.ts` is the jsdom setup.
- One test file per area of the app; add to the existing file if there is one.

## Conventions

- Test what a user sees: query by role, label or text; no class-name queries, no snapshots.
- One concept per `it`, named as a sentence. Cover the happy path, loading, empty, a server error and the edge values.
- No arbitrary sleeps: `findBy*`, `waitFor`, or fake timers.
- Render the real children and let msw answer their requests; mock a module only for something jsdom cannot run.
- A bug found while testing: write `it.todo` or `it.fails`, comment it as a finding, and don't change app behaviour in a test-only change.

## Coverage floors

`coverage-floor.json` holds the floors of `npm run test:coverage:gate` (lines, statements, functions, branches). After CI is green, raise them to the new totals; never lower them.

## Coverage status

| File | % Lines | Status | Notes |
|---|---|---|---|
| `src/App.tsx` | 100 | done | Covered by top-level integration tests |
| `src/api.ts` | 100 | done | Covered through handlers in integration tests |
| `src/route.ts` | 100 | done | Tested indirectly via navigation |
| `src/components/AutoGenerate.tsx` | 95.65 | done | Handled in integration/AutoGenerate.test.tsx |
| `src/components/GeneratePanel.tsx` | 98.21 | done | Generate.test.tsx: the panel, the background job's polling and its error, saving and deleting a preset |
| `src/components/ItemPage.tsx` | 94.73 | done | Tested in ItemPage.test.tsx |
| `src/components/Loupe.tsx` | 100 | done | Handled in Loupe.test.tsx |
| `src/components/ProjectList.tsx` | 100 | done | ProjectList.test.tsx: list, create (with and without a first source), errors, plural |
| `src/components/ProjectPage.tsx` | 100 | done | ProjectPage.test.tsx handles this |
| `src/components/SourcePage.tsx` | 100 | done | SourcePage.test.tsx handles this |
| `src/components/ui.tsx` | 100 | done | Util components |
