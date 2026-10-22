import { afterAll, afterEach, beforeAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { resetMock, server } from './server'

// jsdom: relative `/api/...` URLs resolve against jsdom's origin, so msw sees them as they are. An endpoint without a handler fails the test.
beforeAll(() => server.listen({ onUnhandledFrame: 'error' }))
afterEach(() => { server.resetHandlers(); cleanup(); resetMock(); window.location.hash = '' })
afterAll(() => server.close())
