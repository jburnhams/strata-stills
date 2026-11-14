import { afterAll, afterEach, beforeAll } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { installLayoutShims } from 'strata-react/testing'
import { toast } from 'strata-react'
import { resetBrowse, resetMock, server } from './server'
import { images, installImages } from './images'

// jsdom: relative `/api/...` URLs resolve against jsdom's origin, so msw sees them as they are. An endpoint without a handler fails the test.
// strata-react's layout shims give every element a box and a ResizeObserver, so its measuring components (tables, the compare frames) render.
installLayoutShims()
installImages()
beforeAll(() => server.listen({ onUnhandledFrame: 'error' }))
afterEach(() => { server.resetHandlers(); cleanup(); resetMock(); resetBrowse(); toast.clear(); images.release() })
afterAll(() => server.close())
