import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createHashRouter, RouterProvider } from 'react-router'
import { StrataQueryProvider, ThemeProvider } from 'strata-react'
import './app.css'
import { routes } from './routes'

// Hash routing: the server serves this one page at its root (and behind Caddy at /strata-stills/), and /arena/, /alerts/ and /flagged/ send the browser here (app.py).
const router = createHashRouter(routes)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <StrataQueryProvider>
        <RouterProvider router={router} />
      </StrataQueryProvider>
    </ThemeProvider>
  </StrictMode>,
)
