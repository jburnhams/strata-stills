// The route tree, shared by main.tsx (createHashRouter) and the tests (strata-react/testing renderRoutes), so the tests run the app's own routes.
import { Navigate, Outlet, type RouteObject } from 'react-router'
import { StrataToastRegion } from 'strata-react'
import Shell from './App'
import ProjectList from './components/ProjectList'
import ProjectPage from './components/ProjectPage'
import SourcePage from './components/SourcePage'
import ItemPage from './components/ItemPage'
import ArenaPage from './components/ArenaPage'
import AlertsPage from './components/AlertsPage'
import FlaggedPage from './components/FlaggedPage'
import BrowserPage from './components/BrowserPage'

/** Every screen, and the one place toasts appear (strata-react: once at the app root). */
function Root() {
  return <><Outlet /><StrataToastRegion /></>
}

export const routes: RouteObject[] = [
  {
    element: <Root />,
    children: [
      {
        path: '/',
        element: <Shell />,
        children: [
          { index: true, element: <ProjectList /> },
          { path: 'p/:project', element: <ProjectPage /> },
          { path: 'p/:project/s/:source', element: <SourcePage /> },
          { path: 'p/:project/i/:item', element: <ItemPage /> },
          { path: 'alerts', element: <AlertsPage /> },
          { path: 'flagged', element: <FlaggedPage /> },
          { path: 'browse', element: <BrowserPage /> },
        ],
      },
      // The arena is a screen of its own, outside the shell: the two pictures get the whole width.
      { path: '/arena', element: <ArenaPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
]
