import { renderRoutes } from 'strata-react/testing'
import { routes } from '../../src/routes'

/** Renders the app's own route tree at a route (strata-react renderRoutes: router, query client and theme as the app has them). */
export const open = (route: string) => renderRoutes(routes, { route })
