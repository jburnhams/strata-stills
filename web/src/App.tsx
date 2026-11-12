// The app's frame: strata-react's AppShell (side nav, current page as a pill) around every screen but the arena.
// The top bar holds what every screen uses: the reports, when the server has them, and the light/dark choice (there is no settings page, so it is the compact one).
import { Outlet } from 'react-router'
import { AppShell, ColorSchemeChoice, useLoad } from 'strata-react'
import { api, withBase } from './api'
import { href } from './route'

const nav = [
  { to: href.projects(), label: 'Projects', end: true },
  { to: href.arena(), label: 'Arena' },
  { to: href.alerts(), label: 'Arena flags' },
  { to: href.flagged(), label: 'Flagged pictures' },
]

/** Links to the visual reports that exist (a failed read shows none: they are optional). */
export function ReportLinks() {
  const report = useLoad({ queryKey: ['report'], queryFn: api.report })
  if (report.status !== 'ready') return null
  return (
    <>
      {report.data.neutral && <a href={withBase('/report/neutral/')} target="_blank" rel="noreferrer" className="type-body-sm">Neutral report</a>}
      {report.data.auto && <a href={withBase('/report/auto/')} target="_blank" rel="noreferrer" className="type-body-sm">Auto report</a>}
    </>
  )
}

export default function Shell() {
  return (
    <AppShell product="strata-stills" nav={nav} topBar={<><ReportLinks /><ColorSchemeChoice compact className="ml-auto" /></>}>
      <Outlet />
    </AppShell>
  )
}
