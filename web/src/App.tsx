import { useEffect, useState } from 'react'
import { api, withBase } from './api'
import { useRoute, href } from './route'
import ProjectList from './components/ProjectList'
import ProjectPage from './components/ProjectPage'
import SourcePage from './components/SourcePage'
import ItemPage from './components/ItemPage'

export default function App() {
  const r = useRoute()
  const [report, setReport] = useState({ neutral: false, auto: false })
  useEffect(() => { api.report().then(x => setReport({ neutral: !!x.neutral, auto: !!x.auto })).catch(() => {}) }, [])
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <header className="mb-5 flex items-baseline gap-3">
        <a href={href.projects()} className="text-xl font-semibold">strata-stills</a>
        {report.neutral && <a href={withBase('/report/neutral/')} target="_blank" rel="noreferrer" className="text-sm text-blue-600 dark:text-blue-400 ml-auto">Neutral report</a>}
        {report.auto && <a href={withBase('/report/auto/')} target="_blank" rel="noreferrer" className={`text-sm text-blue-600 dark:text-blue-400 ${report.neutral ? '' : 'ml-auto'}`}>Auto report</a>}
      </header>
      {r.page === 'projects' && <ProjectList />}
      {r.page === 'project' && <ProjectPage project={r.project} />}
      {r.page === 'source' && <SourcePage project={r.project} source={r.source} />}
      {r.page === 'item' && <ItemPage project={r.project} item={r.item} source={r.source} />}
    </div>
  )
}
