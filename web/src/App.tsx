import { useRoute, href } from './route'
import ProjectList from './components/ProjectList'
import ProjectPage from './components/ProjectPage'
import SourcePage from './components/SourcePage'
import ItemPage from './components/ItemPage'

export default function App() {
  const r = useRoute()
  return (
    <div className="max-w-6xl mx-auto px-4 py-6">
      <header className="mb-5 flex items-baseline gap-3">
        <a href={href.projects()} className="text-xl font-semibold">strata-stills</a>
      </header>
      {r.page === 'projects' && <ProjectList />}
      {r.page === 'project' && <ProjectPage project={r.project} />}
      {r.page === 'source' && <SourcePage project={r.project} source={r.source} />}
      {r.page === 'item' && <ItemPage project={r.project} item={r.item} source={r.source} />}
    </div>
  )
}
