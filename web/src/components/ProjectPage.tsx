import { useState } from 'react'
import { api } from '../api'
import { href, useLoad } from '../route'
import { Card, Crumbs, Err, btn, btnQuiet, input, projectCrumbs } from './ui'

export default function ProjectPage({ project }: { project: string }) {
  const proj = useLoad(() => api.project(project), [project])
  const src = useLoad(() => api.sources(project), [project])
  const [path, setPath] = useState(''), [err, setErr] = useState<string | null>(null)
  const act = async (f: () => Promise<unknown>) => { setErr(null); try { await f(); src.reload(); proj.reload() } catch (x) { setErr((x as Error).message) } }
  return (
    <section>
      <Crumbs items={[...projectCrumbs(project, proj.data?.name).slice(0, 1), { label: proj.data?.name || project }]} />
      <h1 className="text-lg font-semibold mb-3">Source folders</h1>
      <Err msg={proj.error || src.error} />
      {src.data && src.data.length === 0 && <p className="text-stone-500 mb-3">This project has no source folders yet. Add one below.</p>}
      <div className="grid gap-3 sm:grid-cols-2">
        {src.data?.map(s => (
          <Card key={s.id} to={s.reachable ? href.source(project, s.id) : undefined}>
            <div className="font-mono text-sm break-all">{s.path}</div>
            {s.reachable
              ? <div className="text-sm text-stone-500 mt-1">{s.stills} stills{s.panos ? `, ${s.panos} panoramas` : ''}{s.videos ? `, ${s.videos} videos` : ''}</div>
              : <div className="text-sm text-red-600 mt-1">Not reachable (is the share on?)</div>}
            <button className={btnQuiet + ' mt-2'} onClick={e => { e.preventDefault(); act(() => api.removeSource(project, s.path)) }}>Remove from project</button>
          </Card>
        ))}
      </div>
      <form className="flex gap-2 mt-5" onSubmit={e => { e.preventDefault(); act(async () => { await api.addSource(project, path.trim()); setPath('') }) }}>
        <input className={input + ' flex-1'} aria-label="Source folder" placeholder="Add a source folder, e.g. \\Server\Drone\20250402" value={path} onChange={e => setPath(e.target.value)} required />
        <button className={btn}>Add source</button>
      </form>
      <Err msg={err} />
    </section>
  )
}
