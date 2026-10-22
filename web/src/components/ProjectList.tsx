import { useState } from 'react'
import { api } from '../api'
import { href, useLoad } from '../route'
import { Card, Err, btn, input } from './ui'

export default function ProjectList() {
  const { data, error, reload } = useLoad(() => Promise.all([api.projects(), api.settings()]), [])
  const [name, setName] = useState(''), [src, setSrc] = useState(''), [err, setErr] = useState<string | null>(null)
  const create = async (e: React.FormEvent) => {
    e.preventDefault(); setErr(null)
    try { const p = await api.createProject(name.trim(), src.trim() ? [src.trim()] : []); window.location.hash = href.project(p.id); setName(''); setSrc(''); reload() } catch (x) { setErr((x as Error).message) }
  }
  const [projects, settings] = data ?? [null, null]
  return (
    <section>
      <h1 className="text-lg font-semibold">Projects</h1>
      {settings && <p className="text-xs text-stone-500 mb-3">Projects folder: <span className="font-mono">{settings.projects_dir}</span></p>}
      <Err msg={error} />
      {projects && projects.length === 0 && <p className="text-stone-500 my-4">No projects yet. Create one below.</p>}
      <div className="grid gap-3 sm:grid-cols-2">
        {projects?.map(p => (
          <Card key={p.id} to={href.project(p.id)}>
            <div className="font-medium">{p.name}</div>
            <div className="text-xs text-stone-500">{p.n_sources} source folder{p.n_sources === 1 ? '' : 's'}</div>
            {p.sources.map(s => <div key={s.path} className="font-mono text-xs text-stone-500 break-all mt-1">{s.path}</div>)}
          </Card>
        ))}
      </div>
      <form onSubmit={create} className="flex flex-wrap gap-2 mt-6">
        <input className={input + ' flex-1 min-w-48'} aria-label="Project name" placeholder="New project name" value={name} onChange={e => setName(e.target.value)} required />
        <input className={input + ' flex-1 min-w-48'} aria-label="First source folder" placeholder="First source folder (optional)" value={src} onChange={e => setSrc(e.target.value)} />
        <button className={btn}>Create project</button>
      </form>
      <Err msg={err} />
    </section>
  )
}
