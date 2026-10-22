import { useState } from 'react'
import { api, imageUrl, type Item } from '../api'
import { href, useLoad } from '../route'
import { Crumbs, Err, projectCrumbs } from './ui'

const BADGE: Record<string, string> = { pano: 'bg-purple-600', video: 'bg-stone-600', still: 'bg-blue-600' }

export function Thumb({ project, item }: { project: string; item: Item }) {
  return (
    <a href={href.item(project, item.id)} className="group block rounded-lg overflow-hidden border border-stone-300 dark:border-stone-700 bg-white dark:bg-stone-900 hover:border-blue-500">
      <div className="aspect-[4/3] bg-stone-200 dark:bg-stone-800 flex items-center justify-center text-stone-500 text-sm">
        {item.kind === 'video' ? <span>video</span> : <img loading="lazy" alt={item.stem} className="w-full h-full object-cover" src={imageUrl(project, item.id, item.primary, 'original', 360)} />}
      </div>
      <div className="p-2 text-xs">
        <div className="font-mono truncate" title={item.stem}>{item.stem.replace(/^DJI_/, '')}</div>
        <div className="flex gap-1 mt-1 flex-wrap">
          <span className={`text-white rounded px-1 ${BADGE[item.kind]}`}>{item.kind}</span>
          {item.variants.map(v => <span key={v.id} className={`rounded px-1 border ${v.id === item.primary ? 'border-blue-500 text-blue-600' : 'border-stone-300 dark:border-stone-600 text-stone-500'}`}>{v.ext.slice(1)}</span>)}
        </div>
      </div>
    </a>
  )
}

export default function SourcePage({ project, source }: { project: string; source: string }) {
  const proj = useLoad(() => api.project(project), [project])
  const srcs = useLoad(() => api.sources(project), [project])
  const items = useLoad(() => api.items(project, source), [project, source])
  const [videos, setVideos] = useState(false)
  const shown = items.data?.filter(i => videos || i.kind !== 'video') ?? []
  const nVideo = items.data?.filter(i => i.kind === 'video').length ?? 0
  const path = srcs.data?.find(s => s.id === source)?.path
  return (
    <section>
      <Crumbs items={[...projectCrumbs(project, proj.data?.name), { label: path ?? 'Source' }]} />
      <h1 className="text-lg font-semibold">Contents</h1>
      {path && <p className="text-xs text-stone-500 mb-2 font-mono break-all">{path}</p>}
      <Err msg={items.error} />
      {items.data && nVideo > 0 && <label className="text-sm text-stone-500 flex items-center gap-2 mb-3"><input type="checkbox" checked={videos} onChange={e => setVideos(e.target.checked)} />Show {nVideo} videos (not processed yet)</label>}
      {!items.data && !items.error && <p className="text-stone-500">Loading…</p>}
      <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
        {shown.map(i => <Thumb key={i.id} project={project} item={i} />)}
      </div>
    </section>
  )
}
