// One question: "Which picture of this folder do I look at?" A grid of thumbnails (the cached ones the Browser shares, a raw's embedded preview; our develop only if that fails) (strata-react Thumbnail: a box of fixed shape with a skeleton, words when the picture is missing,
// and a request only near the screen, so a folder of hundreds does not ask for all of them at once). Videos are not processed yet, so they are hidden behind a switch.
import { useState } from 'react'
import { Link } from 'react-router'
import { Breadcrumb, DataView, EmptyState, PageHeader, Status, Switch, Thumbnail, ThumbnailSkeleton, format, useLoad, type Tone } from 'strata-react'
import { api, imageUrl, itemThumbUrl, type Item } from '../api'
import { href, useIds } from '../route'
import AutoGenerate from './AutoGenerate'

const KIND: Record<Item['kind'], { word: string; tone: Tone }> = { still: { word: 'Still', tone: 'neutral' }, pano: { word: 'Panorama', tone: 'info' }, video: { word: 'Video', tone: 'neutral' } }
/** The short names of an item's variants, the primary first: "raw", "jpg", "gen". */
export const variantWords = (i: Item) => [...i.variants].sort((a, b) => Number(b.id === i.primary) - Number(a.id === i.primary)).map(v => (v.kind === 'generated' ? 'gen' : v.ext.replace(/^\./, '')))

export function Thumb({ project, item, eager }: { project: string; item: Item; eager?: boolean }) {
  return (
    <Link to={href.item(project, item.id)} className="grid gap-1.5 rounded-surface border border-border p-1.5 no-underline transition-state hover:border-muted-foreground">
      <Thumbnail src={item.kind === 'video' || !item.primary ? null : [itemThumbUrl(project, item.id), imageUrl(project, item.id, item.primary, 'original', 360)]} alt={item.stem} aspect="4/3" missing={item.kind === 'video' ? 'Video' : 'No picture'} eager={eager} className="w-full rounded-control" />
      <span className="truncate font-mono type-caption text-foreground" title={item.stem}>{item.stem.replace(/^DJI_/, '')}</span>
      <span className="flex flex-wrap gap-1"><Status tone={KIND[item.kind].tone}>{KIND[item.kind].word}</Status>{item.variants.length > 0 && <span className="type-caption text-muted-foreground">{variantWords(item).join(' · ')}</span>}</span>
    </Link>
  )
}

export default function SourcePage() {
  const { project, source } = useIds()
  const proj = useLoad({ queryKey: ['project', project], queryFn: () => api.project(project) })
  const srcs = useLoad({ queryKey: ['sources', project], queryFn: () => api.sources(project) })
  const items = useLoad({ queryKey: ['items', project, source], queryFn: () => api.items(project, source) })
  const [videos, setVideos] = useState(false)
  const path = srcs.status === 'ready' ? srcs.data.find(s => s.id === source)?.path : undefined
  const name = proj.status === 'ready' ? proj.data.name : project
  const nVideo = items.status === 'ready' ? items.data.filter(i => i.kind === 'video').length : 0
  const grid = 'grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5'

  return (
    <section className="grid gap-6">
      <Breadcrumb items={[{ label: 'Projects', to: href.projects() }, { label: name, to: href.project(project) }, { label: path ?? 'Contents' }]} />
      <PageHeader title="Contents" scope={path ? <span className="font-mono break-all">{path}</span> : undefined} />
      <AutoGenerate project={project} source={source} />
      {nVideo > 0 && <Switch label={`Show ${format.count(nVideo, 'video')}`} detail="Videos are not processed yet." on={videos} onChange={setVideos} />}
      <DataView
        state={items}
        isEmpty={rows => rows.length === 0}
        skeleton={<div className={grid}>{[0, 1, 2, 3, 4].map(i => <ThumbnailSkeleton key={i} aspect="4/3" />)}</div>}
        empty={<EmptyState title="No pictures in this folder">Raws, JPEGs and panoramas show up here; check the folder holds some.</EmptyState>}
      >
        {rows => <ul aria-label="Pictures" className={grid}>{rows.filter(i => videos || i.kind !== 'video').map((i, n) => <li key={i.id} className="grid"><Thumb project={project} item={i} eager={n < 10} /></li>)}</ul>}
      </DataView>
    </section>
  )
}
